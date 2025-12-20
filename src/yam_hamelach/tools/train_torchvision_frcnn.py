import torch
import argparse
import os
import numpy as np
import yaml
import random
import glob
from tqdm import tqdm
import torchvision
from data import DL_Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from history import MetricsTracker, ExperimentConfig

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def collate_function(data):
    return tuple(zip(*data))


def train(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    # Initialize experiment
    experiment_config = ExperimentConfig(config)
    metrics_tracker = MetricsTracker(experiment_config)
    dataset_config = config['dataset_params']
    train_config = config['train_params']
    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    experiment_config.log_message(f"Using device: {device}")
    experiment_config.log_message(f"Random seed: {seed}")

    # Load training data
    train_data = DL_Dataset(im_dir=dataset_config['im_train_path'], ann_dir=dataset_config['ann_train_path'])
    train_dataset = DataLoader(train_data, batch_size=train_config['batch_size'], shuffle=True, num_workers=4, collate_fn=collate_function)

    # Initialize model
    faster_rcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=not args.resume, min_size=600, max_size=1000)
    faster_rcnn_model.roi_heads.box_predictor = FastRCNNPredictor(faster_rcnn_model.roi_heads.box_predictor.cls_score.in_features, num_classes=2)
    experiment_config.log_message("Using Faster R-CNN with ResNet50-FPN backbone")
    faster_rcnn_model.train()
    faster_rcnn_model.to(device)

    # Initialize optimizer
    optimizer = torch.optim.SGD(lr=train_config['lr'], params=filter(lambda p: p.requires_grad, faster_rcnn_model.parameters()),
                                weight_decay=train_config['w_decay'], momentum=train_config['momentum'])

    # Check for existing checkpoint
    start_epoch = 0
    if args.resume:
        checkpoint_name = 'frcnn_r50fpn_'
        checkpoint_pattern = os.path.join(args.resume_dir, f'{checkpoint_name}epoch_*.pth')
        checkpoints = glob.glob(checkpoint_pattern)
        if checkpoints:
            # Get the latest checkpoint
            latest_checkpoint = max(checkpoints, key=os.path.getctime)
            checkpoint = torch.load(latest_checkpoint, map_location=device)
            faster_rcnn_model.load_state_dict(checkpoint)
            
            # Extract epoch number from checkpoint filename
            start_epoch = int(latest_checkpoint.split('_')[-1].split('.')[0]) + 1
            experiment_config.log_message(f"Resuming training from checkpoint: {latest_checkpoint}")
            experiment_config.log_message(f"Starting from epoch: {start_epoch}")
        else:
            experiment_config.log_message("No checkpoint found in specified directory. Starting from scratch.")

    num_epochs = train_config['num_epochs']

    for i in range(start_epoch, start_epoch+num_epochs):
        print(f"Starting epoch {i+1}/{num_epochs}")
        experiment_config.log_message(f"Starting epoch {i+1}/{num_epochs}")
        rpn_classification_losses = []
        rpn_localization_losses = []
        frcnn_classification_losses = []
        frcnn_localization_losses = []
        for ims, targets, path in tqdm(train_dataset):
            optimizer.zero_grad()
            for target in targets:
                target['boxes'] = target['bboxes'].float().to(device)
                del target['bboxes']
                target['labels'] = target['labels'].long().to(device)
            images = [im.float().to(device) for im in ims]
            batch_losses = faster_rcnn_model(images, targets)
            loss = batch_losses['loss_classifier']
            loss += batch_losses['loss_box_reg']
            loss += batch_losses['loss_rpn_box_reg']
            loss += batch_losses['loss_objectness']

            rpn_classification_losses.append(batch_losses['loss_objectness'].item())
            rpn_localization_losses.append(batch_losses['loss_rpn_box_reg'].item())
            frcnn_classification_losses.append(batch_losses['loss_classifier'].item())
            frcnn_localization_losses.append(batch_losses['loss_box_reg'].item())

            loss.backward()
            optimizer.step()

        # Calculate average losses for the epoch
        avg_rpn_class_loss = np.mean(rpn_classification_losses)
        avg_rpn_box_loss = np.mean(rpn_localization_losses)
        avg_det_class_loss = np.mean(frcnn_classification_losses)
        avg_det_box_loss = np.mean(frcnn_localization_losses)
        total_loss = avg_rpn_class_loss + avg_rpn_box_loss + avg_det_class_loss + avg_det_box_loss

        # Update metrics tracker
        metrics = {
            'rpn_class_loss': avg_rpn_class_loss,
            'rpn_bbox_loss': avg_rpn_box_loss,
            'det_class_loss': avg_det_class_loss,
            'det_bbox_loss': avg_det_box_loss,
            'total_loss': total_loss,
        }
        metrics_tracker.update(metrics)
        tot_loss_output = 'Total Loss : {:.4f}'.format(total_loss)
        rpn_loss_output = 'RPN | Classification Loss : {:.4f}'.format(avg_rpn_class_loss)
        rpn_loss_output += ' | Localization Loss : {:.4f}'.format(avg_rpn_box_loss)
        det_loss_output = 'Detection | Classification Loss : {:.4f}'.format(avg_det_class_loss)
        det_loss_output += ' | Localization Loss : {:.4f}'.format(avg_det_box_loss)
        experiment_config.log_message(tot_loss_output)
        experiment_config.log_message(rpn_loss_output)
        experiment_config.log_message(det_loss_output)
        print(tot_loss_output + '\n' + rpn_loss_output + '\n' + det_loss_output)

        # Save checkpoint every 5 epochs
        if (i - start_epoch) % 5 == 4:
            checkpoint_name = 'frcnn_r50fpn_'
            checkpoint_path = os.path.join(experiment_config.checkpoints_dir, f'{checkpoint_name}epoch_{i}.pth')
            torch.save(faster_rcnn_model.state_dict(), checkpoint_path)
            experiment_config.log_message(f"Saved checkpoint: {checkpoint_path}")

    metrics_tracker.plot_metrics(num_epochs-1)
    experiment_config.log_message("Training completed successfully")
    print('Training completed successfully')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for faster rcnn using torchvision code training')
    parser.add_argument('--config', dest='config_path', default='config/q1.yaml', type=str)
    parser.add_argument('--resume', dest='resume', default=False, type=bool, help='Resume training from a checkpoint')
    parser.add_argument('--resume_dir', dest='resume_dir', default='model', help='Directory containing the checkpoint to resume from')
    args = parser.parse_args()
    train(args)