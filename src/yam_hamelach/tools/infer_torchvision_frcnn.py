import torch
import numpy as np
import cv2
import torchvision
import argparse
import random
import os
import yaml
from tqdm import tqdm
from data import DL_Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from history import plot_precision_recall_curve
import glob

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_iou(det, gt):
    det_x1, det_y1, det_x2, det_y2 = det
    gt_x1, gt_y1, gt_x2, gt_y2 = gt

    x_left = max(det_x1, gt_x1)
    y_top = max(det_y1, gt_y1)
    x_right = min(det_x2, gt_x2)
    y_bottom = min(det_y2, gt_y2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    area_intersection = (x_right - x_left) * (y_bottom - y_top)
    det_area = (det_x2 - det_x1) * (det_y2 - det_y1)
    gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
    area_union = float(det_area + gt_area - area_intersection + 1E-6)
    iou = area_intersection / area_union
    return iou


def compute_map(det_boxes, gt_boxes, iou_threshold=0.5, method='area'):
    # det_boxes = [
    #   {
    #       'parchment' : [[x1, y1, x2, y2, score], ...]
    #   }
    #   {det_boxes_img_2},
    #   ...
    #   {det_boxes_img_N},
    # ]
    #
    # gt_boxes = [
    #   {
    #       'parchment' : [[x1, y1, x2, y2, score], ...]
    #   },
    #   {gt_boxes_img_2},
    #   ...
    #   {gt_boxes_img_N},
    # ]

    gt_labels = {cls_key for im_gt in gt_boxes for cls_key in im_gt.keys()}
    gt_labels = sorted(gt_labels)
    all_aps = {}
    # all_aps = {
    #     'label1': ap_value,
    #     'label2': ap_value,
    #     ...
    # }
    all_precisions = {}
    all_recalls = {}

    # average precisions for ALL classes
    aps = []
    for idx, label in enumerate(gt_labels):
        # Get detection predictions of this class
        cls_dets = [
            [im_idx, im_dets_label] for im_idx, im_dets in enumerate(det_boxes)
            if label in im_dets for im_dets_label in im_dets[label]
        ]

        # cls_dets = [
        #   (0, [x1_0, y1_0, x2_0, y2_0, score_0]),
        #   ...
        #   (0, [x1_M, y1_M, x2_M, y2_M, score_M]),
        #   (1, [x1_0, y1_0, x2_0, y2_0, score_0]),
        #   ...
        #   (1, [x1_N, y1_N, x2_N, y2_N, score_N]),
        #   ...
        # ]

        # Sort them by confidence score
        cls_dets = sorted(cls_dets, key=lambda k: -k[1][-1])

        # For tracking which gt boxes of this class have already been matched
        gt_matched = [[False for _ in im_gts[label]] for im_gts in gt_boxes]
        # Number of gt boxes for this class for recall calculation
        num_gts = sum([len(im_gts[label]) for im_gts in gt_boxes])
        tp = [0] * len(cls_dets)
        fp = [0] * len(cls_dets)

        # For each prediction
        for det_idx, (im_idx, det_pred) in enumerate(cls_dets):
            # Get gt boxes for this image and this label
            im_gts = gt_boxes[im_idx][label]
            max_iou_found = -1
            max_iou_gt_idx = -1

            # Get best matching gt box
            for gt_box_idx, gt_box in enumerate(im_gts):
                gt_box_iou = get_iou(det_pred[:-1], gt_box)
                if gt_box_iou > max_iou_found:
                    max_iou_found = gt_box_iou
                    max_iou_gt_idx = gt_box_idx
            # TP only if IoU >= threshold and this gt has not yet been matched
            if max_iou_found < iou_threshold or gt_matched[im_idx][max_iou_gt_idx]:
                fp[det_idx] = 1
            else:
                tp[det_idx] = 1
                # If tp then we set this gt box as matched
                gt_matched[im_idx][max_iou_gt_idx] = True
        
        # Cumulative tp and fp
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)

        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts, eps)
        precisions = tp / np.maximum((tp + fp), eps)

        all_precisions[label] = precisions
        all_recalls[label] = recalls

        if method == 'area':
            recalls = np.concatenate(([0.0], recalls, [1.0]))
            precisions = np.concatenate(([0.0], precisions, [0.0]))

            # Replace precision values with recall r with maximum precision value
            # of any recall value >= r
            # This computes the precision envelope
            for i in range(precisions.size - 1, 0, -1):
                precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
            # For computing area, get points where recall changes value
            i = np.where(recalls[1:] != recalls[:-1])[0]
            # Add the rectangular areas to get ap
            ap = np.sum((recalls[i + 1] - recalls[i]) * precisions[i + 1])
        elif method == 'interp':
            ap = 0.0
            for interp_pt in np.arange(0, 1 + 1E-3, 0.1):
                # Get precision values for recall values >= interp_pt
                prec_interp_pt = precisions[recalls >= interp_pt]

                # Get max of those precision values
                prec_interp_pt = prec_interp_pt.max() if prec_interp_pt.size > 0.0 else 0.0
                ap += prec_interp_pt
            ap = ap / 11.0
        else:
            raise ValueError('Method can only be area or interp')
        if num_gts > 0:
            aps.append(ap)
            all_aps[label] = ap
        else:
            all_aps[label] = np.nan
    # compute mAP
    mean_ap = sum(aps) / len(aps)
    return mean_ap, all_aps, all_precisions, all_recalls


def load_model_and_dataset(args):
    # Read the config file
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']

    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)

    dataset = DL_Dataset(im_dir=dataset_config['im_test_path'], ann_dir=dataset_config['ann_test_path'])
    test_dataset = DataLoader(dataset, batch_size=1, shuffle=False)

    faster_rcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, min_size=600, max_size=1000, box_score_thresh=0.5)
    faster_rcnn_model.roi_heads.box_predictor = FastRCNNPredictor(faster_rcnn_model.roi_heads.box_predictor.cls_score.in_features, num_classes=2)
    faster_rcnn_model.eval()
    faster_rcnn_model.to(device)
    
    # Load the weights
    checkpoint_name = 'frcnn_r50fpn_'
    checkpoint_pattern = os.path.join(model_config['weights_dir'], f'{checkpoint_name}epoch_*.pth')
    checkpoints = glob.glob(checkpoint_pattern)
    if not checkpoints:
        raise ImportError(f"No weights found at {model_config['weights_dir']} to load")
    # Get the latest checkpoint
    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    faster_rcnn_model.load_state_dict(torch.load(latest_checkpoint, map_location=device))
    return faster_rcnn_model, dataset, test_dataset


def infer(args):
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    faster_rcnn_model, dataset, test_dataset = load_model_and_dataset(args)

    random_idxs = random.sample(list(range(len(dataset))), min(10, len(dataset)))
    for random_idx in random_idxs:
        im, target, fname = dataset[random_idx]
        im = im.unsqueeze(0).float().to(device)

        gt_im = cv2.imread(fname)
        gt_im_copy = gt_im.copy()

        # Saving images with ground truth boxes
        for bidx, box in enumerate(target['bboxes']):
            x1, y1, x2, y2 = box.detach().cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            cv2.rectangle(gt_im, (x1, y1), (x2, y2), thickness=2, color=[0, 255, 0])
            cv2.rectangle(gt_im_copy, (x1, y1), (x2, y2), thickness=2, color=[0, 255, 0])
            text = dataset.idx2label[target['labels'][bidx].detach().cpu().item()]
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 1)
            text_w, text_h = text_size
            cv2.rectangle(gt_im_copy, (x1, y1), (x1 + 10 + text_w, y1 + 10 + text_h), [255, 255, 255], -1)
            cv2.putText(gt_im, text=text, org=(x1 + 5, y1 + 15), thickness=1, fontScale=1, color=[0, 0, 0], fontFace=cv2.FONT_HERSHEY_PLAIN)
            cv2.putText(gt_im_copy, text=text, org=(x1 + 5, y1 + 15), thickness=1, fontScale=1, color=[0, 0, 0], fontFace=cv2.FONT_HERSHEY_PLAIN)
        cv2.addWeighted(gt_im_copy, 0.7, gt_im, 0.3, 0, gt_im)
        cv2.imwrite('{}/output_frcnn_gt_{}.png'.format(output_dir, random_idx), gt_im)

        # Getting predictions from trained model
        frcnn_output = faster_rcnn_model(im, None)[0]
        boxes = frcnn_output['boxes']
        labels = frcnn_output['labels']
        scores = frcnn_output['scores']
        im = cv2.imread(fname)
        im_copy = im.copy()

        # Saving images with predicted boxes
        for bidx, box in enumerate(boxes):
            x1, y1, x2, y2 = box.detach().cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(im, (x1, y1), (x2, y2), thickness=2, color=[0, 0, 255])
            cv2.rectangle(im_copy, (x1, y1), (x2, y2), thickness=2, color=[0, 0, 255])
            text = '{} : {:.2f}'.format(dataset.idx2label[labels[bidx].detach().cpu().item()],
                                        scores[bidx].detach().cpu().item())
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 1)
            text_w, text_h = text_size
            cv2.rectangle(im_copy, (x1, y1), (x1 + 10 + text_w, y1 + 10 + text_h), [255, 255, 255], -1)
            cv2.putText(im, text=text, org=(x1 + 5, y1 + 15), thickness=1, fontScale=1, color=[0, 0, 0], fontFace=cv2.FONT_HERSHEY_PLAIN)
            cv2.putText(im_copy, text=text, org=(x1 + 5, y1 + 15), thickness=1, fontScale=1, color=[0, 0, 0], fontFace=cv2.FONT_HERSHEY_PLAIN)
        cv2.addWeighted(im_copy, 0.7, im, 0.3, 0, im)
        cv2.imwrite('{}/output_frcnn_{}.jpg'.format(output_dir, random_idx), im)


def evaluate_map(args):
    # Read the config file
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    eval_config = config['eval_params']
    
    faster_rcnn_model, dataset, test_dataset = load_model_and_dataset(args)
    gts = []
    preds = []
    
    for im, target, fname in tqdm(test_dataset):
        im = im.float().to(device)
        target_boxes = target['bboxes'].float().to(device)[0]
        target_labels = target['labels'].long().to(device)[0]
        
        frcnn_output = faster_rcnn_model(im, None)[0]

        boxes = frcnn_output['boxes']
        labels = frcnn_output['labels']
        scores = frcnn_output['scores']

        pred_boxes = {}
        gt_boxes = {}
        for label_name in dataset.label2idx:
            pred_boxes[label_name] = []
            gt_boxes[label_name] = []

        for bidx, box in enumerate(boxes):
            x1, y1, x2, y2 = box.detach().cpu().numpy()
            label = labels[bidx].detach().cpu().item()
            score = scores[bidx].detach().cpu().item()
            label_name = dataset.idx2label[label]
            pred_boxes[label_name].append([x1, y1, x2, y2, score])
        for bidx, box in enumerate(target_boxes):
            x1, y1, x2, y2 = box.detach().cpu().numpy()
            label = target_labels[bidx].detach().cpu().item()
            label_name = dataset.idx2label[label]
            gt_boxes[label_name].append([x1, y1, x2, y2])

        gts.append(gt_boxes)
        preds.append(pred_boxes)

    mean_ap, all_aps, all_precisions, all_recalls = compute_map(preds, gts, iou_threshold=eval_config['map_iou_thresh'], method='area')
    plot_precision_recall_curve(all_aps['parchment'], all_precisions['parchment'], all_recalls['parchment'], 'interp', 'parchment', )
    print('Class Wise Average Precisions')
    for idx in range(len(dataset.idx2label)):
        print('AP for class {} = {:.4f}'.format(dataset.idx2label[idx], all_aps[dataset.idx2label[idx]]))
    print('Overall Average Precision : {:.4f}'.format(mean_ap))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for inference using torchvision code faster rcnn')
    parser.add_argument('--config', dest='config_path', default='config/voc.yaml', type=str)
    parser.add_argument('--evaluate', dest='evaluate', default=False, type=bool)
    parser.add_argument('--infer_samples', dest='infer_samples', default=False, type=bool)
    parser.add_argument('--samples_dest', dest='output_dir', default='results/samples', type=str)
    args = parser.parse_args()
    
    if args.infer_samples:
        infer(args)
    else:
        print('Not Inferring for samples as `infer_samples` argument is False')

    if args.evaluate:
        evaluate_map(args)
    else:
        print('Not Evaluating as `evaluate` argument is False')