import glob
import os
import json
import numpy as np
from PIL import Image, ImageOps
import torch
import torchvision
from tqdm import tqdm
from torch.utils.data.dataset import Dataset


def load_images_and_anns(im_dir, ann_dir):
    """
    Method to get the labelme files and for each file get all the objects and their
    ground truth detection information for the dataset
    Args:
        im_dir: Path of the images
        ann_dir: Path of annotation (json) files
    """
    im_infos = []

    for ann_file in tqdm(glob.glob(os.path.join(ann_dir, '*.json'))):
        # Loading annotations
        with open(ann_file, 'r') as f:
            annotations = json.load(f)
       
        # Convert annotations
        im_info = {}
        im_info['img_id'] = os.path.basename(ann_file).split('.json')[0]
        im_info['filename'] = os.path.join(im_dir, annotations['imagePath'])
        im_info['width'] = annotations['imageWidth']
        im_info['height'] = annotations['imageHeight']
        detections = []
        for shape in annotations['shapes']:
            det = {}
            label = 1   # Only one class -parchment- in our data
            points = np.array(shape['points'])
            x_min, y_min = points.min(axis=0)
            x_max, y_max = points.max(axis=0)
            bbox = [x_min, y_min, x_max, y_max]
            det['label'] = label
            det['bbox'] = bbox
            detections.append(det)
        im_info['detections'] = detections
        im_infos.append(im_info)
    print('Total {} images found at {}'.format(len(im_infos), im_dir))
    return im_infos


class DL_Dataset(Dataset):
    def __init__(self, im_dir, ann_dir):
        self.im_dir = im_dir
        self.ann_dir = ann_dir
        classes = ['background', 'parchment']
        self.label2idx = {classes[idx]: idx for idx in range(len(classes))}
        self.idx2label = {idx: classes[idx] for idx in range(len(classes))}
        self.images_info = load_images_and_anns(im_dir, ann_dir)
    
    def __len__(self):
        return len(self.images_info)
    
    def __getitem__(self, index):
        im_info = self.images_info[index]
        im = Image.open(im_info['filename'])
        im = ImageOps.exif_transpose(im)
        
        # Convert to tensor
        im_tensor = torchvision.transforms.ToTensor()(im)
        
        # Get all GT boundary boxes
        targets = {}
        targets['bboxes'] = torch.as_tensor([detection['bbox'] for detection in im_info['detections']])
        targets['labels'] = torch.as_tensor([detection['label'] for detection in im_info['detections']])
        
        return im_tensor, targets, im_info['filename']
        