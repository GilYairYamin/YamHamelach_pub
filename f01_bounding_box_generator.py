import os
from tqdm import tqdm
import cv2
from argparse import ArgumentParser
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
from ultralytics import YOLO
import json

MODEL_INPUT_SHAPE = (640, 640)
PATCH_CLS_NAME = 'patch'


def center(box, dtype=None):
    l, t, r, b = box
    center = np.array([(r + l) / 2, (b + t) / 2])
    if dtype is not None:
        center = center.astype(dtype)
    return center


def center_radius(box):
    return np.linalg.norm(center(box))


class PatchFinder():
    def __init__(self,cp):
        self._model = YOLO(cp)
        self._im_fn = None
        self.im = None
        self._extracted_boxes = None
        self._id_map = None
        self._tags = []
        self._patch_info = {}

    @property
    def id_map(self):
        if self._id_map is None or self._id_map.shape[:2] != self.im.shape[:2]:
            cols, rows = (32, 32)
            ids = np.arange(rows * cols)
            self._id_map = ids.reshape((cols, rows))
            self._id_map = cv2.resize(self._id_map, self.im.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
        return self._id_map

    def predict_bounding_box(self):
        im = cv2.resize(self.im, MODEL_INPUT_SHAPE)
        results = self._model.predict([im], verbose=False)
        names = results[0].names
        boxes = results[0].boxes
        cls = boxes.cls.numpy()

        self._images = []
        extracted_boxes = []
        for (i, box) in enumerate(boxes):
            if names[cls[i]] != PATCH_CLS_NAME:
                continue
            l, t, r, b = box.xyxy.numpy()[0]
            im_shape = self.im.shape
            l = int(l / MODEL_INPUT_SHAPE[1] * im_shape[1])
            r = int(r / MODEL_INPUT_SHAPE[1] * im_shape[1])
            t = int(t / MODEL_INPUT_SHAPE[0] * im_shape[0])
            b = int(b / MODEL_INPUT_SHAPE[0] * im_shape[0])
            extracted_boxes.append([l, t, r, b])

        self._extracted_boxes = sorted(extracted_boxes, key=center_radius)

        for l, t, r, b in self._extracted_boxes:
            self._images.append(self.im[t:b, l:r])

    def load_image(self, fn):
        self._im_fn = fn
        self.im = cv2.imread(fn)
        self._tags = []
        self._patch_info = {}

    @property
    def tags(self):
        if self._tags == []:
            for (i, box) in enumerate(self._extracted_boxes):
                center_x, center_y = center(box, dtype=np.uint(16))
                tag = self.id_map[center_y][center_x]
                if tag in self._tags:
                    count = len([t for t in self._tags if int(t) == tag])
                    tag = tag + count / 10
                self._tags.append(tag)
        return self._tags

    def _generate_image_with_detection(self):
        im = np.array(self.im)
        for (i, box) in enumerate(self._extracted_boxes):
            (l, t, r, b) = box
            c = np.random.randint(0, 125, 3)
            im = cv2.rectangle(im, (l, t), (r, b), c.tolist(), 10)
            tag = self.tags[i]
            im = cv2.putText(im, str(tag), (int((r + l) / 2), int((b + t) / 2)),
                             cv2.FONT_HERSHEY_SIMPLEX, 2, c.tolist(), 5)
        return im

    def show_image(self):
        im = self._generate_image_with_detection()
        plt.imshow(im)
        plt.show()

    def save_image(self, fn):
        os.makedirs(fn.parent, exist_ok=True)
        im = self._generate_image_with_detection()
        plt.imsave(fn, im)

    def save_patches(self, path):
        os.makedirs(path, exist_ok=True)
        im = np.array(self.im)
        for (i, box) in enumerate(self._extracted_boxes):
            (l, t, r, b) = box
            patch = im[t:b, l:r]
            tag = self.tags[i]
            fn = Path(path, f"{Path(self._im_fn).stem}_{tag}").with_suffix(".jpg")
            plt.imsave(fn, patch)

            # Store patch information
            self._patch_info[str(tag)] = {
                "filename": fn.name,
                "coordinates": [l, t, r, b]
            }

    def save_patch_info(self, path):
        info_file = Path(path, f"{Path(self._im_fn).stem}_patch_info.json")
        with open(info_file, 'w') as f:
            json.dump(self._patch_info, f, indent=2)


from dotenv import load_dotenv
import os

# Load the .env file
load_dotenv()
base_path = os.getenv('BASE_PATH')


IMAGES_IN = os.path.join(base_path, os.getenv('IMAGES_IN'))
PATCHES_DIR = os.path.join(base_path, os.getenv('PATCHES_IN'))
BBOX_DIR = os.path.join(base_path, os.getenv('BBOXES_IN'))
MODEL_NN_WEIGHTS = os.path.join(base_path, os.getenv('MODEL_NN_WEIGHTS'))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--im_path", help="paths to input images containing multiple patches", default=IMAGES_IN)
    parser.add_argument('--patches_dir', help="path to save images with bounding boxes and patches crops", default=PATCHES_DIR)
    parser.add_argument('--bbox_dir', help="path to save bounding box images", default=BBOX_DIR)

    parser.add_argument('--cp', help="yolov8 cp path", default=MODEL_NN_WEIGHTS)
    args = parser.parse_args()

    im_path = args.im_path
    patches_dir = args.patches_dir
    os.makedirs(patches_dir, exist_ok=True)

    bbox_dir = args.bbox_dir
    os.makedirs(bbox_dir, exist_ok=True)


    patch_finder = PatchFinder(args.cp)
    paths = list(Path(im_path).glob("*.jpg"))
    pbar = tqdm(paths, desc='Processing images')

    for (i, im_fn) in enumerate(pbar):
        pbar.set_description(f"{im_fn} {i + 1}/{len(paths)}")
        patch_finder.load_image(str(im_fn))
        patch_finder.predict_bounding_box()

        fn = Path(BBOX_DIR , im_fn.name)
        patch_finder.save_image(fn)

        fp = Path(PATCHES_DIR,  im_fn.stem)
        patch_finder.save_patches(fp)

        # Save patch information
        patch_finder.save_patch_info(fp)