import os
from tqdm import tqdm

import cv2
from argparse import ArgumentParser
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
from ultralytics import YOLO

MODEL_INPUT_SHAPE = (640, 640)


'''
This module gets:
A. path to images containing multiple patches
B. output path
C. It runs a trained yolov8 model to detect pathes bounding boxes
D. Generates: 
    a. New images with bounding boxes, and mapping identifier index
    b. crops of all patches in with name convention contains original file_name and identifier index in:
        <output path>/patches/ 
  
'''

# default yolov8 checkpoint, please edit before run
cp = '/home/avinoam/workspace/YAM_HAMELACH/weights/train5/weights/best.pt'

PATCH_CLS_NAME = 'patch'

def center(box, dtype=None ):
    l,t,r,b = box
    center = np.array([(r+l)/2,(b+t)/2])
    if dtype is not None:
        center = center.astype(dtype)
    return center

def center_radius(box):
    '''
    calculates the distance from image top-left to box center
    '''
    return np.linalg.norm(center(box))


class PatchFinder():
    def __init__(self):
        self._model = YOLO(cp)
        self._im_fn = None
        self.im = None
        self._bb = None

        self._id_map = None
        self._tags = []

    @property
    def id_map(self):
        # prepare indexing matrix of row*col to generate unique id for each patch,
        # assuming that they don't fall into same box
        if self._id_map is None or self._id_map.shape[:2] != self.im.shape[:2]:
            cols, rows  = (32,32)
            ids = np.arange(rows*cols)
            self._id_map = ids.reshape((cols, rows))
            self._id_map = cv2.resize(self._id_map, self.im.shape[:2][::-1] , interpolation=cv2.INTER_NEAREST)
            # plt.imshow(self._id_map)
            # plt.show()
        return self._id_map

    def predict_bounding_box(self):
        # applying trained yolo on image to generate bounding boxes
        im = cv2.resize(self.im, MODEL_INPUT_SHAPE)
        results = self._model.predict([im], verbose=False)
        names = results[0].names
        boxes = results[0].boxes
        cls = boxes.cls.numpy()

        # iterating over boxes,
        # rescaling boxes to original image scale, and keep top,left,right,bottom into: self._extracted_boxes[]
        # cropping patch from the image. and keep croped image into: self._images[]
        self._images = []
        extracted_boxes = []
        for (i,box) in enumerate(boxes):
            if names[cls[i]] != PATCH_CLS_NAME:
                continue
            l, t, r, b = box.xyxy.numpy()[0]
            im_shape = self.im.shape
            l = int(l/MODEL_INPUT_SHAPE[1]*im_shape[1])
            r = int(r/MODEL_INPUT_SHAPE[1]*im_shape[1])
            t = int(t/MODEL_INPUT_SHAPE[0]*im_shape[0])
            b = int(b/MODEL_INPUT_SHAPE[0]*im_shape[0])
            extracted_boxes.append([l,t,r,b])

        self._extracted_boxes = extracted_boxes
        self._extracted_boxes = sorted(self._extracted_boxes, key=center_radius)

        for l,t,r,b in self._extracted_boxes:
            self._images.append(self.im[t:b,l:r])

        DISPLAY = False
        if DISPLAY:
            for i in range(25):
                if i == len(self._images):
                    break
                plt.subplot(5,5,i+1)
                plt.imshow(self._images[i])
            plt.show()

    def load_image(self, fn):
        self._im_fn = fn
        self.im = cv2.imread(fn)
        self._tags = []

    @property
    def tags(self):
        '''
        generate list of tags according to indexing policy,
        the list map each patch in:  self._extracted_boxes[i]
        to unique tag in:            self._tags[i]
        '''
        if self._tags == []:
            for (i, box) in enumerate(self._extracted_boxes):
                center_x, center_y = center(box, dtype=np.uint(16))
                tag = self.id_map[center_y][center_x]
                if tag in self._tags:
                    count = len([t for t in self._tags if int(t)==tag])
                    tag = tag + count/10
                self._tags.append(tag)
        return self._tags

    def _generate_image_with_detection(self):
        '''
        generate image with rectangles over bounding-boxes, and identifier in the center, for visualization
        '''
        im = np.array(self.im)
        for (i, box) in enumerate(self._extracted_boxes):
            (l, t, r, b) = box
            c = np.random.randint(0, 125, 3)
            im = cv2.rectangle(im, (l, t), (r, b), c.tolist(), 10)
            center_x, center_y = center(box, dtype=np.uint(16))
            # tag = self.id_map[center_y][center_x]
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
        '''
        iterates over patches and save croped image into <path>/<filename_convention>.jpg>
        '''
        os.makedirs(path, exist_ok=True)
        im = np.array(self.im)
        for (i, box) in enumerate(self._extracted_boxes):
            (l, t, r, b) = box
            patch = im[t:b,l:r]
            tag = self.tags[i]
            fn = Path(path, f"{fp.name}_{tag}").with_suffix(".jpg")
            plt.imsave(fn, patch)

    def threshold(self):
        raise Exception("I do not think that anyone uses this.. feel free to remove :)")
        img = self.im
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        th = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,155,10)
        plt.imshow(th)
        plt.show()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--im_path", help="paths to input images containing multiple patches")
    parser.add_argument('--out_path', help="path to save images with bounding boxes and patches crops")
    parser.add_argument('--cp', help="yolov8 cp path", default=cp)
    args = parser.parse_args()

    im_path = args.im_path
    out_path = args.out_path
    os.makedirs(out_path, exist_ok=True)


    patch_finder = PatchFinder()
    paths = list(Path(im_path).glob("*.jpg"))
    pbar = tqdm(paths, desc='description')

    for (i,im_fn) in enumerate(pbar): # tqdm():
        pbar.set_description(f"{im_fn} {i}/{len(paths)}")
        patch_finder.load_image(str(im_fn))
        patch_finder.predict_bounding_box()

        # patch_finder.show_image()

        fn = Path(out_path, "bounding_boxes",im_fn.name)
        patch_finder.save_image(fn)
        fp = Path(out_path, "patches", im_fn.stem)
        patch_finder.save_patches(fp)

