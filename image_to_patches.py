import cv2
from matplotlib import pyplot as plt
import numpy as np
from ultralytics import YOLO
from pathlib import Path
MODEL_INPUT_SHAPE = (640, 640)
PATCH_CLS_NAME = 'patch'

im_path = "/home/avinoam/workspace/YAM_HAMELACH/dataset/"
# cp = '/home/avinoam/Desktop/autobrains/DL_Engineer/assignment_files/runs/detect/train16/weights/best.pt'
cp = '/home/avinoam/workspace/YAM_HAMELACH/weights/train5/weights/best.pt'

def center(box, dtype=None ):
    l,t,r,b = box
    center = np.array([(r+l)/2,(b+t)/2])
    if dtype is not None:
        center = center.astype(dtype)
    return center

def center_radius(box):
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
    def id_map(self): # a discrete ids map
        if self._id_map is None or self._id_map.shape[:2] != self.im.shape[:2]:
            cols, rows  = (32,32)
            ids = np.arange(rows*cols)
            self._id_map = ids.reshape((cols, rows))
            self._id_map = cv2.resize(self._id_map, self.im.shape[:2][::-1] , interpolation=cv2.INTER_NEAREST)
            # plt.imshow(self._id_map)
            # plt.show()
        return self._id_map

    def predict_bounding_box(self):
        im = cv2.resize(self.im, MODEL_INPUT_SHAPE)
        results = self._model.predict([im], verbose=False)
        names = results[0].names
        boxes = results[0].boxes
        cls = boxes.cls.numpy()
        images = []
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
            images.append(self.im[t:b,l:r])

        DISPLAY = False
        if DISPLAY:
            for i in range(25):
                if i == len(images):
                    break
                plt.subplot(5,5,i+1)
                plt.imshow(images[i])
            plt.show()

    def load_image(self, fn):
        self._im_fn = fn
        self.im = cv2.imread(fn)
        self._tags = []

    @property
    def tags(self):
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
        im = self._generate_image_with_detection()
        plt.imsave(fn, im)

    def threshold(self):
        img = self.im
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        th = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,155,10)
        plt.imshow(th)
        plt.show()

if __name__ == "__main__":
    out_path = "/home/avinoam/workspace/YAM_HAMELACH/results/17_05/"

    patch_finder = PatchFinder()
    for im_fn in Path(im_path).glob("*.jpg"):
        patch_finder.load_image(str(im_fn))
        # patch_finder.id_map

        patch_finder.predict_bounding_box()

        # patch_finder.show_image()

        fn = Path(out_path, im_fn.name)
        patch_finder.save_image(fn)
    # patch_finder.threshold()

