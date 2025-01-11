import json
import os
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from tqdm import tqdm
from ultralytics import YOLO


def center(box, dtype=None):
    left, top, right, bottom = box
    center = np.array([(right + left) / 2, (bottom + top) / 2])
    if dtype is not None:
        center = center.astype(dtype)
    return center


def center_radius(box):
    return np.linalg.norm(center(box))


class PatchFinder:
    def __init__(self, cp):
        self.image = None
        self._model = YOLO(cp)
        self._img_filename = None
        self._extracted_boxes = None
        self._id_map = None
        self._tags = []
        self._patch_info = {}

    @property
    def id_map(self):
        if (
            self._id_map is None
            or self._id_map.shape[:2] != self.image.shape[:2]
        ):
            cols, rows = (32, 32)
            ids = np.arange(rows * cols)
            self._id_map = ids.reshape((cols, rows))
            self._id_map = cv2.resize(
                self._id_map,
                self.image.shape[:2][::-1],
                interpolation=cv2.INTER_NEAREST,
            )
        return self._id_map

    def predict_bounding_box(
        self,
        model_input_shape: tuple[int, ...] = (640, 640),
        patch_cls_name: str = "patch",
    ):
        img = cv2.resize(self.image, model_input_shape)
        results = self._model.predict([img], verbose=False)
        names = results[0].names
        boxes = results[0].boxes
        cls = boxes.cls.numpy()

        self._images = []
        extracted_boxes = []
        for i, box in enumerate(boxes):
            if names[cls[i]] != patch_cls_name:
                continue

            left, top, right, bottom = box.xyxy.numpy()[0]
            im_shape = self.image.shape
            left = int(left / model_input_shape[1] * im_shape[1])
            right = int(right / model_input_shape[1] * im_shape[1])
            top = int(top / model_input_shape[0] * im_shape[0])
            bottom = int(bottom / model_input_shape[0] * im_shape[0])
            extracted_boxes.append([left, top, right, bottom])

        self._extracted_boxes = sorted(extracted_boxes, key=center_radius)

        for left, top, right, bottom in self._extracted_boxes:
            self._images.append(self.image[top:bottom, left:right])

    def load_image(self, img_filename):
        self._img_filename = img_filename
        self.image = cv2.imread(img_filename)
        self._tags = []
        self._patch_info = {}

    @property
    def tags(self):
        if self._tags == []:
            for i, box in enumerate(self._extracted_boxes):
                center_x, center_y = center(box, dtype=np.uint(16))
                tag = self.id_map[center_y][center_x]
                if tag in self._tags:
                    count = len([t for t in self._tags if int(t) == tag])
                    tag = tag + count / 10
                self._tags.append(tag)
        return self._tags

    def _generate_image_with_detection(self):
        im = np.array(self.image)
        for i, box in enumerate(self._extracted_boxes):
            (left, top, right, bottom) = box
            c = np.random.randint(0, 125, 3)
            im = cv2.rectangle(
                im, (left, top), (right, bottom), c.tolist(), 10
            )
            tag = self.tags[i]
            im = cv2.putText(
                im,
                str(tag),
                (int((right + left) / 2), int((bottom + top) / 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                c.tolist(),
                5,
            )
        return im

    def show_image(self):
        img = self._generate_image_with_detection()
        plt.imshow(img)
        plt.show()

    def save_image(self, fn):
        os.makedirs(fn.parent, exist_ok=True)
        img = self._generate_image_with_detection()
        plt.imsave(fn, img)

    def save_patches(self, path):
        os.makedirs(path, exist_ok=True)
        img = np.array(self.image)
        for i, box in enumerate(self._extracted_boxes):
            (left, top, right, bottom) = box
            patch = img[top:bottom, left:right]
            tag = self.tags[i]
            filename = Path(
                path, f"{Path(self._img_filename).stem}_{tag}"
            ).with_suffix(".jpg")
            plt.imsave(filename, patch)

            # Store patch information
            self._patch_info[str(tag)] = {
                "filename": filename.name,
                "coordinates": [left, top, right, bottom],
            }

    def save_patch_info(self, path):
        info_file = Path(
            path, f"{Path(self._img_filename).stem}_patch_info.json"
        )
        with open(info_file, "w") as file:
            json.dump(self._patch_info, file, indent=2)


# Load the .env file


def load_args():
    load_dotenv()
    args = {}
    args["base_path"] = os.getenv("BASE_PATH")
    args["images_in"] = os.path.join(args["base_path"], os.getenv("IMAGES_IN"))
    args["patches_dir"] = os.path.join(
        args["base_path"], os.getenv("PATCHES_IN")
    )
    args["bbox_dir"] = os.path.join(args["base_path"], os.getenv("BBOXES_IN"))
    args["model_nn_weights"] = os.path.join(
        args["base_path"], os.getenv("MODEL_NN_WEIGHTS")
    )

    parser = ArgumentParser()
    parser.add_argument(
        "--im_path",
        help="paths to input images containing multiple patches",
        default=args["images_in"],
    )
    parser.add_argument(
        "--patches_dir",
        help="path to save images with bounding boxes and patches crops",
        default=args["patches_dir"],
    )
    parser.add_argument(
        "--bbox_dir",
        help="path to save bounding box images",
        default=args["bbox_dir"],
    )

    parser.add_argument(
        "--cp", help="yolov8 cp path", default=args["model_nn_weights"]
    )
    args.update(parser.parse_args())
    args = SimpleNamespace(**args)
    return args


if __name__ == "__main__":
    args = load_args()

    os.makedirs(args.im_path, exist_ok=True)
    os.makedirs(args.patches_dir, exist_ok=True)

    patch_finder = PatchFinder(args.cp)
    paths = list(Path(args.im_path).glob("*.jpg"))
    pbar = tqdm(paths, desc="Processing images")

    for i, img_filename in enumerate(pbar):
        pbar.set_description(f"{img_filename} {i + 1}/{len(paths)}")
        patch_finder.load_image(str(img_filename))
        patch_finder.predict_bounding_box()

        filename = Path(args.bbox_dir, img_filename.name)
        patch_finder.save_image(filename)

        filepath = Path(args.patches_dir, img_filename.stem)
        patch_finder.save_patches(filepath)

        # Save patch information
        patch_finder.save_patch_info(filepath)
