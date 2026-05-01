"""
Stage 3 image classification dataset loader.
"""

import pickle
from pathlib import Path

import numpy as np

from local_code.base_class.dataset import dataset


class Dataset_Loader_Image_Classification(dataset):
    dataset_source_folder_path = None
    dataset_source_file_name = None
    image_mode = "auto"

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def _load_pickle(self):
        data_path = Path(self.dataset_source_folder_path) / self.dataset_source_file_name
        with open(data_path, "rb") as handle:
            return pickle.load(handle)

    def _infer_input_shape(self, image):
        image = np.asarray(image)
        if self.image_mode == "grayscale":
            if image.ndim == 3:
                image = image[:, :, 0]
            return 1, image.shape[0], image.shape[1]
        if image.ndim == 2:
            return 1, image.shape[0], image.shape[1]
        return image.shape[2], image.shape[0], image.shape[1]

    def _encode_labels(self, split, label_to_index):
        encoded = []
        for instance in split:
            item = dict(instance)
            item["encoded_label"] = label_to_index[item["label"]]
            encoded.append(item)
        return encoded

    def load(self):
        print(f"loading {self.dataset_source_file_name} image data...")
        loaded = self._load_pickle()

        labels = [item["label"] for item in loaded["train"] + loaded["test"]]
        class_labels = sorted(set(labels))
        label_to_index = {label: index for index, label in enumerate(class_labels)}

        train = self._encode_labels(loaded["train"], label_to_index)
        test = self._encode_labels(loaded["test"], label_to_index)
        input_shape = self._infer_input_shape(train[0]["image"])

        return {
            "train": train,
            "test": test,
            "metadata": {
                "dataset_name": self.dataset_source_file_name,
                "image_mode": self.image_mode,
                "input_shape": input_shape,
                "num_classes": len(class_labels),
                "class_labels": class_labels,
                "label_to_index": label_to_index,
                "train_size": len(train),
                "test_size": len(test),
            },
        }
