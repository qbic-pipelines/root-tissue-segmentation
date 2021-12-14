import os
import os.path
import warnings
from glob import glob
from typing import Any, Callable, Dict, Optional, Tuple
from urllib.error import URLError

import cv2
import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torch.utils.data as data
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torchvision.datasets.utils import download_and_extract_archive


class PHDFM(data.Dataset):
    mirrors = [
        'https://drive.google.com/file/d/',
    ]

    resources = [
        ("images.tar.gz", "18Fd4WQ_M6gz9qV-43rPjkRmNw2LT3zxI/view?usp=sharing", "54e12b0ad17e4d89796d64b92801c90a"),
        ("masks.tar.gz", "12ViQsa9K8EqgJE1DrhFJiKot41WUEj3R/view?usp=sharing", "b7fa40aa75788235f9fa3696b760ea2d"),
    ]

    training_file = 'training.pt'
    validation_file = 'validation.pt'
    test_file = 'test.pt'
    weight_file = "class_weights.csv"
    classes = ['background', 'root', 'early elongation zone', 'late elongation zone', 'meristematic zone']

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def validation_labels(self):
        warnings.warn("validation_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def validation_data(self):
        warnings.warn("validation_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(
        self,
        root: str,
        set: str = "training",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False
    ) -> None:
        super(PHDFM, self).__init__()
        self.set = set  # training set or test set
        self.transform = transform
        self.target_transform = target_transform
        self.root = root

        if download:
            self.download()
        if not self._check_exists():
            raise RuntimeError('Dataset not found.')

        if self.set == "training":
            data_file = self.training_file
        elif self.set == "validation":
            data_file = self.validation_file
        else:
            data_file = self.test_file
        data_mask = torch.load(os.path.join(self.processed_folder, data_file))
        self.data = data_mask[:, :, :, 0:1]
        self.targets = data_mask[:, :, :, 1:2]
        self.class_weights = pd.read_csv(os.path.join(self.processed_folder, self.weight_file))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        :param: index (int): Index

        :return tuple: (image, target) where target is index of the target class.
        """

        img, target = self.data[index], self.targets[index].int()
        if self.transform is not None:
            augmented = self.transform(image=img.numpy(), mask=target.numpy())
            img = augmented['image'].transpose(2, 0, 1)
            target = augmented['mask'].astype('int64').squeeze()
            img = img.astype('float32')

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def class_to_idx(self) -> Dict[str, int]:
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self) -> bool:
        return (os.path.exists(os.path.join(self.processed_folder,
                                            self.training_file)) and
                os.path.exists(os.path.join(self.processed_folder,
                                            self.validation_file)) and
                os.path.exists(os.path.join(self.processed_folder,
                                            self.test_file)) and
                os.path.exists(os.path.join(self.processed_folder,
                                            self.weight_file))
                )

    def download(self) -> None:
        """Download the PHDFM data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # download files
        for filename, uniqueID, md5 in self.resources:
            for mirror in self.mirrors:
                url = "{}{}".format(mirror, uniqueID)
                try:
                    print("Downloading {}".format(url))
                    download_and_extract_archive(
                        url, download_root=self.raw_folder,
                        filename=filename,
                        md5=md5
                    )
                except URLError as error:
                    print(
                        "Failed to download (trying next):\n{}".format(error)
                    )
                    continue
                finally:
                    print()
                break
            else:
                raise RuntimeError("Error downloading {}".format(filename))

        # process and save as torch files
        print('Processing...')
        training_set, validation_set, test_set, weight_df = self.transform_files()
        with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.processed_folder, self.validation_file), 'wb') as f:
            torch.save(validation_set, f)
        with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)
        weight_df.to_csv(os.path.join(self.processed_folder, self.weight_file), index=False)
        print('Done!')

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")

    def transform_files(self) -> (torch.FloatTensor, torch.IntTensor, pd.DataFrame):
        """
        Transforms files to .pt tensor files.
        :return: training set, test set, training weights
        """
        img_ids = glob(os.path.join(self.raw_folder, "images", '*' + ".png"))
        img_ids = sorted([os.path.splitext(os.path.basename(p))[0] for p in img_ids])
        train_ids, val_test_ids = train_test_split(img_ids, test_size=0.2, random_state=42)
        val_ids, test_ids = train_test_split(val_test_ids, test_size=0.5, random_state=42)

        ids = {"training": train_ids,
               "test": val_test_ids,
               "validation": val_test_ids}

        classes = ['background', 'root', 'early elongation zone', 'late elongation zone', 'meristematic zone']
        tensors = {}
        weight_df = pd.DataFrame(columns=["class_ids", "classes", "weights", "set_name"])
        for set_name, set_ids in ids.items():
            training = [[], [], []]
            for img_id in set_ids:
                img = cv2.imread(os.path.join(self.raw_folder, "images", img_id + ".png"), cv2.IMREAD_GRAYSCALE)[
                    ..., None]
                mask = []
                for idx, cls in enumerate(classes):
                    label = cv2.imread(os.path.join(self.raw_folder, "masks", cls,
                                                    img_id + ".png"), cv2.IMREAD_GRAYSCALE)[..., None]
                    label[label > 0] = 1
                    mask.append(label)
                    if idx > 1:
                        mask[1][label > 0] = 0
                mask_one_hot = np.dstack(mask).astype(int)
                mask = np.expand_dims(np.argmax(mask_one_hot, axis=2), 2)
                training[0].append(torch.FloatTensor(img))
                training[1].append(torch.IntTensor(mask))
                training[2].append(int(img_id))

            imgs = torch.stack(training[0], dim=0)
            masks = torch.stack(training[1], dim=0)
            wt = []
            for i in range(5):
                post = len(masks[masks==i])
                neg = len(masks[masks!=i])+post
                wt.append(post/neg)
            weights = 1/np.array(wt)
            print(weights/np.max(weights))
            unique = np.unique(masks)
            class_weights = weights/np.max(weights)#compute_class_weight('balanced', unique, masks.numpy().flatten())
            weights = pd.DataFrame({"class_ids": unique, "classes": classes, "weights": class_weights})
            weights['set_name'] = set_name
            weight_df = weight_df.append(weights, ignore_index=True)
            tensor = torch.cat([imgs, masks], dim=3)
            tensors[set_name] = tensor
        return tensors["training"], tensors["validation"], tensors["test"], weight_df
