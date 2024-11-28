import os
import random
import pickle as pkl
from pathlib import Path

import cv2
import numpy as np

from isegm.data.base import ISDataset
from isegm.data.sample import DSample


class OpenImagesDataset(ISDataset):
    def __init__(self, dataset_path, split='train', **kwargs):
        super().__init__(**kwargs)
        assert split in {'train', 'val', 'test'}

        self.dataset_path = Path(dataset_path)
        self._split_path = self.dataset_path / split
        self._images_path = self._split_path / 'images'
        self._masks_path = self._split_path / 'masks'
        self.dataset_split = split

        clean_anno_path = self._split_path / f'{split}-annotations-object-segmentation_clean.pkl'
        if os.path.exists(clean_anno_path):
            with clean_anno_path.open('rb') as f:
                annotations = pkl.load(f)
        else:
            raise RuntimeError(f"Can't find annotations at {clean_anno_path}")
        self.image_id_to_masks = annotations['image_id_to_masks']
        self.dataset_samples = annotations['dataset_samples']

    def get_sample(self, index) -> DSample:
        image_id = self.dataset_samples[index]
        # select random mask for an image
        current_sample = random.choice(self.image_id_to_masks[image_id])
        image_path = str(current_sample['image_path'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_path = str(current_sample['mask_path'])
        instances_mask = cv2.imread(mask_path)
        instances_mask = cv2.cvtColor(instances_mask, cv2.COLOR_BGR2GRAY)
        instances_mask[instances_mask > 0] = 1
        instances_mask = instances_mask.astype(np.int32)

        min_width = min(image.shape[1], instances_mask.shape[1])
        min_height = min(image.shape[0], instances_mask.shape[0])

        if image.shape[0] != min_height or image.shape[1] != min_width:
            image = cv2.resize(image, (min_width, min_height), interpolation=cv2.INTER_LINEAR)
        if instances_mask.shape[0] != min_height or instances_mask.shape[1] != min_width:
            instances_mask = cv2.resize(instances_mask, (min_width, min_height), interpolation=cv2.INTER_NEAREST)

        object_ids = [1] if instances_mask.sum() > 0 else []

        return DSample(image, instances_mask, objects_ids=object_ids, sample_id=index)
