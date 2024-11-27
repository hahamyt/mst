from pathlib import Path
import numpy as np
import cv2
from isegm.data.base import ISDataset
from isegm.data.sample import DSample


class HQSegEvalDataset(ISDataset):
    def __init__(self, dataset_path, **kwargs):
        super(HQSegEvalDataset, self).__init__(**kwargs)
        dataset_path = Path(dataset_path)

        im_path = dataset_path / 'im'
        mask_path = dataset_path / 'gt'

        sample_names = [x.stem for x in im_path.iterdir() if x.suffix == '.jpg']
        self.dataset_samples = [path for path in im_path.iterdir() if path.suffix == '.jpg']
        self.masks_paths = [path for path in mask_path.iterdir() if path.stem in sample_names]

        self.dataset_samples.sort()
        self.masks_paths.sort()

    def get_sample(self, index) -> DSample:
        image_path = self.dataset_samples[index]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_path = self.masks_paths[index]
        instances_mask = np.max(cv2.imread(str(mask_path)).astype(np.uint8), axis=2)
        instances_mask[instances_mask > 0] = 1


        return DSample(image, instances_mask, objects_ids=[1], sample_id=index, image_path=image_path, mask_path=mask_path)
