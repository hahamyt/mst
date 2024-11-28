from pathlib import Path
import pickle
import random
import numpy as np
import json
import cv2
from copy import deepcopy
from isegm.data.base import ISDataset
from isegm.data.sample import DSample

class LoveDADataset(ISDataset):
    def __init__(self, dataset_path, split='train', stuff_prob=0.0, scale='all',
                 allow_list_name=None, anno_file='hannotation.pickle', **kwargs):
        super(LoveDADataset, self).__init__(**kwargs)
        dataset_path = Path(dataset_path)
        self._split_path = dataset_path / split
        self.split = split
        self._images_path = self._split_path / 'images'
        self._masks_path = self._split_path / 'masks'
        self.stuff_prob = stuff_prob

        with open(self._split_path / anno_file, 'rb') as f:
            self.dataset_samples = sorted(pickle.load(f).items())
        self.scale = scale
        self.valid_samples = None

        if scale != 'all':
            # 多尺度区分
            ms_meta = np.load(self._split_path / 'lovedams.npy', allow_pickle=True).item()[scale]
            self.valid_samples = []
            tmp = []
            for k, v in ms_meta.items():
                if len(v)!=0:
                    idx = int(k)
                    self.valid_samples.append(v)
                    tmp.append(self.dataset_samples[idx])

            self.dataset_samples = tmp

        if allow_list_name is not None:
            allow_list_path = self._split_path / allow_list_name
            with open(allow_list_path, 'r') as f:
                allow_images_ids = json.load(f)
            allow_images_ids = set(allow_images_ids)

            self.dataset_samples = [sample for sample in self.dataset_samples
                                    if sample[0] in allow_images_ids]

    @property
    def name(self):
        return 'LoveDA_' + self.scale


    def get_sample(self, index) -> DSample:
        image_id, sample = self.dataset_samples[index]
        image_path = self._images_path / f'{image_id}.jpg'

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        packed_masks_path = self._masks_path / f'{image_id}.pickle'
        with open(packed_masks_path, 'rb') as f:
            encoded_layers, objs_mapping = pickle.load(f)
        layers = [cv2.imdecode(x, cv2.IMREAD_UNCHANGED) for x in encoded_layers]
        layers = np.stack(layers, axis=2)

        if self.valid_samples:
            valid_obj = self.valid_samples[index]
            sample['num_instance_masks'] = len(valid_obj)
            tmp_objs_mapping = []
            tmp_layers = np.zeros_like(layers)
            tmp_hierarchy = {}
            for idx, obj_id in enumerate(valid_obj):
                tmp_objs_mapping.append((0, idx+1))   # 对layers操作要使用objs_mapping
                tmp_layers[layers==objs_mapping[obj_id][-1]] = idx+1
                tmp_hierarchy[idx] = None

            layers = tmp_layers
            objs_mapping = tmp_objs_mapping
            sample['hierarchy'] = tmp_hierarchy

        instances_info = deepcopy(sample['hierarchy'])
        for inst_id, inst_info in list(instances_info.items()):
            if inst_info is None:
                inst_info = {'children': [], 'parent': None, 'node_level': 0}
                instances_info[inst_id] = inst_info
            inst_info['mapping'] = objs_mapping[inst_id]

        if self.stuff_prob > 0 and random.random() < self.stuff_prob:
            for inst_id in range(sample['num_instance_masks'], len(objs_mapping)):
                instances_info[inst_id] = {
                    'mapping': objs_mapping[inst_id],
                    'parent': None,
                    'children': []
                }
        else:
            for inst_id in range(sample['num_instance_masks'], len(objs_mapping)):
                layer_indx, mask_id = objs_mapping[inst_id]
                layers[:, :, layer_indx][layers[:, :, layer_indx] == mask_id] = 0

        return DSample(image, layers, objects=instances_info)