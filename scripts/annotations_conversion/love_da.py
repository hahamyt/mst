import cv2
import pickle
from collections import OrderedDict
import os
from isegm.utils.misc import get_bbox_from_mask
from scripts.annotations_conversion.common import get_masks_hierarchy, encode_masks
import skimage.measure as measure
from tqdm.contrib import tzip

COLOR_MAP = OrderedDict(
    Background=(255, 255, 255),
    Building=(255, 0, 0),
    Road=(255, 255, 0),
    Water=(0, 0, 255),
    Barren=(159, 129, 183),
    Forest=(0, 255, 0),
    Agricultural=(255, 195, 128),
)


LABEL_MAP = OrderedDict(
    Background=1,
    Building=2,
    Road=3,
    Water=4,
    Barren=5,
    Forest=6,
    Agricultural=7
)

out_path = "/home/x/datasets/IS_LOVEDA"

def create_annotations(dataset_path, dataset_split='train', n_jobs=1):
    _rural_imgs_path = dataset_path/dataset_split/"Rural/images_png"
    _rural_masks_path = dataset_path/dataset_split/ "Rural/masks_png"
    rural_imgs = [dataset_path/dataset_split/"Rural/images_png" / f for f in os.listdir(_rural_imgs_path)]
    rural_masks = [dataset_path/dataset_split/"Rural/masks_png" / f for f in os.listdir(_rural_masks_path)]

    _urban_imgs_path = dataset_path / dataset_split / "Urban/images_png"
    _urban_masks_path = dataset_path / dataset_split / "Urban/masks_png"
    urban_imgs = [dataset_path/dataset_split/"Urban/images_png" / f for f in os.listdir(_urban_imgs_path)]
    urban_masks = [dataset_path/dataset_split/"Urban/masks_png" / f for f in os.listdir(_urban_masks_path)]

    images = rural_imgs + urban_imgs
    masks = rural_masks + urban_masks

    loveda_annotation = dict()

    for im_path, mask_path in tzip(images, masks):
        image = cv2.imread(str(im_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_name = str(im_path).split('/')[-4]+str(im_path).split('/')[-3]+str(im_path).split('/')[-2]+str(im_path).split('/')[-1]
        mask = cv2.imread(str(mask_path))[:, :, 0]

        masks = []
        for idx, name in zip(range(2, 8), list(LABEL_MAP)[1:]):
            obj_mask = mask==idx

            if obj_mask.sum() > 1000:
                labels = measure.label(obj_mask, connectivity=2)

                if labels.max() > 1:
                    for i in range(1, labels.max()+1):
                        if (labels==i).sum() > 1000:
                            masks.append(labels==i)
                else:
                    masks.append(obj_mask)

        masks_meta = [(get_bbox_from_mask(x), x.sum()) for x in masks]
        if not masks:
            continue
        hierarchy = get_masks_hierarchy(masks, masks_meta)
        for obj_id, obj_info in list(hierarchy.items()):
            if obj_info['parent'] is None and len(obj_info['children']) == 0:
                hierarchy[obj_id] = None

        num_instance_masks = len(masks)

        loveda_annotation[img_name] = {
            'num_instance_masks': num_instance_masks,
            'hierarchy': hierarchy
        }

        # print(num_instance_masks)
        cv2.imwrite('{}/{}/{}'.format(out_path, dataset_split, f'images/{img_name}.jpg'), image)
        with open('{}/{}/{}'.format(out_path, dataset_split, f'masks/{img_name}.pickle'), 'wb') as f:
            pickle.dump(encode_masks(masks), f)

    with open('{}/{}/{}'.format(out_path, dataset_split, 'hannotation.pickle'), 'wb') as f:
        pickle.dump(loveda_annotation, f, protocol=pickle.HIGHEST_PROTOCOL)
