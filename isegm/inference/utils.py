from datetime import timedelta
from pathlib import Path
import torch
import numpy as np
from isegm.data.datasets import *
from isegm.utils.serialization import load_model


def get_time_metrics(all_ious, elapsed_time):
    n_images = len(all_ious)
    n_clicks = sum(map(len, all_ious))

    mean_spc = elapsed_time / n_clicks
    mean_spi = elapsed_time / n_images

    return mean_spc, mean_spi


def load_is_model(checkpoint, device, eval_ritm=False, **kwargs):
    if isinstance(checkpoint, (str, Path)):
        checkpoint_dict = torch.load(checkpoint, map_location='cpu')
    else:
        checkpoint_dict = checkpoint

    if isinstance(checkpoint_dict, list):
        models = [load_single_is_model(x, device, eval_ritm, **kwargs) for x in checkpoint_dict]
        return models
    else:
        model = load_single_is_model(checkpoint_dict, device, eval_ritm, **kwargs)
        return model


def load_single_is_model(checkpoint_dict, device, eval_ritm=False, **kwargs):
    config = checkpoint_dict['config']
    model = load_model(config, eval_ritm, **kwargs)

    state_dict_key = 'state_dict'
    if state_dict_key in checkpoint_dict:
        model.load_state_dict(checkpoint_dict[state_dict_key], strict=False)
    else:
        raise ValueError(f"Expected key '{state_dict_key}' in checkpoint")

    model.to(device)
    if eval_ritm:
        model.eval()
    return model


def get_dataset(dataset_name, cfg):
    if dataset_name == 'GrabCut':
        dataset = GrabCutDataset(cfg.GRABCUT_PATH)
    elif dataset_name == 'Berkeley':
        dataset = BerkeleyDataset(cfg.BERKELEY_PATH)
    elif dataset_name == 'DAVIS':
        dataset = DavisDataset(cfg.DAVIS_PATH)
    elif dataset_name == 'SBD':
        dataset = SBDEvaluationDataset(cfg.SBD_PATH)
    elif dataset_name == 'SBD_Train':
        dataset = SBDEvaluationDataset(cfg.SBD_PATH, split='train')
    elif dataset_name == 'PascalVOC':
        dataset = PascalVocDataset(cfg.PASCALVOC_PATH, split='val')
    elif dataset_name == 'COCO_MVal':
        dataset = DavisDataset(cfg.COCO_MVAL_PATH)
    elif dataset_name == 'BraTS':
        dataset = BraTSDataset(cfg.BraTS_PATH)
    elif dataset_name == 'ssTEM':
        dataset = ssTEMDataset(cfg.ssTEM_PATH)
    elif dataset_name == 'OAIZIB':
        dataset = OAIZIBDataset(cfg.OAIZIB_PATH)
    elif dataset_name == 'HARD':
        dataset = HARDDataset(cfg.HARD_PATH)
    elif dataset_name == 'HQSeg44k':
        dataset = HQSeg44kDataset(cfg.HQSeg44K_PATH, split='val')
    elif dataset_name == 'LoveDA':
        dataset = LoveDADataset(cfg.IS_LoveDA_PATH, split='val', scale='all')
    elif dataset_name == 'LoveDASMALL':
        dataset = LoveDADataset(cfg.IS_LoveDA_PATH, split='val', scale='small')
    elif dataset_name == 'LoveDALARGE':
        dataset = LoveDADataset(cfg.IS_LoveDALARGE_PATH, split='val', scale='medium')
    elif dataset_name == 'LoveDAMEDIUM':
        dataset = LoveDADataset(cfg.IS_LoveDAMEDIUM_PATH, split='val', scale='large')
    elif dataset_name == 'LoveDAHUGE':
        dataset = LoveDADataset(cfg.IS_LoveDAHUGE_PATH, split='val', scale='huge')
    else:
        dataset = None

    return dataset


def get_iou(gt_mask, pred_mask, ignore_label=-1):
    ignore_gt_mask_inv = gt_mask != ignore_label
    obj_gt_mask = gt_mask == 1

    intersection = np.logical_and(np.logical_and(pred_mask, obj_gt_mask), ignore_gt_mask_inv).sum()
    union = np.logical_and(np.logical_or(pred_mask, obj_gt_mask), ignore_gt_mask_inv).sum()

    return intersection / union


def compute_noc_metric(all_ious, iou_thrs, max_clicks=20):
    def _get_noc(iou_arr, iou_thr):
        vals = iou_arr >= iou_thr
        return np.argmax(vals) + 1 if np.any(vals) else max_clicks

    noc_list = []
    noc_list_std = []
    over_max_list = []
    for iou_thr in iou_thrs:
        scores_arr = np.array([_get_noc(iou_arr, iou_thr)
                               for iou_arr in all_ious], dtype=np.int32)

        score = scores_arr.mean()
        score_std = scores_arr.std()
        over_max = (scores_arr == max_clicks).sum()

        noc_list.append(score)
        noc_list_std.append(score_std)
        over_max_list.append(over_max)

    return noc_list, noc_list_std, over_max_list


def find_checkpoint(weights_folder, checkpoint_name):
    weights_folder = Path(weights_folder)
    if ':' in checkpoint_name:
        model_name, checkpoint_name = checkpoint_name.split(':')
        models_candidates = [x for x in weights_folder.glob(f'{model_name}*') if x.is_dir()]
        assert len(models_candidates) == 1
        model_folder = models_candidates[0]
    else:
        model_folder = weights_folder

    if checkpoint_name.endswith('.pth'):
        if Path(checkpoint_name).exists():
            checkpoint_path = checkpoint_name
        else:
            checkpoint_path = weights_folder / checkpoint_name
    else:
        model_checkpoints = list(model_folder.rglob(f'{checkpoint_name}*.pth'))
        assert len(model_checkpoints) == 1
        checkpoint_path = model_checkpoints[0]

    return str(checkpoint_path)


def get_results_table(noc_list, over_max_list, brs_type, dataset_name, mean_spc, elapsed_time,
                      n_clicks=20, model_name=None):
    table_header = (f'|{"Predictor":^13}|{"Dataset":^11}|'
                    f'{"NoC@80%":^9}|{"NoC@85%":^9}|{"NoC@90%":^9}|{"NoC@95%":^9}|'
                    f'{">="+str(n_clicks)+"@90%":^9}|{">="+str(n_clicks)+"@95%":^9}|'
                    f'{"SPC,s":^7}|{"Time":^9}|')
    row_width = len(table_header)

    header = f'Eval results for model: {model_name}\n' if model_name is not None else ''
    header += '-' * row_width + '\n'
    header += table_header + '\n' + '-' * row_width

    eval_time = str(timedelta(seconds=int(elapsed_time)))
    table_row = f'|{brs_type:^13}|{dataset_name:^11}|'
    table_row += f'{noc_list[0]:^9.2f}|'
    table_row += f'{noc_list[1]:^9.2f}|' if len(noc_list) > 1 else f'{"?":^9}|'
    table_row += f'{noc_list[2]:^9.2f}|' if len(noc_list) > 2 else f'{"?":^9}|'
    table_row += f'{noc_list[3]:^9.2f}|' if len(noc_list) > 3 else f'{"?":^9}|'

    table_row += f'{over_max_list[1]:^9}|' if len(noc_list) > 1 else f'{"?":^9}|'
    table_row += f'{over_max_list[2]:^9}|' if len(noc_list) > 2 else f'{"?":^9}|'
    table_row += f'{mean_spc:^7.3f}|{eval_time:^9}|'

    return header, table_row
