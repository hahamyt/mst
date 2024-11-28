import numpy as np
import torch
import torch.nn.functional as F

class SAMPredictor(object):
    def __init__(self, device,
                 type='SAM',
                 net_clicks_limit=None,
                 with_flip=False,
                 zoom_in=None,
                 max_size=None,
                 cascade_step=0,
                 cascade_adaptive=False,
                 cascade_clicks=1,
                 **kwargs):

        self.zoom_in = zoom_in
        if type == 'SAM':
            from isegm.model.modeling.segment_anything import sam_model_registry, SamPredictor
            sam_checkpoint = "./weights/sam_vit_b_01ec64.pth"
            # sam_checkpoint = "./weights/sam_vit_h_4b8939.pth"
            model_type = "vit_b"    # hq_vit_h
            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to(device=device)
            self.sam_predictor = SamPredictor(sam)
        elif type == 'HQSAM':
            from isegm.model.modeling.segment_anything import sam_model_registry, HQSamPredictor
            # sam_checkpoint = "./weights/sam_hq_vit_h.pth"
            sam_checkpoint = "./weights/sam_vit_b_01ec64.pth"
            model_type = 'hq_vit_b'

            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to(device=device)
            self.sam_predictor = HQSamPredictor(sam)
        elif type == 'HQSAM_CUSTOM':
            from isegm.model.modeling.segment_anything import sam_model_registry, HQSamPredictor_CUSTOM
            sam_checkpoint = "./weights/sam_vit_b_01ec64.pth"
            model_type = 'hq_vit_b'

            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            # logs_prefix = kwargs['logs_prefix']
            # ckpt = torch.load(f'/root/workspace/sam-hq/train/work_dirs/hq_sam_b_gnn/epoch_{int(logs_prefix.split("_")[-1])}.pth')['sam']
            # sam.load_state_dict(ckpt, strict=False)
            sam.to(device=device)

            self.sam_predictor = HQSamPredictor_CUSTOM(sam, kwargs['logs_prefix'])
        elif type == 'HQSAM_BASE':
            from isegm.model.modeling.segment_anything import sam_model_registry, HQSamPredictor_BASE
            sam_checkpoint = "./weights/sam_vit_b_01ec64.pth"
            model_type = 'hq_vit_b'

            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to(device=device)
            self.sam_predictor = HQSamPredictor_BASE(sam, kwargs['logs_prefix'])
        elif type == 'SAM_BASE':
            from isegm.model.modeling.segment_anything import sam_model_registry, SamPredictor_BASE
            sam_checkpoint = "./weights/sam_vit_b_01ec64.pth"
            # sam_checkpoint = "./weights/sam_vit_h_4b8939.pth"
            model_type = 'hq_vit_b'

            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to(device=device)
            self.sam_predictor = SamPredictor_BASE(sam, kwargs['logs_prefix'])

        self.name = type

    def set_input_image(self, image):
        self.sam_predictor.set_image(image)
        self.im_sz = image.shape[:2]

    def get_prediction(self, clicker, prev_mask=None, **kwargs):
        clicks_list = clicker.get_clicks()

        pred_logits = self._get_prediction(clicks_list)['instances']
        prediction = F.interpolate(pred_logits, mode='bilinear', align_corners=True,
                                   size=self.im_sz)[0]

        return prediction.cpu().numpy()[0]

    def _get_prediction(self, clicks_lists):
        input_point, input_label = self.get_points_nd(clicks_lists)
        masks, scores, logits = self.sam_predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        return {'instances': logits[None, None, np.argmax(scores.cpu().numpy()), ...]}

    def get_points_nd(self, clicks_lists):
        coord = []
        label = []
        for click in clicks_lists:
            coord.append((click.coords[1], click.coords[0]))
            if click.is_positive:
                label.append(1)
            else:
                label.append(0)
        return np.array(coord), np.array(label)

