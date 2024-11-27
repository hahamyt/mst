import torch

from isegm.model.is_adaptivevit_model import AdaptiveVitModel
from isegm.utils.exp_imports.default import *
from isegm.model.modeling.transformer_helper.cross_entropy_loss import CrossEntropyLoss
from torch.nn.parallel import DistributedDataParallel as DDP

MODEL_NAME = 'cclvs_adaptivevit_base224'

def main(cfg, rank=0):
    model, model_cfg = init_model(cfg, rank)
    train(model, cfg, model_cfg, rank)


def init_model(cfg, rank):
    model_cfg = edict()
    model_cfg.crop_size = (224, 224)
    model_cfg.num_max_points = 24
    model_cfg.with_prev_mask = True

    backbone_params = dict(
        img_size=model_cfg.crop_size,
        patch_size=(16,16),
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
    )

    neck_params = dict(
        in_dim = 768,
        out_dims = [256, 256, 256, 256],
    )

    head_params = dict(
        in_channels=[256, 256, 256, 256],
        in_index=[0, 1, 2, 3],
        dropout_ratio=0.1,
        num_classes=1,
        loss_decode=CrossEntropyLoss(),
        align_corners=False,
        upsample=cfg.upsample,
        channels={'x1':256, 'x2': 128, 'x4': 64}[cfg.upsample],
    )

    model = AdaptiveVitModel(
        use_disks=True,
        norm_radius=5,
        with_prev_mask=True,
        backbone_params=backbone_params,
        neck_params=neck_params,
        head_params=head_params,
        random_split=cfg.random_split,
    )

    model.backbone.init_weights_from_pretrained(cfg.IMAGENET_PRETRAINED_MODELS.MAE_BASE)
    if cfg.distributed:
        torch.cuda.set_device(rank)
        model = DDP(model.to(rank), device_ids=[rank], output_device=rank, find_unused_parameters=False)
    else:
        if not cfg.multi_gpu:
            model.to(cfg.device)

    return model, model_cfg

def train(model, cfg, model_cfg, rank):
    cfg.batch_size = 32 if cfg.batch_size < 1 else cfg.batch_size
    cfg.epochs = 5 if cfg.epochs < 1 else cfg.epochs
    cfg.val_batch_size = cfg.batch_size
    crop_size = model_cfg.crop_size

    loss_cfg = edict()
    loss_cfg.instance_loss = NormalizedFocalLossSigmoid(alpha=0.5, gamma=2)
    loss_cfg.instance_loss_weight = 1.0
    loss_cfg.token_loss = PatchLoss()
    loss_cfg.token_loss_weight = 1e-3
    loss_cfg.score_loss = ScoreLoss()
    loss_cfg.score_loss_weight = 1e-3

    train_augmentator = Compose([
        UniformRandomResize(scale_range=(0.75, 1.40)),
        HorizontalFlip(),
        PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
        RandomCrop(*crop_size),
        RandomBrightnessContrast(brightness_limit=(-0.25, 0.25), contrast_limit=(-0.15, 0.4), p=0.75),
        RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.75)
    ], p=1.0)

    val_augmentator = Compose([
        PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
        RandomCrop(*crop_size)
    ], p=1.0)

    points_sampler = MultiPointSampler(model_cfg.num_max_points,
                                       prob_gamma=0.80,
                                       merge_objects_prob=0.15,
                                       use_hierarchy=False,
                                       max_num_merged_objects=2)

    trainset = CocoLvisDataset(
        cfg.LVIS_v1_PATH,
        split='train',
        augmentator=train_augmentator,
        min_object_area=1000,
        keep_background_prob=0.05,
        points_sampler=points_sampler,
        epoch_len=30000,
        stuff_prob=0.30,
    )

    valset = CocoLvisDataset(
        cfg.LVIS_v1_PATH,
        split='val',
        augmentator=val_augmentator,
        min_object_area=1000,
        points_sampler=points_sampler,
        epoch_len=2000
    )

    optimizer_params = {
        'lr': 5e-6, 'betas': (0.9, 0.999), 'eps': 1e-8,
    }

    lr_scheduler = partial(torch.optim.lr_scheduler.MultiStepLR,
                           milestones=[60, 80], gamma=0.1)
    trainer = ISTrainer(model, cfg, model_cfg, loss_cfg,
                        trainset, valset,
                        optimizer='adamw',
                        optimizer_params=optimizer_params,
                        layerwise_decay=cfg.layerwise_decay,
                        lr_scheduler=lr_scheduler,
                        checkpoint_interval=2,
                        image_dump_interval=300,
                        metrics=[AdaptiveIoU()],
                        max_interactive_points=model_cfg.num_max_points,
                        max_num_next_clicks=3,
                        rank=rank)
    trainer.run(num_epochs=cfg.epochs if not cfg.debug else 1, validation=True)

