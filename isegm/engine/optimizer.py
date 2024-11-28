import torch
import math
from isegm.utils.log import logger
import isegm.utils.lr_decay as lrd

try:
    from lion_pytorch import Lion
    LION_AVAILABLE = True
except ImportError:
    LION_AVAILABLE = False
    logger.warning("Lion optimizer is not installed. Install it with 'pip install lion-pytorch' if needed.")

def get_optimizer(model, opt_name, opt_kwargs):
    params = []
    names = []
    base_lr = opt_kwargs['lr']
    for name, param in model.named_parameters():
        param_group = {'params': [param]}
        if not param.requires_grad:
            params.append(param_group)
            continue
        else:
            names.append(name)
        if not math.isclose(getattr(param, 'lr_mult', 1.0), 1.0):
            logger.info(f'Applied lr_mult={param.lr_mult} to "{name}" parameter.')
            param_group['lr'] = param_group.get('lr', base_lr) * param.lr_mult

        params.append(param_group)

    optimizer_dict = {
        'sgd': torch.optim.SGD,
        'adam': torch.optim.Adam,
        'adamw': torch.optim.AdamW,
    }

    if LION_AVAILABLE:
        optimizer_dict['lion'] = Lion

    opt_name_lower = opt_name.lower()
    if opt_name_lower not in optimizer_dict:
        raise ValueError(f"Unsupported optimizer: {opt_name}. Available: {list(optimizer_dict.keys())}")

    optimizer_class = optimizer_dict[opt_name_lower]
    if opt_name_lower == 'lion':
        opt_kwargs = {k: v for k, v in opt_kwargs.items() if k != 'eps'}
    optimizer = optimizer_class(
        filter(lambda p: p.requires_grad, model.parameters()), **opt_kwargs
    )

    return optimizer

def get_optimizer_with_layerwise_decay(model, opt_name, opt_kwargs):
    lr = opt_kwargs['lr']
    param_groups = lrd.param_groups_lrd(
        model,
        lr,
        weight_decay=opt_kwargs.get('weight_decay', 0.02),
        no_weight_decay_list=model.module.backbone.no_weight_decay(),
        layer_decay=0.75
    )

    optimizer_dict = {
        'sgd': torch.optim.SGD,
        'adam': torch.optim.Adam,
        'adamw': torch.optim.AdamW,
    }

    if LION_AVAILABLE:
        optimizer_dict['lion'] = Lion

    opt_name_lower = opt_name.lower()
    if opt_name_lower not in optimizer_dict:
        raise ValueError(f"Unsupported optimizer: {opt_name}. Available: {list(optimizer_dict.keys())}")

    optimizer_class = optimizer_dict[opt_name_lower]
    if opt_name_lower == 'lion':
        opt_kwargs = {k: v for k, v in opt_kwargs.items() if k != 'eps'}
    optimizer = optimizer_class(param_groups, **opt_kwargs)

    return optimizer