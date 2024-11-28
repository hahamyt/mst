import torch
import math
from isegm.utils.log import logger
import isegm.utils.lr_decay as lrd

# 引入 Lion 优化器
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

    # 添加 Lion 优化器到字典
    if LION_AVAILABLE:
        optimizer_dict['lion'] = Lion

    # 检查提供的优化器是否存在
    opt_name_lower = opt_name.lower()
    if opt_name_lower not in optimizer_dict:
        raise ValueError(f"Unsupported optimizer: {opt_name}. Available: {list(optimizer_dict.keys())}")

    optimizer_class = optimizer_dict[opt_name_lower]
    # 根据优化器类型动态过滤参数
    if opt_name_lower == 'lion':
        # Lion 不支持 eps 参数，过滤掉
        opt_kwargs = {k: v for k, v in opt_kwargs.items() if k != 'eps'}
    optimizer = optimizer_class(
        filter(lambda p: p.requires_grad, model.parameters()), **opt_kwargs
    )

    return optimizer

def get_optimizer_with_layerwise_decay(model, opt_name, opt_kwargs):
    # 构建分层学习率衰减的参数组
    lr = opt_kwargs['lr']
    param_groups = lrd.param_groups_lrd(
        model,
        lr,
        weight_decay=opt_kwargs.get('weight_decay', 0.02),
        no_weight_decay_list=model.module.backbone.no_weight_decay(),
        layer_decay=0.75  # 分层学习率衰减因子
    )

    # 扩展优化器支持 Lion
    optimizer_dict = {
        'sgd': torch.optim.SGD,
        'adam': torch.optim.Adam,
        'adamw': torch.optim.AdamW,
    }

    if LION_AVAILABLE:
        optimizer_dict['lion'] = Lion

    # 检查优化器名称是否合法
    opt_name_lower = opt_name.lower()
    if opt_name_lower not in optimizer_dict:
        raise ValueError(f"Unsupported optimizer: {opt_name}. Available: {list(optimizer_dict.keys())}")

    # 初始化优化器
    optimizer_class = optimizer_dict[opt_name_lower]
    # 根据优化器类型动态过滤参数
    if opt_name_lower == 'lion':
        # Lion 不支持 eps 参数，过滤掉
        opt_kwargs = {k: v for k, v in opt_kwargs.items() if k != 'eps'}
    optimizer = optimizer_class(param_groups, **opt_kwargs)

    return optimizer