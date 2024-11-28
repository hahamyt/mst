from functools import partial
from typing import Optional, Sequence, Tuple, Union
import torch
from torch import Tensor, nn
from isegm.utils.log import logger
from .models import Block
from .patch_embed import FlexiPatchEmbed
from .utils import resize_abs_pos_embed, to_2tuple
from .pos_embed import interpolate_pos_embed

class AdaptiveVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size: int = 240,
        patch_size=(16, 16),
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: int = 4,
        qkv_bias: bool = True,
        pre_norm: bool = False,
        drop_rate: float = 0,
        attn_drop_rate: float = 0,
        embed_layer: nn.Module = FlexiPatchEmbed,  # type:ignore
        norm_layer: Optional[nn.Module] = None,
        act_layer: Optional[nn.Module] = None,
        patch_size_seq: Sequence[int] = (8, 16, 28),
        base_pos_embed_size: Union[int, Tuple[int, int]] = 7,
        patch_size_probs: Optional[Sequence[float]] = None,
        interpolation: str = "bicubic",
        antialias: bool = True,
        proj_drop_rate=0.,
        is_visdebug=False
    ) -> None:
        super().__init__()
        assert embed_layer == FlexiPatchEmbed, "embed_layer should be a FlexiPatchEmbed"
        self.is_visdebug = is_visdebug
        if is_visdebug:
            import visdom
            self.viz = visdom.Visdom()
        # Position embedding resizing function
        self.resize_pos_embed = partial(
            resize_abs_pos_embed,
            # old_size=base_pos_embed_size,
            interpolation=interpolation,
            antialias=antialias,
            num_prefix_tokens=1 #  if class_token and not no_embed_class else 0,
        )
        self.patch_size = patch_size
        self.img_size = to_2tuple(img_size)

        base_pos_embed_size = img_size[0]//patch_size[0] #
        embed_layer_fn = partial(
            FlexiPatchEmbed,
            patch_size_seq=patch_size_seq,
            patch_size_probs=patch_size_probs,
            grid_size=base_pos_embed_size,
            interpolation=interpolation,
            antialias=antialias,
        )

        self.patch_embed = embed_layer_fn(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
        )

        self.pos_embed = nn.Parameter(torch.randn(1, self.patch_embed.num_patches+1, embed_dim) * .02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=False)  # 使用Adapter，则不请求梯度
        self.img_size = to_2tuple(img_size)
        norm_layer = norm_layer if norm_layer else partial(nn.LayerNorm, eps=1e-6)
        intervel = 4
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  attn_drop=attn_drop_rate, proj_drop=proj_drop_rate, norm_layer=norm_layer,
                  act_layer=act_layer, use_score=((d + 1) % intervel == 0), depth=d)
            for d in range(depth)])

        # self.fc_norm = norm_layer(embed_dim)

    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def shuffle(self, x):
        """
        in: x (B, N, C)
        out: x_shuffle (B, N, C), ids_restore (B, N)
        """
        B, N, C = x.shape
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        x_shuffle = torch.gather(x, 1, index=ids_shuffle.unsqueeze(-1).repeat(1, 1, C))

        return x_shuffle, ids_restore

    def unshuffle(self, x, ids_restore):
        B, N, C = x.shape
        x_unshuffle = torch.gather(x, 1, index=ids_restore.unsqueeze(-1).repeat(1, 1, C))

        return x_unshuffle

    def split(self, x):
        B, N, C = x.shape
        num_tokens_per_split = 224 * 224
        num_splits = max(1, N // num_tokens_per_split)
        out = []
        for i in range(num_splits):
            if i == num_splits - 1:
                out.append(x[:, i*num_tokens_per_split:])
                return out
            out.append(x[:, i*num_tokens_per_split:(i+1)*num_tokens_per_split])

    # window split for finetuning on larger size (the pretraining size should be 224 x 224)
    def patchify(self, x):
        """
        in: (B, N, C)
        out: (B*win_w*win_h, N//(win_w*win_h), C)
        """
        B, N, C = x.shape
        grid_h, grid_w = self.patch_embed.grid_size
        win_h_grid = 224 // self.patch_embed.patch_size[0]
        win_w_grid = 224 // self.patch_embed.patch_size[1]
        win_h, win_w = grid_h // win_h_grid, grid_w // win_w_grid
        x = x.view(B, win_h, grid_h // win_h, win_w, grid_w // win_w, C)
        x_patchified = x.permute((0, 1, 3, 2, 4, 5)).contiguous()
        x_patchified = x_patchified.view(B * win_h * win_w, grid_h * grid_w // (win_h * win_w), C)

        return x_patchified

    # recover the window split
    def unpatchify(self, x):
        """
        in: (B*win_h*win_w, N//(win_h*win_w), C)
        out: (B, N, C)
        """
        B, N, C = x.shape
        grid_h, grid_w = self.patch_embed.grid_size
        win_h_grid = 224 // self.patch_embed.patch_size[0]
        win_w_grid = 224 // self.patch_embed.patch_size[1]
        win_h, win_w = grid_h // win_h_grid, grid_w // win_w_grid
        x = x.view(B // (win_h * win_w), win_h, win_w, grid_h // win_h, grid_w // win_w, C)
        x = x.permute((0, 1, 3, 2, 4, 5)).contiguous().view(B // (win_h * win_w), win_h * win_w * N, C)

        return x

    def pt2patchidx(self, points, im_sz, patch_sz=16):
        patchs = torch.div(points, patch_sz, rounding_mode='floor')
        idxs = patchs[:, :, 0] * (im_sz // patch_sz) + patchs[:, :, 1]
        idxs = idxs.int()
        return idxs

    def _pos_embed(self, x: Tensor, patch_size: Tuple[int, int], return_posemb=False):
        # Resize position embedding based on current patch size
        new_size = (
            int(self.img_size[0] // patch_size[0]),
            int(self.img_size[1] // patch_size[1]),
        )
        pos_embed = self.resize_pos_embed(self.pos_embed, new_size)

        # Position embedding does not overlap with class token, add then concat
        x = x + pos_embed[:, 1:]
        # if self.cls_token is not None:
        #     x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        if return_posemb:
            return self.pos_drop(x), pos_embed[:, 1:]
        else:
            return self.pos_drop(x)

    def forward_features(
        self, x: Tensor, af: Tensor=None, points: Tensor=None
    ) -> Tensor:
        patch_size = self.patch_size
        p_sz8 = [8, 8] # x // 2 for x in patch_size]
        p_sz16 = [x for x in patch_size]
        p_sz28 = [32, 32] # int(x*1.75) for x in patch_size]

        idxs8 = self.pt2patchidx(points, x.shape[2], patch_sz=p_sz8[0])
        idxs16 = self.pt2patchidx(points, x.shape[2], patch_sz=p_sz16[0])
        idxs28 = self.pt2patchidx(points, x.shape[2], patch_sz=p_sz28[0])

        self.img_size = to_2tuple(x.shape[-1])

        # 用于交互
        x_8 = self.patch_embed(x, p_sz8, return_patch_size=False)
        x_28 = self.patch_embed(x, p_sz28, return_patch_size=False)

        x_8, pos_emb_8 = self._pos_embed(x_8, p_sz8, return_posemb=True)
        x_28, pos_emb_28 = self._pos_embed(x_28, p_sz28, return_posemb=True)

        x_interact = [[x_8, pos_emb_8], [x_28, pos_emb_28]]
        # 原始的
        x_16 = self.patch_embed(x, p_sz16, return_patch_size=False)
        if af is not None:
            x_16 += af

        x_16 = self._pos_embed(x_16, p_sz16)

        num_blocks = len(self.blocks)
        all_tokens = []
        for i in range(1, num_blocks + 1):
            x_16, x_interact, tokens = self.blocks[i - 1](x_16, x_interact, idxs=[idxs16, idxs8, idxs28],  im_sz=self.img_size[0], blk_depth=i)
            if len(tokens) != 0:
                all_tokens.append(tokens)

        return x_16, all_tokens

    def forward_backbone(
        self, x: Tensor, additional_features=None, points=None
    ) -> Tensor:
        x = self.forward_features(x, additional_features, points)
        return x


    def init_weights_from_pretrained(self, pretrained_path):
        if pretrained_path:
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            logger.info("Load pretrained checkpoint from: %s" % pretrained_path)
            checkpoint_model = checkpoint['model']

            # interpolate position embedding
            interpolate_pos_embed(self, checkpoint_model)

            # load pre-trained model
            msg = self.load_state_dict(checkpoint_model, strict=False)
            # logger.info(msg)
        else:
            logger.info("No pretrained checkpoint was used.")

def flexivit_tiny(**kwargs) -> AdaptiveVisionTransformer:
    return AdaptiveVisionTransformer(embed_dim=192, depth=12, num_heads=3, **kwargs)


def flexivit_small(**kwargs) -> AdaptiveVisionTransformer:
    return AdaptiveVisionTransformer(embed_dim=384, depth=12, num_heads=6, **kwargs)


def flexivit_base(**kwargs) -> AdaptiveVisionTransformer:
    return AdaptiveVisionTransformer(embed_dim=768, depth=12, num_heads=12, **kwargs)


def flexivit_large(**kwargs) -> AdaptiveVisionTransformer:
    return AdaptiveVisionTransformer(embed_dim=1024, depth=24, num_heads=16, **kwargs)


def flexivit_huge(**kwargs) -> AdaptiveVisionTransformer:
    return AdaptiveVisionTransformer(embed_dim=1280, depth=32, num_heads=16, **kwargs)
