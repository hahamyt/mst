import torch.nn as nn

from isegm.utils.serialization import serialize
from .is_model import ISModel
from .modeling.adaptive_vit import AdaptiveVisionTransformer
from .modeling.modules_vit import PatchEmbed
from .modeling.swin_transformer import SwinTransfomerSegHead


class SimpleFPN(nn.Module):
    def __init__(self, in_dim=768, out_dims=[128, 256, 512, 1024]):
        super().__init__()
        self.down_4_chan = max(out_dims[0]*2, in_dim // 2)
        self.down_4 = nn.Sequential(
            nn.ConvTranspose2d(in_dim, self.down_4_chan, 2, stride=2),
            nn.GroupNorm(1, self.down_4_chan),
            nn.GELU(),
            nn.ConvTranspose2d(self.down_4_chan, self.down_4_chan // 2, 2, stride=2),
            nn.GroupNorm(1, self.down_4_chan // 2),
            nn.Conv2d(self.down_4_chan // 2, out_dims[0], 1),
            nn.GroupNorm(1, out_dims[0]),
            nn.GELU()
        )
        self.down_8_chan = max(out_dims[1], in_dim // 2)
        self.down_8 = nn.Sequential(
            nn.ConvTranspose2d(in_dim, self.down_8_chan, 2, stride=2),
            nn.GroupNorm(1, self.down_8_chan),
            nn.Conv2d(self.down_8_chan, out_dims[1], 1),
            nn.GroupNorm(1, out_dims[1]),
            nn.GELU()
        )
        self.down_16 = nn.Sequential(
            nn.Conv2d(in_dim, out_dims[2], 1),
            nn.GroupNorm(1, out_dims[2]),
            nn.GELU()
        )
        self.down_32_chan = max(out_dims[3], in_dim * 2)
        self.down_32 = nn.Sequential(
            nn.Conv2d(in_dim, self.down_32_chan, 2, stride=2),
            nn.GroupNorm(1, self.down_32_chan),
            nn.Conv2d(self.down_32_chan, out_dims[3], 1),
            nn.GroupNorm(1, out_dims[3]),
            nn.GELU()
        )

        self.init_weights()

    def init_weights(self):
        pass

    def forward(self, x):
        x_down_4 = self.down_4(x)
        x_down_8 = self.down_8(x)
        x_down_16 = self.down_16(x)
        x_down_32 = self.down_32(x)

        return [x_down_4, x_down_8, x_down_16, x_down_32]


class AdaptiveVitModel(ISModel):
    @serialize
    def __init__(
        self,
        backbone_params={},
        neck_params={},
        head_params={},
        random_split=False,
        **kwargs
        ):

        super().__init__(**kwargs)
        self.random_split = random_split

        self.backbone = AdaptiveVisionTransformer(**backbone_params)
        self.neck = SimpleFPN(**neck_params)
        self.head = SwinTransfomerSegHead(**head_params)

        self.patch_embed_coords = PatchEmbed(
            img_size= backbone_params['img_size'],
            patch_size=backbone_params['patch_size'],
            in_chans=3 if self.with_prev_mask else 2,
            embed_dim=backbone_params['embed_dim'],
        )

        # self.freeze_backbone()
        # self.freeze_rest()


    def freeze_backbone(self):
        for name, param in self.backbone.named_parameters():
            if 'score' in name or 'cross' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def freeze_rest(self):
        for name, param in self.neck.named_parameters():
            param.requires_grad = False
        # for name, param in self.head.named_parameters():
        #     param.requires_grad = False
        for name, param in self.patch_embed_coords.named_parameters():
            param.requires_grad = False


    def backbone_forward(self, image, coord_features=None, points=None):
        coord_features = self.patch_embed_coords(coord_features)
        backbone_features, tokens = self.backbone.forward_backbone(image, additional_features=coord_features, points=points)
        # Extract 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        B, N, C = backbone_features.shape
        gsz16 = image.shape[-1] // self.backbone.patch_size[0]
        x = backbone_features.transpose(-1, -2).view(B, C, gsz16, gsz16)
        multi_scale_features = self.neck(x)

        return {'instances': self.head(multi_scale_features), 'instances_aux': None, 'tokens': tokens}

'''
docker run -itd --gpus all --name gpu_server --privileged -p 7999:22 -p 8098:8097 -p 6007:6006 --shm-size 32g -v /home/x/workspace:/root/workspace -v /home/x/datasets:/root/datasets -v /home/x/softwares:/root/softwares -v /home/x/tmpdata/:/root/tmpdata gpu_server:v0.2
'''