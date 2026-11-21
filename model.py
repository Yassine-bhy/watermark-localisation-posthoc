"""Step 2 model: Watermark Anything's pretrained extractor (SAM ViT-B encoder + pixel decoder, MIT weights in
external/wam) with a new last 1x1 layer: for each method one "watermarked?" channel and one channel per message bit.
    net = LocNet(res=256); det, bits = net(x, method_idx)   x: B x 3 x 512 x 512 in [0, 1]
    det: B x 128 x 128 logits (channel of each sample's own method); bits: B x 256 x 128 x 128 logits
    (first nbits of the sample's method are meaningful). The image is resized to `res` before the encoder."""
import json
import sys

import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F

import data

WAM = data.HERE / "external" / "wam"
sys.path.insert(0, str(WAM))
from watermark_anything.modules.pixel_decoder import PixelDecoder  # noqa: E402
from watermark_anything.modules.vit import ImageEncoderViT  # noqa: E402

NBITS = {"pixelseal": 256, "trustmark": 100, "stegastamp": 100}
BIT_OFF = {"pixelseal": 0, "trustmark": 256, "stegastamp": 356}   # where each method's bits live in the head
N_OUT = 3 + 456
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class LocNet(nn.Module):
    def __init__(self, res=256, pretrained=True):
        super().__init__()
        cfg = omegaconf.OmegaConf.load(WAM / "configs" / "extractor.yaml")["sam_base"]
        cfg.encoder.img_size = res; cfg.pixel_decoder.nbits = 32
        self.res = res
        self.enc = ImageEncoderViT(**cfg.encoder)
        self.dec = PixelDecoder(**cfg.pixel_decoder)
        if pretrained:
            sd = torch.load(WAM / "checkpoints" / "wam_mit.pth", map_location="cpu", weights_only=True)
            sd = {k[len("detector."):]: v for k, v in sd.items() if k.startswith("detector.")}
            pe = sd["image_encoder.pos_embed"]                       # 1 x 16 x 16 x 768 (trained at 256)
            if pe.shape[1] != res // 16:
                pe = F.interpolate(pe.permute(0, 3, 1, 2), size=(res // 16,) * 2, mode="bicubic", align_corners=False).permute(0, 2, 3, 1)
                sd["image_encoder.pos_embed"] = pe
            own = self.enc.state_dict()
            for k in list(sd):                                      # global-attention relative positions
                kk = k[len("image_encoder."):]
                if "rel_pos" in k and kk in own and sd[k].shape != own[kk].shape:
                    sd[k] = F.interpolate(sd[k].T[None], size=own[kk].shape[0], mode="linear", align_corners=False)[0].T
            self.enc.load_state_dict({k[len("image_encoder."):]: v for k, v in sd.items() if k.startswith("image_encoder.")})
            self.dec.load_state_dict({k[len("pixel_decoder."):]: v for k, v in sd.items() if k.startswith("pixel_decoder.")})
        old = self.dec.last_layer
        self.head = nn.Conv2d(old.in_channels, N_OUT, 1)
        with torch.no_grad():                                       # start each detection channel from WAM's
            nn.init.normal_(self.head.weight, std=1e-3); nn.init.zeros_(self.head.bias)
            self.head.weight[:3] = old.weight[0:1]; self.head.bias[:3] = old.bias[0]
        self.dec.last_layer = nn.Identity()

    def forward(self, x, method_idx):
        x = F.interpolate(x, size=(self.res, self.res), mode="bilinear", align_corners=False, antialias=True) if self.res != x.shape[-1] else x
        f = self.dec(self.enc((x - MEAN.to(x)) / STD.to(x)))      # B x 48 x res x res
        f = F.adaptive_avg_pool2d(f, 128)
        out = self.head(f)
        det = out[torch.arange(len(x)), method_idx]                 # B x 128 x 128
        return det, out[:, 3:]


def method_bits(bits_all, method):
    o = BIT_OFF[method]; return bits_all[:, o:o + NBITS[method]]


def save(net, path, **info):
    torch.save({"state": net.state_dict(), "res": net.res, **info}, path)
    (path.with_suffix(".json")).write_text(json.dumps({k: v for k, v in info.items()}, default=str, indent=1))


def load(path, device="cuda"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = LocNet(res=ck["res"], pretrained=False); net.load_state_dict(ck["state"]); return net.to(device).eval()
