"""Checkpoint-compatible architecture; preprocessing is kept separate."""
import torch
from torch import nn
from torch.nn import functional as F

class TerraMindMethanePulse(nn.Module):
    def __init__(self, encoder, embed_dim=384, img_size=512):
        super().__init__()
        self.encoder  = encoder
        self.img_size = img_size
        self.seg_decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64,  kernel_size=2, stride=2),
            nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64,  32,  kernel_size=2, stride=2),
            nn.BatchNorm2d(32),  nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )
        self.severity_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        B = x.shape[0]
        x_224 = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        with torch.no_grad():
            features = self.encoder(x_224)
        feat     = features[-1]
        feat_map = feat.permute(0, 2, 1).contiguous().reshape(B, 384, 14, 14)
        seg_logits = self.seg_decoder(feat_map)
        seg_logits = F.interpolate(seg_logits, size=(512, 512),
                                   mode='bilinear', align_corners=False)
        pooled          = feat.mean(dim=1)
        severity_logits = self.severity_head(pooled)
        return seg_logits, severity_logits

