import torch
import torch.nn as nn
import torch.nn.functional as F

def mask_to_boundary(mask, dilation_ratio=0.02):
    # mask: [B, 1, H, W] or [B, H, W]
    if mask.dim() == 4:
        mask = mask[:, 0]
    batch, h, w = mask.shape
    device = mask.device
    boundary = torch.zeros_like(mask)
    for i in range(batch):
        m = mask[i].float()
        # Use max pooling to get boundary
        kernel = int(round(dilation_ratio * max(h, w)))
        if kernel < 1:
            kernel = 1
        pad = kernel // 2
        mp = F.max_pool2d(m.unsqueeze(0).unsqueeze(0), kernel, stride=1, padding=pad)
        boundary[i] = (mp[0,0] - m).abs() > 0
    return boundary.unsqueeze(1)

class BoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
    def forward(self, pred, gt):
        pred_prob = torch.sigmoid(pred)
        pred_boundary = mask_to_boundary(pred_prob > 0.5)
        gt_boundary = mask_to_boundary(gt > 0.5)
        return self.bce(pred_boundary.float(), gt_boundary.float())
