import torch
import torch.nn as nn
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1

class IdentityLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        # 加载预训练的 FaceNet (InceptionResnetV1)
        # classify=False 表示提取 512 维特征向量
        self.net = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        # 冻结参数，不参与训练
        for param in self.net.parameters():
            param.requires_grad = False

    def forward(self, pred, gt):
        # pred, gt 范围通常是 [-1, 1], 需要缩放到 FaceNet 喜欢的范围
        # 同时 FaceNet 输入通常需要 resize 到 160x160
        pred = F.interpolate(pred, size=(160, 160), mode='bilinear', align_corners=False)
        gt = F.interpolate(gt, size=(160, 160), mode='bilinear', align_corners=False)
        
        # 提取特征
        feat_pred = self.net(pred)
        feat_gt = self.net(gt)
        
        # 计算余弦相似度损失: 相似度越高，损失越小
        # cosine_similarity 返回 1 表示完全一样，-1 表示完全相反
        loss = 1 - F.cosine_similarity(feat_pred, feat_gt).mean()
        return loss