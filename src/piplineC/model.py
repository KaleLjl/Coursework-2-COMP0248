import torch
from torch_points3d.applications.pointnet2 import PointNet2


class TableSegmentationModel(torch.nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # Using PointNet++ with SSG (Single Scale Grouping)
        self.backbone = PointNet2(
            architecture="unet",
            input_nc=3,  # x, y, z coordinates
            num_layers=4,
            output_nc=64
        )

        # Classification head
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(64, 32),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(32, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: point cloud [B, N, 3]
        Returns:
            point-wise logits [B, N, 2]
        """
        # Reshape for backbone
        batch_size, num_points, _ = x.shape
        x = x.transpose(1, 2).contiguous()  # [B, 3, N]

        # Get features
        features = self.backbone(x)

        # Get per-point predictions
        logits = self.classifier(features)

        # Reshape to [B, N, 2]
        logits = logits.transpose(1, 2).contiguous()

        return logits
