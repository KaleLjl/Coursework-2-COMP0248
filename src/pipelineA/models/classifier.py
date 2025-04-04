import torch
import torch.nn as nn
import torch.nn.functional as F
    
def knn(x, k):
    """k-nearest neighbors.
    
    Args:
        x (torch.Tensor): Input points (B, N, C)
        k (int): Number of neighbors
        
    Returns:
        torch.Tensor: Indices of nearest neighbors (B, N, k)
    """
    inner = -2 * torch.matmul(x, x.transpose(2, 1))
    xx = torch.sum(x**2, dim=2, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    
    # Find k nearest neighbors (excluding self)
    idx = pairwise_distance.topk(k=k+1, dim=-1)[1][:, :, 1:]  # (B, N, k)
    return idx

def get_graph_feature(x, k=20, idx=None):
    """Get edge features for each point.
    
    Args:
        x (torch.Tensor): Input points (B, C, N)
        k (int): Number of neighbors
        idx (torch.Tensor, optional): Precomputed indices of neighbors
        
    Returns:
        torch.Tensor: Edge features (B, 2*C, N, k)
    """
    batch_size, num_dims, num_points = x.size()
    
    # Transpose to (B, N, C) for kNN
    x_transposed = x.transpose(2, 1)
    
    # Get indices of nearest neighbors if not provided
    if idx is None:
        idx = knn(x_transposed, k=k)  # (B, N, k)
    
    # Force idx to have the right dtype
    idx = idx.to(dtype=torch.long)
    
    # Expand indices for gather operation
    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    
    # Gather neighbors
    x_transposed = x_transposed.contiguous()
    neighbors = x_transposed.view(batch_size * num_points, -1)[idx, :]
    neighbors = neighbors.view(batch_size, num_points, k, num_dims)
    
    # Reshape x for broadcasting
    x_tiled = x_transposed.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    # Concatenate central points with neighbors
    edge_feature = torch.cat([x_tiled, neighbors - x_tiled], dim=3)
    
    # Reshape to (B, 2*C, N, k)
    edge_feature = edge_feature.permute(0, 3, 1, 2)
    
    return edge_feature

class EdgeConv(nn.Module):
    """Edge Convolution layer for N."""
    
    def __init__(self, in_channels, out_channels, k=20):
        """Initialize EdgeConv layer.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            k (int): Number of neighbors
        """
        super(EdgeConv, self).__init__()
        self.k = k
        # Use GroupNorm which works with any batch size
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(4, out_channels),  # GroupNorm instead of BatchNorm
            nn.LeakyReLU(negative_slope=0.2)
        )
    
    def forward(self, x):
        """Forward pass.
        
        Args:
            x (torch.Tensor): Input points (B, C, N)
            
        Returns:
            torch.Tensor: Output features (B, C', N)
        """
        # Get edge features
        edge_features = get_graph_feature(x, k=self.k)  # (B, 2*C, N, k)
        
        # Apply convolution
        edge_features = self.conv(edge_features)  # (B, C', N, k)
        
        # Max pooling over neighbors
        x = edge_features.max(dim=-1, keepdim=False)[0]  # (B, C', N)
        
        return x

class DGCNN(nn.Module):
    """Dynamic Graph CNN for point cloud classification."""
    
    def __init__(self, num_classes=2, k=20, emb_dims=1024, dropout=0.5, feature_dropout=0.0):
        """Initialize DGCNN model.
        
        Args:
            num_classes (int): Number of output classes
            k (int): Number of neighbors
            emb_dims (int): Embedding dimensions
            dropout (float): Dropout rate for final classifier MLP
            feature_dropout (float): Dropout rate after EdgeConv layers
        """
        super(DGCNN, self).__init__()
        self.feature_dropout_rate = feature_dropout # Store feature dropout rate
        
        # Edge convolution layers
        self.edge_conv1 = EdgeConv(3, 64, k=k)
        self.edge_conv2 = EdgeConv(64, 64, k=k)
        self.edge_conv3 = EdgeConv(64, 128, k=k)
        self.edge_conv4 = EdgeConv(128, 256, k=k)
        
        # MLP for global features - using GroupNorm
        self.mlp = nn.Sequential(
            nn.Conv1d(512, emb_dims, kernel_size=1, bias=False),
            nn.GroupNorm(16, emb_dims),  # GroupNorm instead of BatchNorm
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        # Final MLP classifier - using LayerNorm
        self.fc1 = nn.Linear(emb_dims, 512)
        self.ln1 = nn.LayerNorm(512)  # LayerNorm instead of BatchNorm
        self.act1 = nn.LeakyReLU(negative_slope=0.2)
        self.drop1 = nn.Dropout(p=dropout)
        
        self.fc2 = nn.Linear(512, 256)
        self.ln2 = nn.LayerNorm(256)  # LayerNorm instead of BatchNorm
        self.act2 = nn.LeakyReLU(negative_slope=0.2)
        self.drop2 = nn.Dropout(p=dropout)
        
        self.fc3 = nn.Linear(256, 64)
        self.ln3 = nn.LayerNorm(64)  # LayerNorm instead of BatchNorm
        self.act3 = nn.LeakyReLU(negative_slope=0.2)
        self.drop3 = nn.Dropout(p=dropout)
        
        self.fc4 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        """Forward pass.
        
        Args:
            x (torch.Tensor): Input point cloud (B, N, 3)
            
        Returns:
            torch.Tensor: Class logits (B, num_classes)
        """
        batch_size = x.size(0)
            
        # Transpose to (B, 3, N)
        x = x.transpose(2, 1)
        
        # Apply edge convolutions with optional feature dropout
        x1 = self.edge_conv1(x)
        if self.feature_dropout_rate > 0 and self.training:
             x1 = F.dropout(x1, p=self.feature_dropout_rate)
             
        x2 = self.edge_conv2(x1)
        if self.feature_dropout_rate > 0 and self.training:
             x2 = F.dropout(x2, p=self.feature_dropout_rate)
             
        x3 = self.edge_conv3(x2)
        if self.feature_dropout_rate > 0 and self.training:
             x3 = F.dropout(x3, p=self.feature_dropout_rate)
             
        x4 = self.edge_conv4(x3)
        if self.feature_dropout_rate > 0 and self.training:
             x4 = F.dropout(x4, p=self.feature_dropout_rate)
        
        # Concatenate features
        x = torch.cat([x1, x2, x3, x4], dim=1) # Shape: (B, 64+64+128+256=512, N)
        
        # MLP to get global features
        x = self.mlp(x)
        
        # Global max pooling
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        
        # Classification with layer norm
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.act1(x)
        x = self.drop1(x)
        
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.act2(x)
        x = self.drop2(x)
        
        x = self.fc3(x)
        x = self.ln3(x)
        x = self.act3(x)
        x = self.drop3(x)
        
        x = self.fc4(x)
        
        return x

class PointNet(nn.Module):
    """PointNet for point cloud classification."""
    
    def __init__(self, num_classes=2, dropout=0.5):
        """Initialize PointNet model.
        
        Args:
            num_classes (int): Number of output classes
            dropout (float): Dropout rate
        """
        super(PointNet, self).__init__()
        
        # MLP for point-wise features - using GroupNorm
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.gn1 = nn.GroupNorm(4, 64)  # GroupNorm
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.gn2 = nn.GroupNorm(4, 64)  # GroupNorm
        
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.gn3 = nn.GroupNorm(4, 64)  # GroupNorm
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.gn4 = nn.GroupNorm(8, 128)  # GroupNorm
        self.conv5 = nn.Conv1d(128, 1024, kernel_size=1, bias=False)
        self.gn5 = nn.GroupNorm(16, 1024)  # GroupNorm
        
        # MLP for classification - using LayerNorm
        self.fc1 = nn.Linear(1024, 512)
        self.ln1 = nn.LayerNorm(512)  # LayerNorm
        self.drop1 = nn.Dropout(p=dropout)
        
        self.fc2 = nn.Linear(512, 256)
        self.ln2 = nn.LayerNorm(256)  # LayerNorm
        self.drop2 = nn.Dropout(p=dropout)
        
        self.fc3 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        """Forward pass.
        
        Args:
            x (torch.Tensor): Input point cloud (B, N, 3)
            
        Returns:
            torch.Tensor: Class logits (B, num_classes)
        """
        batch_size = x.size(0)
        
        # Transpose to (B, 3, N)
        x = x.transpose(2, 1)
        
        # Point-wise features with GroupNorm
        x = F.relu(self.gn1(self.conv1(x)))
        x = F.relu(self.gn2(self.conv2(x)))
        
        x = F.relu(self.gn3(self.conv3(x)))
        x = F.relu(self.gn4(self.conv4(x)))
        x = F.relu(self.gn5(self.conv5(x)))
        
        # Global max pooling
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        
        # Classification with LayerNorm
        x = self.fc1(x)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.drop1(x)
        
        x = self.fc2(x)
        x = self.ln2(x)
        x = F.relu(x)
        x = self.drop2(x)
        
        x = self.fc3(x)
        
        return x

def get_model(model_type='dgcnn', num_classes=2, **kwargs):
    """Get model based on model type.
    
    Args:
        model_type (str): Model type ('dgcnn' or 'pointnet')
        num_classes (int): Number of output classes
        **kwargs: Additional model parameters
        
    Returns:
        nn.Module: Model
    """
    if model_type == 'dgcnn':
        return DGCNN(
            num_classes=num_classes,
            k=kwargs.get('k', 20),
            emb_dims=kwargs.get('emb_dims', 1024),
            dropout=kwargs.get('dropout', 0.5),
            feature_dropout=kwargs.get('feature_dropout', 0.0) # Add feature_dropout
        )
    elif model_type == 'pointnet':
        # PointNet doesn't have feature dropout in this implementation
        return PointNet(
            num_classes=num_classes,
            dropout=kwargs.get('dropout', 0.5)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
