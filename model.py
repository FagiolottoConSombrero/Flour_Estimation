import torch.nn as nn
import torch.nn.functional as F

class HSILLPMLP(nn.Module):
    """
    MLP per LLP su patch HSI [B, 121, 16, 16].

    - Input:  X [B, 121, H, W]
    - Output: logits [B, P, K] con P = H*W (pixel), K = n_classi
    """
    def __init__(self, in_bands=121, n_classes=5, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.in_bands = in_bands
        self.n_classes = n_classes

        self.fc1 = nn.Linear(in_bands, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim*2)
        self.fc3 = nn.Linear(hidden_dim*2, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, n_classes)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [B, 121, H, W]
        ritorna: logits [B, P, K] con P = H*W
        """
        B, C, H, W = x.shape

        # Porta le bande come ultima dimensione: [B, H, W, C]
        x = x.permute(0, 2, 3, 1).contiguous()

        # Flatten dei pixel: [B, H*W, C]
        P = H * W
        x = x.view(B * P, C)   # [B*P, C]

        # MLP per-pixel
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = F.relu(x)
        x = self.dropout(x)

        logits = self.fc_out(x)   # [B*P, K]

        # Torna a [B, P, K]
        logits = logits.view(B, P, self.n_classes)
        return logits
