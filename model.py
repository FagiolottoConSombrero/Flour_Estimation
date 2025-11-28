import torch.nn as nn
import torch.nn.functional as F
#from torchinfo import summary


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
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
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
        #x = self.dropout(x)

        x = self.fc2(x)
        x = F.relu(x)
        #x = self.dropout(x)

        x = self.fc3(x)
        x = F.relu(x)

        logits = self.fc_out(x)   # [B*P, K]

        # Torna a [B, P, K]
        logits = logits.view(B, P, self.n_classes)
        return logits


class HSILSpectralCNN(nn.Module):
    """
    Spectral CNN per LLP su patch HSI [B, 121, 16, 16].

    - Input:  X [B, 121, H, W]
    - Output: logits [B, P, K] con P = H*W (pixel), K = n_classi

    Idea:
      - Ogni pixel è uno spettro di lunghezza in_bands.
      - Applico Conv1D lungo la dimensione spettrale per estrarre "feature di forma".
      - Poi un piccolo MLP finale porta alle K classi.
    """
    def __init__(self, in_bands=121, n_classes=5,
                 hidden_dim=256, dropout=0.0, conv_channels=64):
        super().__init__()
        self.in_bands = in_bands
        self.n_classes = n_classes

        # Blocchi spettrali 1D: input per pixel = [N, 1, in_bands]
        self.conv1 = nn.Conv1d(1, conv_channels, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(conv_channels)

        self.conv2 = nn.Conv1d(conv_channels, conv_channels, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(conv_channels)

        self.conv3 = nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(conv_channels)

        # Pooling globale sullo spettro → embedding compatto per pixel
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # MLP finale per andare a K classi
        self.fc1 = nn.Linear(conv_channels, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, n_classes)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [B, 121, H, W]
        ritorna: logits [B, P, K] con P = H*W
        """
        B, C, H, W = x.shape
        assert C == self.in_bands, f"Mi aspetto {self.in_bands} bande, ma ho {C}"

        # Porta le bande come ultima dimensione: [B, H, W, C]
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, 121]

        # Flatten dei pixel: [B*P, C]
        P = H * W
        x = x.view(B * P, C)   # [B*P, 121]

        # Aggiungo dimensione canale per conv1d: [B*P, 1, 121]
        x = x.unsqueeze(1)

        # --- blocco spettrale ---
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        # global average pooling lungo lo spettro: [B*P, conv_channels, 1]
        x = self.global_pool(x)
        x = x.squeeze(-1)   # [B*P, conv_channels]

        # --- MLP finale per-pixel ---
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)

        logits = self.fc_out(x)   # [B*P, K]

        # Torna a [B, P, K]
        logits = logits.view(B, P, self.n_classes)
        return logits
'''model = HSILSpectralCNN()
model.eval()
input1 = torch.randn(1, 121, 16, 16)
summary(model=model,
        input_data=input1,
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
)'''