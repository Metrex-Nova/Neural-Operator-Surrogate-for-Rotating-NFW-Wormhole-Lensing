import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPBaseline(nn.Module):
    def __init__(self, n_params=2, n_out=150, hidden=512, n_layers=4):
        super().__init__()
        layers = [nn.Linear(n_params, hidden), nn.GELU()]
        for _ in range(n_layers - 1):
            layers += [nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU()]
        layers.append(nn.Linear(hidden, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, params):
        return self.net(params)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CNNBaseline(nn.Module):
    def __init__(self, n_params=2, n_out=150, stem_dim=256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(n_params, 128), nn.GELU(),
            nn.Linear(128, stem_dim), nn.GELU(),
        )

        self.init_len = 4
        self.init_ch  = stem_dim // self.init_len

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(self.init_ch, 128, kernel_size=4, stride=4), nn.GELU(),
            nn.ConvTranspose1d(128, 64,       kernel_size=4, stride=4), nn.GELU(),
            nn.ConvTranspose1d(64,  32,       kernel_size=4, stride=4), nn.GELU(),
            nn.Conv1d(32, 1, kernel_size=3, padding=1),
        )
        self.n_out = n_out

    def forward(self, params):
        x = self.stem(params)
        x = x.view(x.shape[0], self.init_ch, self.init_len)
        x = self.decoder(x)
        x = F.interpolate(x, size=self.n_out, mode='linear', align_corners=False)
        return x.squeeze(1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
