import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes        = modes

        scale     = 1.0 / (in_channels * out_channels)
        self.w_re = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes))
        self.w_im = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes))

    def forward(self, x):
        N    = x.shape[-1]
        x_ft = torch.fft.rfft(x)

        w      = torch.complex(self.w_re, self.w_im)
        out_ft = torch.zeros(x.shape[0], self.out_channels, N // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes] = torch.einsum(
            "bim,iom->bom", x_ft[:, :, :self.modes], w)

        return torch.fft.irfft(out_ft, n=N)


class FNOBlock(nn.Module):
    # No normalisation: InstanceNorm collapses sparse intensity profiles to zero
    def __init__(self, width, modes):
        super().__init__()
        self.spectral = SpectralConv1d(width, width, modes)
        self.w        = nn.Conv1d(width, width, kernel_size=1)

    def forward(self, x):
        return F.gelu(self.spectral(x) + self.w(x))


class FNO1d(nn.Module):
    # modes <= N_GRID/4 avoids spectral degeneracy
    def __init__(self, modes=24, width=32, n_layers=4, in_ch=3, out_ch=1):
        super().__init__()
        self.fc0    = nn.Linear(in_ch, width)
        self.blocks = nn.ModuleList([FNOBlock(width, modes) for _ in range(n_layers)])
        self.fc1    = nn.Linear(width, 128)
        self.fc2    = nn.Linear(128, out_ch)

    def forward(self, x):
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        for block in self.blocks:
            x = block(x)
        x = x.permute(0, 2, 1)
        x = F.gelu(self.fc1(x))
        return self.fc2(x).squeeze(-1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
