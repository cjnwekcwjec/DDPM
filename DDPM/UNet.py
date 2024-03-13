import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.n_channels = n_channels
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        self.act = Swish()
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t: torch.Tensor):
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)

        return emb


class ResidualBlock(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, time_channel: int,
                 n_groups: int = 32, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(n_groups, in_channel)
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, 1, 1)
        self.norm2 = nn.GroupNorm(n_groups, out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        self.act = Swish()
        self.dropout = nn.Dropout(dropout)
        self.time_emb = nn.Linear(time_channel, out_channel)
        if in_channel != out_channel:
            self.shortcut = nn.Conv2d(in_channel, out_channel, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        h = self.conv1(self.act(self.norm1(x)))
        t = self.time_emb(self.act(t))[:, :, None, None]
        h = h + t
        h = self.conv2(self.dropout(self.act(self.norm2(h))))
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    def __init__(self, n_channel: int, n_heads: int = 1, d_k: int = None):
        super().__init__()
        if d_k is None:
            d_k = n_channel
        self.d_k = d_k
        self.n_heads = n_heads
        self.project = nn.Linear(n_channel, n_heads * d_k * 3)
        self.output = nn.Linear(n_heads * d_k, n_channel)
        self.scale = d_k ** 0.5

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None):
        _ = t
        batch_size, n_channel, height, width = x.shape
        # x -> [batch_size, height * width, n_channel]
        x = x.view(batch_size, n_channel, -1).permute(0, 2, 1)
        # qkv -> [batch_size, height * width, n_heads, d_k*3]
        qkv = self.project(x).view(batch_size, -1, self.n_heads, self.d_k * 3)
        # q,k,v -> [batch_size, height * width, n_heads, d_k]
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # q,k,v -> [batch_size, n_heads, height * width, d_k]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # k -> [batch_size, n_heads, d_k, height * width]
        k = k.transpose(2, 3)
        # attn -> [batch_size, n_heads, height * width, height * width]
        attn = q @ k / self.scale
        attn = F.softmax(attn, dim=-1)
        # out -> [batch_size, n_heads, height * width, d_k]
        out = attn @ v
        # out -> [batch_size, height * width, n_heads * d_k]
        out = out.transpose(1, 2).view(batch_size, -1, self.n_heads * self.d_k)
        # out -> [batch_size, height * width, n_channel]
        out = self.output(out)
        out = out + x
        out = out.transpose(1, 2).view(batch_size, n_channel, height, width)
        return out


class DownBlock(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, time_channel: int, has_attn: bool):
        super().__init__()
        self.res = ResidualBlock(in_channel, out_channel, time_channel)
        if has_attn:
            self.attn = AttentionBlock(out_channel)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, time_channel: int, has_attn: bool):
        super().__init__()
        self.res = ResidualBlock(in_channel + out_channel, out_channel, time_channel)
        if has_attn:
            self.attn = AttentionBlock(out_channel)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)
        return x


class MiddleBlock(nn.Module):
    def __init__(self, n_channel: int, time_channel: int):
        super().__init__()
        self.res1 = ResidualBlock(n_channel, n_channel, time_channel)
        self.attn = AttentionBlock(n_channel)
        self.res2 = ResidualBlock(n_channel, n_channel, time_channel)

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None):
        x = self.res1(x, t)
        x = self.attn(x)
        x = self.res2(x, t)
        return x


class DownSample(nn.Module):
    def __init__(self, n_channel: int):
        super().__init__()
        self.conv = nn.Conv2d(n_channel, n_channel, 3, 2, 1)

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None):
        _ = t
        x = self.conv(x)
        return x


class UpSample(nn.Module):
    def __init__(self, n_channel: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(n_channel, n_channel, 4, 2, 1)

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None):
        _ = t
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, img_channel: int = 3, n_channel: int = 64, ch_mult: List[int] = [1, 2, 2, 4],
                 has_attn: List[bool] = [False, False, True, True], n_block: int = 2):
        super().__init__()
        self.img_pro = nn.Conv2d(img_channel, n_channel, 3, 1, 1)
        self.time_emb = TimeEmbedding(n_channel * 4)
        self.n_block = n_block
        num_layers = len(ch_mult)

        down = []
        in_channel = out_channel = n_channel
        for i in range(num_layers):
            out_channel = in_channel * ch_mult[i]
            for _ in range(n_block):
                down.append(DownBlock(in_channel, out_channel, n_channel * 4, has_attn[i]))
                in_channel = out_channel
            if i < num_layers - 1:
                down.append(DownSample(in_channel))
        self.down = nn.ModuleList(down)

        self.middle = MiddleBlock(out_channel, n_channel * 4)

        up = []
        in_channel = out_channel
        for i in reversed(range(num_layers)):
            out_channel = in_channel
            for _ in range(n_block):
                up.append(UpBlock(in_channel, out_channel, n_channel * 4, has_attn[i]))
            out_channel = in_channel // ch_mult[i]
            up.append(UpBlock(in_channel, out_channel, n_channel * 4, has_attn[i]))
            in_channel = out_channel
            if i > 0:
                up.append(UpSample(in_channel))
        self.up = nn.ModuleList(up)

        self.final = nn.Conv2d(n_channel, img_channel, 3, 1, 1)
        self.act = Swish()
        self.norm = nn.GroupNorm(8, n_channel)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.img_pro(x)
        t = self.time_emb(t)
        h = [x]
        for m in self.down:
            x = m(x, t)
            h.append(x)
        x = self.middle(x, t)
        for m in self.up:
            if isinstance(m, UpSample):
                x = m(x, t)
            else:
                s = h.pop()
                x = torch.cat((s, x), dim=1)
                x = m(x, t)
        x = self.final(self.act(self.norm(x)))
        return x


if __name__ == '__main__':
    x = torch.ones((2, 3, 32, 32))
    t = torch.randint(0, 1000, (2,))
    model = UNet()
    y = model(x, t)
    print(y.shape)
