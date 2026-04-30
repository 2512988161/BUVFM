"""VideoVAEPlus pretraining model.

Extracted from VideoVAEPlus (pretrain/VideoVAEPlus/src/) and simplified:
  - Removed PyTorch Lightning dependency
  - Removed text-guided/cross-attention paths (not needed for ultrasound)
  - Removed image-video joint training
  - Kept core: 2+1D ConvNet VAE with temporal compression + discriminator

Architecture:
  Encoder2plus1D → EncoderTemporal1DCNN → latent z
  DecoderTemporal1DCNN → Decoder2plus1D → reconstruction
  Loss: L1 + LPIPS + KL + 3D PatchGAN discriminator

Checkpoint compatibility with buildmodel.py:
  Save format: {"encoder": encoder_state_dict, "classifiers": []}
  The encoder consists of Encoder2plus1D + quant_conv + EncoderTemporal1DCNN.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from functools import partial


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def nonlinearity(x):
    return x * torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


# ---------------------------------------------------------------------------
# Temporal convolution layer (factorized 2+1D)
# ---------------------------------------------------------------------------

class TemporalConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm = Normalize(in_channels)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3),
                              stride=1, padding=(1, 1, 1))
        nn.init.constant_(self.conv.weight, 0)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        h = self.norm(x)
        h = nonlinearity(h)
        h = self.conv(h)
        return h


# ---------------------------------------------------------------------------
# ResNet block (2+1D factorized)
# ---------------------------------------------------------------------------

class ResnetBlock2plus1D(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout=0.0, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 3),
                               stride=1, padding=(0, 1, 1))
        self.conv1_tmp = TemporalConvLayer(out_channels, out_channels)

        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)

        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(1, 3, 3),
                               stride=1, padding=(0, 1, 1))
        self.conv2_tmp = TemporalConvLayer(out_channels, out_channels)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv3d(in_channels, out_channels,
                                               kernel_size=(1, 3, 3), stride=1,
                                               padding=(0, 1, 1))
            else:
                self.nin_shortcut = nn.Conv3d(in_channels, out_channels,
                                              kernel_size=(1, 1, 1), stride=1,
                                              padding=(0, 0, 0))
        self.conv3_tmp = TemporalConvLayer(out_channels, out_channels)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        h = self.conv1_tmp(h) + h

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        h = self.conv2_tmp(h) + h

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
            x = self.conv3_tmp(x) + x

        return x + h


# ---------------------------------------------------------------------------
# 3D Attention block
# ---------------------------------------------------------------------------

class AttnBlock3D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.norm = Normalize(in_channels)
        self.q = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = self.norm(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, t, h, w = q.shape
        q = rearrange(q, "b c t h w -> (b t) (h w) c")
        k = rearrange(k, "b c t h w -> (b t) c (h w)")

        w_ = torch.bmm(q, k)
        w_ = w_ * (int(c) ** (-0.5))
        w_ = F.softmax(w_, dim=2)

        v = rearrange(v, "b c t h w -> (b t) c (h w)")
        w_ = w_.permute(0, 2, 1)
        h_ = torch.bmm(v, w_)
        h_ = rearrange(h_, "(b t) c (h w) -> b c t h w", b=b, h=h)
        h_ = self.proj_out(h_)
        return x + h_


# ---------------------------------------------------------------------------
# Temporal attention (relative position)
# ---------------------------------------------------------------------------

class RelativePosition(nn.Module):
    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(
            torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        device = self.embeddings_table.device
        range_vec_q = torch.arange(length_q, device=device)
        range_vec_k = torch.arange(length_k, device=device)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position,
                                           self.max_relative_position)
        final_mat = (distance_mat_clipped + self.max_relative_position).long()
        return self.embeddings_table[final_mat]


class TemporalAttention(nn.Module):
    def __init__(self, channels, num_heads=1, max_temporal_length=64):
        super().__init__()
        self.num_heads = num_heads
        self.norm = Normalize(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        nn.init.constant_(self.qkv.weight, 0)
        nn.init.constant_(self.qkv.bias, 0)

        head_dim = channels // num_heads
        self.relative_position_k = RelativePosition(num_units=head_dim,
                                                    max_relative_position=max_temporal_length)
        self.relative_position_v = RelativePosition(num_units=head_dim,
                                                    max_relative_position=max_temporal_length)

        self.proj_out = nn.Conv1d(channels, channels, 1)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    def forward(self, x):
        b, c, t, h, w = x.shape
        out = rearrange(x, "b c t h w -> (b h w) c t")
        qkv = self.qkv(self.norm(out))

        bs, width, length = qkv.shape
        assert width % (3 * self.num_heads) == 0
        ch = width // (3 * self.num_heads)

        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))

        weight = torch.einsum("bct,bcs->bts",
                              (q * scale).view(bs * self.num_heads, ch, length),
                              (k * scale).view(bs * self.num_heads, ch, length))

        k_rp = self.relative_position_k(length, length)
        v_rp = self.relative_position_v(length, length)
        weight2 = torch.einsum("bct,tsc->bst",
                               (q * scale).view(bs * self.num_heads, ch, length), k_rp)
        weight = weight + weight2

        weight = F.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight,
                         v.reshape(bs * self.num_heads, ch, length))
        a2 = torch.einsum("bts,tsc->btc", weight, v_rp).transpose(1, 2)
        a = a + a2

        out = a.reshape(bs, -1, length)
        out = self.proj_out(out)
        out = rearrange(out, "(b h w) c t -> b c t h w", b=b, h=h, w=w)
        return x + out


# ---------------------------------------------------------------------------
# Downsample / Upsample (2+1D)
# ---------------------------------------------------------------------------

class Downsample2plus1D(nn.Module):
    def __init__(self, in_channels, with_conv, temp_down):
        super().__init__()
        self.with_conv = with_conv
        self.in_channels = in_channels
        self.temp_down = temp_down
        if self.with_conv:
            self.conv = nn.Conv3d(in_channels, in_channels, kernel_size=(1, 3, 3),
                                  stride=(1, 2, 2), padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1, 0, 0)
            x = F.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        return x


class Upsample2plus1D(nn.Module):
    def __init__(self, in_channels, with_conv, temp_up):
        super().__init__()
        self.with_conv = with_conv
        self.in_channels = in_channels
        self.temp_up = temp_up
        if self.with_conv:
            self.conv = nn.Conv3d(in_channels, in_channels, kernel_size=(1, 3, 3),
                                  stride=1, padding=(0, 1, 1))

    def forward(self, x):
        t = x.shape[2]
        x = rearrange(x, "b c t h w -> b (c t) h w")
        x = F.interpolate(x, scale_factor=(2.0, 2.0), mode="nearest")
        x = rearrange(x, "b (c t) h w -> b c t h w", t=t)
        if self.with_conv:
            x = self.conv(x)
        return x


# ---------------------------------------------------------------------------
# Encoder 2plus1D
# ---------------------------------------------------------------------------

class Encoder2plus1D(nn.Module):
    def __init__(self, *, ch, out_ch, temporal_down_factor, ch_mult=(1, 2, 4, 8),
                 num_res_blocks, attn_resolutions, dropout=0.0, resamp_with_conv=True,
                 in_channels, resolution, z_channels, double_z=True,
                 **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.n_temporal_down = int(math.log2(temporal_down_factor))
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        self.conv_in = nn.Conv3d(in_channels, self.ch, kernel_size=(1, 3, 3),
                                 stride=1, padding=(0, 1, 1))

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock2plus1D(in_channels=block_in, out_channels=block_out,
                                                temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = nn.ModuleList()
            if i_level != self.num_resolutions - 1:
                temp_down = i_level <= self.n_temporal_down - 1
                down.downsample = Downsample2plus1D(block_in, resamp_with_conv, temp_down)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock2plus1D(in_channels=block_in, out_channels=block_in,
                                              temb_channels=self.temb_ch, dropout=dropout)
        self.mid.attn_1 = AttnBlock3D(block_in)
        self.mid.attn_1_tmp = TemporalAttention(block_in, num_heads=1)
        self.mid.block_2 = ResnetBlock2plus1D(in_channels=block_in, out_channels=block_in,
                                              temb_channels=self.temb_ch, dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv3d(block_in, 2 * z_channels if double_z else z_channels,
                                  kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))

    def forward(self, x):
        temb = None
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.attn_1_tmp(h)
        h = self.mid.block_2(h, temb)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


# ---------------------------------------------------------------------------
# Decoder 2plus1D
# ---------------------------------------------------------------------------

class Decoder2plus1D(nn.Module):
    def __init__(self, *, ch, out_ch, temporal_down_factor, ch_mult=(1, 2, 4, 8),
                 num_res_blocks, attn_resolutions, dropout=0.0, resamp_with_conv=True,
                 in_channels, resolution, z_channels, **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.n_temporal_up = int(math.log2(temporal_down_factor))
        self.n_spatial_up = self.num_resolutions - 1
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)

        self.conv_in = nn.Conv3d(z_channels, block_in, kernel_size=(1, 3, 3),
                                 stride=1, padding=(0, 1, 1))

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock2plus1D(in_channels=block_in, out_channels=block_in,
                                              temb_channels=self.temb_ch, dropout=dropout)
        self.mid.attn_1 = AttnBlock3D(block_in)
        self.mid.attn_1_tmp = TemporalAttention(block_in, num_heads=1)
        self.mid.block_2 = ResnetBlock2plus1D(in_channels=block_in, out_channels=block_in,
                                              temb_channels=self.temb_ch, dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock2plus1D(in_channels=block_in, out_channels=block_out,
                                                temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = nn.ModuleList()
            if i_level != 0:
                temp_up = i_level <= self.num_resolutions - 1 - (self.n_spatial_up - self.n_temporal_up)
                up.upsample = Upsample2plus1D(block_in, resamp_with_conv, temp_up)
                curr_res = curr_res * 2
            self.up.insert(0, up)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv3d(block_in, out_ch, kernel_size=(1, 3, 3),
                                  stride=1, padding=(0, 1, 1))

    def forward(self, z):
        temb = None
        h = self.conv_in(z)
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.attn_1_tmp(h)
        h = self.mid.block_2(h, temb)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


# ---------------------------------------------------------------------------
# Temporal 1D-CNN encoder / decoder
# ---------------------------------------------------------------------------

class SamePadConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True,
                 padding_type="replicate"):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]:
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input
        self.padding_type = padding_type
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride,
                              padding=0, bias=bias)

    def forward(self, x):
        return self.conv(F.pad(x, self.pad_input, mode=self.padding_type))


def silu(x):
    return x * torch.sigmoid(x)


class SiLU(nn.Module):
    def forward(self, x):
        return silu(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False,
                 dropout=0.0, norm_type="group", padding_type="replicate"):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        _norm = lambda c: nn.GroupNorm(32, c, eps=1e-6, affine=True) if norm_type == "group" else nn.BatchNorm3d(c)
        self.norm1 = _norm(in_channels)
        self.conv1 = SamePadConv3d(in_channels, out_channels, kernel_size=3, padding_type=padding_type)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = _norm(in_channels)
        self.conv2 = SamePadConv3d(out_channels, out_channels, kernel_size=3, padding_type=padding_type)
        if self.in_channels != self.out_channels:
            self.conv_shortcut = SamePadConv3d(in_channels, out_channels, kernel_size=3, padding_type=padding_type)

    def forward(self, x):
        h = self.norm1(x)
        h = silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = silu(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            x = self.conv_shortcut(x)
        return x + h


class EncoderTemporal1DCNN(nn.Module):
    def __init__(self, *, ch, out_ch, temporal_scale_factor=4, hidden_channel=128, **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temporal_scale_factor = temporal_scale_factor

        self.conv_in = SamePadConv3d(ch, hidden_channel, kernel_size=3, padding_type="replicate")

        self.mid_blocks = nn.ModuleList()
        num_ds = int(math.log2(temporal_scale_factor))
        for i in range(num_ds):
            block = nn.Module()
            in_channels = hidden_channel * 2**i
            out_channels = hidden_channel * 2 ** (i + 1)
            block.down = SamePadConv3d(in_channels, out_channels, kernel_size=3,
                                       stride=(2, 1, 1), padding_type="replicate")
            block.res = ResBlock(out_channels, out_channels)
            block.attn = nn.ModuleList()
            self.mid_blocks.append(block)

        self.final_block = nn.Sequential(
            nn.GroupNorm(32, out_channels, eps=1e-6, affine=True),
            SiLU(),
            SamePadConv3d(out_channels, out_ch * 2, kernel_size=3, padding_type="replicate"),
        )
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, (nn.Linear, nn.Conv3d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def forward(self, x):
        h = self.conv_in(x)
        for block in self.mid_blocks:
            h = block.down(h)
            h = block.res(h)
        h = self.final_block(h)
        return h


class DecoderTemporal1DCNN(nn.Module):
    def __init__(self, *, ch, out_ch, temporal_scale_factor=4, hidden_channel=128, **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temporal_scale_factor = temporal_scale_factor
        num_us = int(math.log2(temporal_scale_factor))

        enc_out_channels = hidden_channel * 2**num_us
        self.conv_in = SamePadConv3d(ch, enc_out_channels, kernel_size=3, padding_type="replicate")

        self.mid_blocks = nn.ModuleList()
        for i in range(num_us):
            block = nn.Module()
            in_channels = enc_out_channels if i == 0 else hidden_channel * 2 ** (num_us - i + 1)
            out_channels = hidden_channel * 2 ** (num_us - i)
            block.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(3, 3, 3),
                                          stride=(2, 1, 1), padding=(1, 1, 1),
                                          output_padding=(1, 0, 0))
            block.res1 = ResBlock(out_channels, out_channels)
            block.attn1 = nn.ModuleList()
            block.res2 = ResBlock(out_channels, out_channels)
            block.attn2 = nn.ModuleList()
            self.mid_blocks.append(block)

        self.conv_last = SamePadConv3d(out_channels, out_ch, kernel_size=3)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, (nn.Linear, nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def forward(self, x):
        h = self.conv_in(x)
        for block in self.mid_blocks:
            h = block.up(h)
            h = block.res1(h)
            h = block.res2(h)
        h = self.conv_last(h)
        return h


# ---------------------------------------------------------------------------
# Gaussian distribution
# ---------------------------------------------------------------------------

class DiagonalGaussianDistribution:
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean)

    def sample(self):
        noise = torch.randn(self.mean.shape, device=self.parameters.device)
        return self.mean + self.std * noise

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        if other is None:
            return 0.5 * torch.sum(torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                                   dim=[1, 2, 3])
        else:
            return 0.5 * torch.sum(
                torch.pow(self.mean - other.mean, 2) / other.var
                + self.var / other.var - 1.0 - self.logvar + other.logvar,
                dim=[1, 2, 3])

    def mode(self):
        return self.mean


# ---------------------------------------------------------------------------
# 3D PatchGAN discriminator
# ---------------------------------------------------------------------------

class NLayerDiscriminator3D(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False):
        super().__init__()
        norm_layer = nn.BatchNorm3d
        use_bias = norm_layer != nn.BatchNorm3d

        kw = 3
        padw = 1
        sequence = [
            nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=(kw, kw, kw),
                          stride=(2 if n == 1 else 1, 2, 2), padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=(kw, kw, kw),
                      stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]
        sequence += [nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.main = nn.Sequential(*sequence)

    def forward(self, x):
        return self.main(x)


# ---------------------------------------------------------------------------
# Discriminator losses
# ---------------------------------------------------------------------------

def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    return 0.5 * (loss_real + loss_fake)


def vanilla_d_loss(logits_real, logits_fake):
    return (F.softplus(-logits_real).mean() + F.softplus(logits_fake).mean()) * 0.5


def adopt_weight(weight, global_step, threshold=0, value=0.0):
    if global_step < threshold:
        weight = value
    return weight


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# ---------------------------------------------------------------------------
# LPIPS + Discriminator loss (3D version)
# ---------------------------------------------------------------------------

class LPIPSWithDiscriminator3D(nn.Module):
    """Reconstruction loss: L1 + LPIPS + KL + 3D PatchGAN discriminator."""

    def __init__(self, disc_start=50001, kl_weight=1e-6, perceptual_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0,
                 disc_weight=0.5, disc_loss="hinge", logvar_init=0.0):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.perceptual_weight = perceptual_weight
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)
        self.discriminator = NLayerDiscriminator3D(
            input_nc=disc_in_channels, n_layers=disc_num_layers).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight

        if self.perceptual_weight > 0:
            try:
                from taming.modules.losses.vqperceptual import LPIPS as LPIPSModel
            except ImportError:
                import lpips
                LPIPSModel = lpips.LPIPS
            self.perceptual_loss = LPIPSModel().eval()
            for p in self.perceptual_loss.parameters():
                p.requires_grad = False
        else:
            self.perceptual_loss = None

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer):
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight * self.discriminator_weight

    def forward(self, inputs, reconstructions, posteriors, optimizer_idx, global_step,
                last_layer=None, split="train"):
        t = inputs.shape[2]
        inputs_2d = rearrange(inputs, "b c t h w -> (b t) c h w")
        reconstructions_2d = rearrange(reconstructions, "b c t h w -> (b t) c h w")

        # L1 loss
        rec_loss = torch.abs(inputs_2d.contiguous() - reconstructions_2d.contiguous())

        # LPIPS
        if self.perceptual_weight > 0 and self.perceptual_loss is not None:
            p_loss = self.perceptual_loss(inputs_2d.contiguous(), reconstructions_2d.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]

        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        # Before discriminator start: only reconstruction + KL
        if global_step < self.discriminator_iter_start:
            loss = nll_loss + self.kl_weight * kl_loss
            log = {
                f"{split}/total_loss": loss.detach(),
                f"{split}/logvar": self.logvar.detach(),
                f"{split}/kl_loss": kl_loss.detach(),
                f"{split}/nll_loss": nll_loss.detach(),
                f"{split}/rec_loss": rec_loss.detach().mean(),
            }
            return loss, log

        # GAN phase
        if optimizer_idx == 0:
            # Generator update
            logits_fake = self.discriminator(reconstructions.contiguous())
            g_loss = -torch.mean(logits_fake)
            d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer)
            disc_factor = adopt_weight(self.disc_factor, global_step,
                                       threshold=self.discriminator_iter_start)
            loss = nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss
            log = {
                f"{split}/total_loss": loss.detach(),
                f"{split}/logvar": self.logvar.detach(),
                f"{split}/kl_loss": kl_loss.detach(),
                f"{split}/nll_loss": nll_loss.detach(),
                f"{split}/rec_loss": rec_loss.detach().mean(),
                f"{split}/d_weight": d_weight.detach(),
                f"{split}/disc_factor": torch.tensor(disc_factor),
                f"{split}/g_loss": g_loss.detach(),
            }
            return loss, log

        if optimizer_idx == 1:
            # Discriminator update
            logits_real = self.discriminator(inputs.contiguous().detach())
            logits_fake = self.discriminator(reconstructions.contiguous().detach())
            disc_factor = adopt_weight(self.disc_factor, global_step,
                                       threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)
            log = {
                f"{split}/disc_loss": d_loss.detach(),
                f"{split}/logits_real": logits_real.detach().mean(),
                f"{split}/logits_fake": logits_fake.detach().mean(),
            }
            return d_loss, log


# ---------------------------------------------------------------------------
# Full VAE pretrain model
# ---------------------------------------------------------------------------

class VideoVAEPretrainModel(nn.Module):
    """VideoVAEPlus pretraining model.

    Encoder 2+1D → Temporal compression → Latent z
    Latent z → Temporal decompression → Decoder 2+1D → Reconstruction
    """

    def __init__(
        self,
        ddconfig,
        ppconfig,
        lossconfig,
        embed_dim=4,
        use_quant_conv=True,
        input_dim=5,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_quant_conv = use_quant_conv

        # Spatial-temporal encoder/decoder
        self.encoder = Encoder2plus1D(**ddconfig)
        self.decoder = Decoder2plus1D(**ddconfig)

        # Temporal compression
        self.encoder_temporal = EncoderTemporal1DCNN(**ppconfig)
        self.decoder_temporal = DecoderTemporal1DCNN(**ppconfig)

        # Quantization
        if use_quant_conv:
            self.quant_conv = nn.Conv3d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
            self.post_quant_conv = nn.Conv3d(embed_dim, ddconfig["z_channels"], 1)

        # Loss
        self.loss = LPIPSWithDiscriminator3D(**lossconfig)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @property
    def device(self):
        return next(self.parameters()).device

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def encode(self, x):
        h = self.encoder(x)
        if self.use_quant_conv:
            h = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(h)
        z = posterior.sample()
        posterior_t = self.encoder_temporal(z)
        posterior_t = DiagonalGaussianDistribution(posterior_t)
        z_t = posterior_t.sample()
        return z_t, posterior_t

    def decode(self, z):
        if hasattr(self, 'decoder_temporal'):
            z = self.decoder_temporal(z)
        if self.use_quant_conv:
            z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, x):
        z, posterior = self.encode(x)
        dec = self.decode(z)
        return dec, posterior

    def get_encoder_state_dict(self):
        """Return encoder weights for buildmodel.py compatibility."""
        encoder_state = {}
        for k, v in self.state_dict().items():
            if k.startswith("encoder.") or k.startswith("quant_conv.") or k.startswith("encoder_temporal."):
                encoder_state[k] = v
        return encoder_state


def get_vae_config(embed_dim=4, resolution=224, z_channels=4, temporal_scale_factor=4):
    """Return default VAE config matching config_4z.yaml."""
    ddconfig = {
        "double_z": True,
        "z_channels": z_channels,
        "resolution": resolution,
        "in_channels": 3,
        "out_ch": 3,
        "ch": 128,
        "ch_mult": [1, 2, 4, 4],
        "temporal_down_factor": 1,
        "num_res_blocks": 2,
        "attn_resolutions": [],
        "dropout": 0.0,
    }
    ppconfig = {
        "temporal_scale_factor": temporal_scale_factor,
        "z_channels": embed_dim,
        "out_ch": embed_dim,
        "ch": embed_dim,
        "attn_temporal_factor": [],
    }
    lossconfig = {
        "disc_start": 50001,
        "kl_weight": 0.000001,
        "disc_weight": 0.5,
    }
    return ddconfig, ppconfig, lossconfig


