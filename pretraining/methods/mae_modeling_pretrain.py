"""VideoMAEv2 pretraining model with VJEPA ViT-g encoder.

Reuses VideoMAEv2's asymmetric encoder-decoder architecture but replaces
the encoder with VJEPA's VisionTransformer (vit_giant_xformers: RoPE + SwiGLU)
and the decoder with VJEPA's Block (standard attention, no RoPE).

Checkpoint compatibility with buildmodel.py:
  State dict keys are like: encoder.vit.patch_embed.proj.weight
  convert_checkpoint.py strips the "encoder.vit." prefix to produce
  {"encoder": state_dict, "classifiers": []} for build_model().
"""

from functools import partial

import numpy as np
import torch
import torch.nn as nn

import src.models.vision_transformer as vit
from src.models.utils.modules import Block
from src.models.utils.pos_embs import get_1d_sincos_pos_embed_from_grid
from src.masks.utils import apply_masks


class VJEPAEncoder(nn.Module):
    """Wraps VJEPA's VisionTransformer (vit_giant_xformers) to accept
    VideoMAEv2-style boolean masks.

    Forward: (x, mask) -> x_vis
        x:    [B, C, T, H, W] video tensor
        mask: [B, N] boolean, True = masked (removed), False = visible (kept)
    Returns: [B, num_visible, embed_dim]
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        num_frames=16,
        tubelet_size=2,
        in_chans=3,
        embed_dim=1408,
        depth=40,
        num_heads=22,
        mlp_ratio=48 / 11,
        qkv_bias=True,
        drop_path_rate=0.0,
        use_rope=True,
        uniform_power=True,
        with_cp=False,
        **kwargs,
    ):
        super().__init__()
        self.vit = vit.VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_path_rate=drop_path_rate,
            use_rope=use_rope,
            uniform_power=uniform_power,
            out_layers=None,
            use_activation_checkpointing=with_cp,
            **kwargs,
        )
        self.embed_dim = embed_dim
        self.num_features = embed_dim
        self.num_classes = 0
        self.num_patches = self.vit.num_patches  # e.g. 1568 for 16fr@224
        self.with_cp = with_cp

        # Store patch_size as tuple for engine compatibility
        self._patch_size = patch_size

    @property
    def patch_embed(self):
        """Compatibility: engine accesses model.encoder.patch_embed.patch_size[0]

        Returns self so that .patch_size works as a tuple-like attribute.
        """
        return self

    @property
    def patch_size(self):
        return (self._patch_size, self._patch_size)

    def forward(self, x, mask):
        """
        Args:
            x: [B, C, T, H, W]
            mask: [B, N] boolean, True=masked (remove), False=visible (keep)
        Returns:
            [B, num_visible, embed_dim]
        """
        B = x.shape[0]
        mask_flat = mask.flatten(1).to(torch.bool)  # [B, N]

        ids_keep_list = []
        for b in range(B):
            ids = torch.where(~mask_flat[b])[0]
            ids_keep_list.append(ids)
        ids_keep = torch.stack(ids_keep_list, dim=0)  # [B, num_vis]

        x_out = self.vit(x, masks=[ids_keep])  # [B, num_vis, embed_dim]
        return x_out

    def get_num_layers(self):
        return len(self.vit.blocks)

    def no_weight_decay(self):
        return set()


class MAEDecoder(nn.Module):
    """Shallow Transformer decoder for MAE pretraining.

    Uses VJEPA's Block with use_rope=False, act_layer=nn.GELU,
    which is functionally identical to VideoMAEv2's Block.
    """

    def __init__(
        self,
        patch_size=16,
        embed_dim=512,
        depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        tubelet_size=2,
        decoder_num_classes=1536,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        with_cp=False,
    ):
        super().__init__()
        self.decoder_num_classes = decoder_num_classes
        self.embed_dim = embed_dim
        self.with_cp = with_cp

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=0.0,
                attn_drop=0.0,
                drop_path=dpr[i],
                act_layer=nn.GELU,
                use_rope=False,
                norm_layer=norm_layer,
                use_sdpa=True,
            )
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, decoder_num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, return_token_num):
        for blk in self.blocks:
            if self.with_cp:
                x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        if return_token_num > 0:
            x = self.head(self.norm(x[:, -return_token_num:]))
        else:
            x = self.head(self.norm(x))
        return x


class MAEPretrainModel(nn.Module):
    """VJEPA ViT-g encoder + shallow transformer decoder for MAE pretraining.

    Architecture matches VideoMAEv2's PretrainVisionTransformerVJEPA exactly.
    Encoder processes only visible patches. Decoder reconstructs masked patches.
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        num_frames=16,
        tubelet_size=2,
        in_chans=3,
        encoder_embed_dim=1408,
        encoder_depth=40,
        encoder_num_heads=22,
        encoder_mlp_ratio=48 / 11,
        decoder_embed_dim=512,
        decoder_depth=4,
        decoder_num_heads=8,
        decoder_num_classes=None,
        norm_layer=None,
        drop_path_rate=0.0,
        use_rope=True,
        uniform_power=True,
        with_cp=False,
        **kwargs,
    ):
        super().__init__()

        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        if decoder_num_classes is None:
            decoder_num_classes = 3 * tubelet_size * patch_size * patch_size  # 1536

        # ---- Encoder (VJEPA ViT-g) ----
        self.encoder = VJEPAEncoder(
            img_size=img_size,
            patch_size=patch_size,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            in_chans=in_chans,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=encoder_mlp_ratio,
            qkv_bias=True,
            drop_path_rate=drop_path_rate,
            use_rope=use_rope,
            uniform_power=uniform_power,
            with_cp=with_cp,
        )

        # ---- Decoder (shallow Transformer) ----
        self.decoder = MAEDecoder(
            patch_size=patch_size,
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=4.0,
            tubelet_size=tubelet_size,
            decoder_num_classes=decoder_num_classes,
            drop_path_rate=0.0,
            with_cp=False,
        )

        # ---- Encoder-to-decoder projection ----
        self.encoder_to_decoder = nn.Linear(
            encoder_embed_dim, decoder_embed_dim, bias=False
        )

        # ---- Learnable mask token ----
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # ---- Decoder positional embeddings (sinusoidal) ----
        num_patches = self.encoder.num_patches
        pos_embed_np = get_1d_sincos_pos_embed_from_grid(
            decoder_embed_dim, np.arange(num_patches, dtype=float)
        )
        self.register_buffer(
            "pos_embed",
            torch.from_numpy(pos_embed_np).float().unsqueeze(0),
            persistent=True,
        )

        # Initialize mask token
        from timm.models.layers import trunc_normal_
        trunc_normal_(self.mask_token, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def no_weight_decay(self):
        return {"mask_token", "pos_embed"}

    def forward(self, x, bool_masked_pos, decode_masked_pos=None):
        """
        Args:
            x:                  [B, C, T, H, W] video tensor
            bool_masked_pos:    [B, N] boolean, True=masked (encoder ignores)
            decode_masked_pos:  [B, N] boolean, True=visible to decoder
                                (None = decoder sees all masked positions)
        Returns:
            x_rec: [B, N_mask_dec, decoder_num_classes] reconstructed pixels
        """
        decode_vis = bool_masked_pos if decode_masked_pos is None else ~decode_masked_pos

        # 1. Encode visible patches only
        x_vis = self.encoder(x, bool_masked_pos)  # [B, N_vis, 1408]
        x_vis = self.encoder_to_decoder(x_vis)    # [B, N_vis, 512]

        B, N_vis, C = x_vis.shape

        # 2. Partition positional embeddings into visible and masked portions
        expand_pos_embed = self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device)

        mask_flat = bool_masked_pos.flatten(1).to(torch.bool)
        decode_flat = decode_vis.flatten(1).to(torch.bool)

        pos_emd_vis = expand_pos_embed[~mask_flat].reshape(B, -1, C)
        pos_emd_mask = expand_pos_embed[decode_flat].reshape(B, -1, C)

        # 3. Concatenate visible encoded + mask tokens
        x_full = torch.cat([
            x_vis + pos_emd_vis,
            self.mask_token + pos_emd_mask,
        ], dim=1)

        # 4. Decode, returning only the mask token predictions
        x_rec = self.decoder(x_full, pos_emd_mask.shape[1])
        return x_rec
