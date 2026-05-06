#!/usr/bin/env python3
"""Convert pretraining checkpoints to buildmodel.py-compatible format.

Input formats:
  - VideoMAEv2: checkpoint-{epoch}.pth with key "model"
  - VAE: checkpoint-{epoch}.pth from VideoVAEPlus pretraining

Output format: {"encoder": state_dict, "classifiers": []}
  The encoder state_dict uses VJEPA VisionTransformer keys:
      patch_embed.proj.weight, blocks.0.norm1.weight, norm.weight, ...

Usage:
  python pretraining/convert_checkpoint.py \
      --method videomae \
      --input pretraining/output/videomae_v2_vitg/checkpoint-299.pth \
      --output pretraining/output/videomae_v2_vitg/encoder_checkpoint.pt

  python pretraining/convert_checkpoint.py --method vae  --input pretraining/output/vae_4z/checkpoint-2.pth --output ckpts/vitg_vae_e1.pt
"""

import argparse
import os

import torch


def strip_prefix(state_dict, prefix):
    """Strip a prefix from all keys in state_dict."""
    new_dict = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_dict[k[len(prefix):]] = v
    return new_dict


def convert_videomae(input_path, output_path):
    """Convert VideoMAEv2 PretrainVisionTransformerVJEPA checkpoint.

    Extracts encoder weights (VJEPA ViT) from the full pretraining model.
    Checkpoint structure: {"model": full_state_dict, "optimizer": ..., "epoch": ...}
    """
    ckpt = torch.load(input_path, map_location="cpu", weights_only=False)
    if "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    # Remove DDP "module." prefix if present
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = strip_prefix(state_dict, "module.")

    # Extract encoder.vit.* weights → map to VJEPA ViT keys
    encoder_dict = {}
    for k, v in state_dict.items():
        if k.startswith("encoder.vit."):
            new_key = k[len("encoder.vit."):]
            encoder_dict[new_key] = v

    if len(encoder_dict) == 0:
        raise ValueError(
            "No encoder.vit.* keys found. Is this a VideoMAEv2 VJEPA checkpoint?"
        )

    print(f"[VideoMAEv2] Extracted {len(encoder_dict)} encoder keys")
    for k in list(encoder_dict.keys())[:5]:
        print(f"  {k}: {encoder_dict[k].shape}")
    print("  ...")

    output = {"encoder": encoder_dict, "classifiers": []}
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(output, output_path)
    print(f"Saved to {output_path}")


def convert_vae(input_path, output_path):
    """Convert VAE pretraining checkpoint with ViT encoder.

    Extracts VisionTransformer encoder weights from the full VAE model.
    The ViT encoder is stored under "encoder.*" keys in the state dict.
    Strips the "encoder." prefix to produce bare VisionTransformer keys
    compatible with build_model().

    Input formats (both supported):
      - Raw training checkpoint: {"model": full_state_dict, "optimizer": ..., ...}
        Keys like: encoder.patch_embed.proj.weight, encoder.blocks.0.norm1.weight, ...
      - Pre-saved encoder checkpoint: {"encoder": state_dict, "classifiers": [...]}
    """
    ckpt = torch.load(input_path, map_location="cpu", weights_only=False)

    # Case 1: Already in build_model format {"encoder": ..., "classifiers": ...}
    if "encoder" in ckpt and "model" not in ckpt:
        # Check if encoder keys have "encoder." prefix (raw training format)
        encoder_dict = ckpt["encoder"]
        sample_keys = list(encoder_dict.keys())[:3]
        if sample_keys and sample_keys[0].startswith("encoder."):
            # Strip "encoder." prefix
            encoder_dict = {
                k[len("encoder."):]: v
                for k, v in encoder_dict.items()
                if k.startswith("encoder.")
            }
        output = {"encoder": encoder_dict, "classifiers": ckpt.get("classifiers", [])}
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        torch.save(output, output_path)
        print(f"[VAE] Already in build_model format, saved to {output_path}")
        return

    # Case 2: Raw training checkpoint {"model": state_dict, ...}
    if "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    # Remove DDP "module." prefix if present
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = strip_prefix(state_dict, "module.")

    # Extract ViT encoder weights (encoder.*) and strip prefix
    encoder_dict = {}
    for k, v in state_dict.items():
        if k.startswith("encoder."):
            encoder_dict[k[len("encoder."):]] = v

    if len(encoder_dict) == 0:
        raise ValueError(
            "No encoder.* keys found. Is this a VAE checkpoint with ViT encoder?"
        )

    print(f"[VAE] Extracted {len(encoder_dict)} ViT encoder keys")
    for k in list(encoder_dict.keys())[:5]:
        print(f"  {k}: {encoder_dict[k].shape}")
    print("  ...")

    output = {"encoder": encoder_dict, "classifiers": []}
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(output, output_path)
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert pretraining checkpoint")
    parser.add_argument("--method", required=True, choices=["videomae", "vae"],
                        help="Pretraining method")
    parser.add_argument("--input", required=True, help="Input checkpoint path")
    parser.add_argument("--output", required=True, help="Output checkpoint path")
    args = parser.parse_args()

    if args.method == "videomae":
        convert_videomae(args.input, args.output)
    elif args.method == "vae":
        convert_vae(args.input, args.output)


if __name__ == "__main__":
    main()
