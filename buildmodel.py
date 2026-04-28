import torch
import torch.nn as nn
import src.models.vision_transformer as vit
from src.masks.utils import apply_masks
from src.models.utils.pos_embs import get_1d_sincos_pos_embed
from src.models.attentive_pooler import AttentiveClassifier
import logging

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class ClipAggregation(nn.Module):
    """
    Process each clip independently and concatenate all tokens
    """

    def __init__(
        self,
        model,
        tubelet_size=2,
        max_frames=128,
        use_pos_embed=False,
        out_layers=None,
    ):
        super().__init__()
        self.model = model
        self.tubelet_size = tubelet_size
        self.embed_dim = embed_dim = model.embed_dim
        self.num_heads = model.num_heads

        # 1D-temporal pos-embedding
        self.pos_embed = None
        if use_pos_embed:
            max_T = max_frames // tubelet_size
            self.pos_embed = nn.Parameter(torch.zeros(1, max_T, embed_dim), requires_grad=False)
            sincos = get_1d_sincos_pos_embed(embed_dim, max_T)
            self.pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def forward(self, x, clip_indices=None):

        num_clips = len(x)
        num_views_per_clip = len(x[0])
        B, C, F, H, W = x[0][0].size()

        # Concatenate all spatial and temporal views along batch dimension
        x = [torch.cat(xi, dim=0) for xi in x]
        x = torch.cat(x, dim=0)

        outputs = self.model(x)
        outputs = torch.cat(outputs, dim=1)

        def multiviews_postprocess(outputs):
            _, N, D = outputs.size()
            T = F // self.tubelet_size  # num temporal indices
            S = N // T  # num spatial tokens

            # Unroll outputs into a 2D array [spatial_views x temporal_views]
            eff_B = B * num_views_per_clip
            all_outputs = [[] for _ in range(num_views_per_clip)]
            for i in range(num_clips):
                o = outputs[i * eff_B : (i + 1) * eff_B]
                for j in range(num_views_per_clip):
                    all_outputs[j].append(o[j * B : (j + 1) * B])

            for i, outputs in enumerate(all_outputs):
                # Concatenate along temporal dimension
                outputs = [o.reshape(B, T, S, D) for o in outputs]
                outputs = torch.cat(outputs, dim=1).flatten(1, 2)
                # Compute positional embedding
                if (self.pos_embed is not None) and (clip_indices is not None):
                    _indices = [c[:, :: self.tubelet_size] for c in clip_indices]
                    pos_embed = self.pos_embed.repeat(B, 1, 1)  # [B, max_T, D]
                    pos_embed = apply_masks(pos_embed, _indices, concat=False)  # list(Tensor([B, T, D]))
                    pos_embed = torch.cat(pos_embed, dim=1)  # concatenate along temporal dimension
                    pos_embed = pos_embed.unsqueeze(2).repeat(1, 1, S, 1)  # [B, T*num_clips, S, D]
                    pos_embed = pos_embed.flatten(1, 2)
                    outputs += pos_embed
                all_outputs[i] = outputs

            return all_outputs

        return multiviews_postprocess(outputs)
        
def build_model(checkpoint_path, resolution=224, frames_per_clip=16, num_classes=3, num_heads=16, num_probe_blocks=1):
    """
    Build and load the VJEPA2 model with classifier
    
    Args:
        checkpoint_path: Path to the checkpoint file
        resolution: Input image resolution
        frames_per_clip: Number of frames per clip
        num_classes: Number of classification classes
        num_heads: Number of attention heads in classifier
        num_probe_blocks: Number of probe blocks in classifier
    
    Returns:
        model: Loaded VJEPA2 model
        classifier: Loaded AttentiveClassifier
    """
    logger.info(f"Loading pretrained model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Model configuration based on ultrasound_videos_full_fuse.yaml
    enc_kwargs = {
        "checkpoint_key": "encoder",
        "img_temporal_dim_size": None,
        "model_name": "vit_giant_xformers",
        "patch_size": 16,
        "tubelet_size": 2,
        "uniform_power": True,
        "use_rope": True
    }

    wrapper_kwargs = {
        "max_frames": 16,
        "use_pos_embed": False,
        "out_layers": [39]
    }

    out_layers = wrapper_kwargs.get("out_layers")

    # Initialize the encoder model
    model = vit.__dict__[enc_kwargs["model_name"]](
        img_size=resolution, 
        num_frames=frames_per_clip, 
        out_layers=out_layers, 
        **enc_kwargs
    )

    # Load pretrained weights
    pretrained_dict = checkpoint[enc_kwargs["checkpoint_key"]]
    
    # Remove module., backbone., and model. prefixes if present
    for prefix in ["module.", "backbone.", "model."]:
        pretrained_dict = {
            (k[len(prefix):] if k.startswith(prefix) else k): v 
            for k, v in pretrained_dict.items()
        }
    
    # Check for mismatched keys
    for k, v in model.state_dict().items():
        if k not in pretrained_dict:
            logger.info(f'key "{k}" could not be found in loaded state dict')
        elif pretrained_dict[k].shape != v.shape:
            logger.info(f'key "{k}" is of different shape in model and loaded state dict')
            pretrained_dict[k] = v
    
    # Load state dict with strict=False to ignore missing keys
    msg = model.load_state_dict(pretrained_dict, strict=False)
    logger.info(f"loaded pretrained model with msg: {msg}")

    # Wrap with ClipAggregation
    model = ClipAggregation(
        model,
        tubelet_size=model.tubelet_size,
        **wrapper_kwargs,
    )
    
    # Create classifier
    classifier = AttentiveClassifier(
        embed_dim=model.embed_dim,
        num_heads=num_heads,
        depth=num_probe_blocks,
        num_classes=num_classes,
        use_activation_checkpointing=True,
    )
    
    # Load classifier weights if present in checkpoint
    if "classifiers" in checkpoint and checkpoint["classifiers"]:
        classifier_dict = checkpoint["classifiers"][0] if isinstance(checkpoint["classifiers"], list) else checkpoint["classifiers"]
        # Remove module. prefix if present
        classifier_dict = {
            (k[7:] if k.startswith("module.") else k): v 
            for k, v in classifier_dict.items()
        }
        msg = classifier.load_state_dict(classifier_dict, strict=False)
        logger.info(f"loaded classifier weights with msg: {msg}")
    
    return model, classifier

def test_model():
    """
    Test the model with dummy input
    """
    # Path to the checkpoint
    checkpoint_path = "/home/lx/alg/baselines/vjepa/checkpoint_epoch_20.pt"
    
    # Build the model and classifier
    model, classifier = build_model(checkpoint_path)
    model.eval()
    classifier.eval()
    
    # Create dummy input
    # Input shape: [num_clips, num_views_per_clip, batch_size, channels, frames, height, width]
    num_clips = 1
    num_views_per_clip = 1
    batch_size = 1
    channels = 3
    frames = 16
    height = 224
    width = 224
    
    dummy_input = [
        [torch.randn(batch_size, channels, frames, height, width) for _ in range(num_views_per_clip)]
        for _ in range(num_clips)
    ]
    
    # Forward pass through model
    with torch.no_grad():
        outputs = model(dummy_input)
        # Forward pass through classifier
        classifier_outputs = [classifier(o) for o in outputs]
    
    # Print output shapes
    print(f"Model output shape: {outputs[0].shape}")
    print(f"Model embedding dimension: {model.embed_dim}")
    print(f"Classifier output shape: {classifier_outputs[0].shape}")
    print("Test completed successfully!")

if __name__ == "__main__":
    test_model()