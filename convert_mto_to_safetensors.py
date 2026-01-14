#!/usr/bin/env python3
"""
Convert NVIDIA ModelOpt checkpoint to SafeTensors with proper quantization metadata
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import torch
    from safetensors.torch import save_file
    import modelopt.torch.opt as mto
except ImportError as e:
    print(f"Error: Missing required libraries: {e}")
    print("Please install: pip install torch safetensors nvidia-modelopt")
    sys.exit(1)


def convert_mto_to_safetensors(
    mto_checkpoint_path,
    output_safetensors_path,
    quant_format="nvfp4",
    quant_algo="svdquant",
    model_type="flux-dev"
):
    """
    Convert NVIDIA ModelOpt .pt checkpoint to SafeTensors with ComfyUI metadata.

    Args:
        mto_checkpoint_path: Path to ModelOpt .pt checkpoint
        output_safetensors_path: Path for output .safetensors file
        quant_format: Quantization format ('nvfp4' or 'float8_e4m3fn')
        quant_algo: Quantization algorithm used
        model_type: Model type (flux-dev, sd3-medium, etc.)
    """
    mto_path = Path(mto_checkpoint_path)
    if not mto_path.exists():
        raise FileNotFoundError(f"ModelOpt checkpoint not found: {mto_path}")

    output_path = Path(output_safetensors_path)

    print(f"📂 Loading ModelOpt checkpoint: {mto_path}")
    print(f"   Format: {quant_format}, Algorithm: {quant_algo}")

    # Create a dummy backbone model to restore into
    # We need to know the model architecture
    print(f"\n⚠️  NOTE: This requires loading the base model first!")
    print(f"   Model type: {model_type}")

    # Load base model
    from diffusers import FluxTransformer2DModel, StableDiffusion3Transformer2DModel

    if model_type in ["flux-dev", "flux-schnell"]:
        print("📦 Creating FLUX transformer architecture...")
        backbone = FluxTransformer2DModel(
            attention_head_dim=128,
            guidance_embeds=True,
            in_channels=64,
            joint_attention_dim=4096,
            num_attention_heads=24,
            num_layers=19,
            num_single_layers=38,
            patch_size=1,
            pooled_projection_dim=768,
        )
    elif model_type in ["sd3-medium", "sd3.5-medium"]:
        print("📦 Creating SD3 transformer architecture...")
        backbone = StableDiffusion3Transformer2DModel.from_pretrained(
            "stabilityai/stable-diffusion-3-medium",
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    print(f"✅ Base model created")

    # Restore quantized weights using ModelOpt
    print(f"\n🔄 Restoring quantized weights from ModelOpt checkpoint...")
    mto.restore(backbone, str(mto_path))
    print(f"✅ Quantized weights restored")

    # Extract state dict
    print(f"\n📋 Extracting state dict...")
    state_dict = backbone.state_dict()

    # Check for quantized layers
    quant_layers = {}
    weight_scale_count = 0
    input_scale_count = 0

    for key in state_dict.keys():
        if '.weight_scale' in key or '.input_scale' in key:
            if '.weight_scale' in key:
                weight_scale_count += 1
            if '.input_scale' in key:
                input_scale_count += 1

            layer_name = key.rsplit('.', 1)[0]
            if layer_name not in quant_layers:
                quant_layers[layer_name] = {"format": quant_format}

    print(f"✅ Found {len(quant_layers)} quantized layers")
    print(f"   Weight scales: {weight_scale_count}")
    print(f"   Input scales: {input_scale_count}")

    if not quant_layers:
        print("❌ ERROR: No quantized layers found!")
        print("   The checkpoint may not be properly quantized.")
        return False

    # Add ComfyUI metadata
    print(f"\n🔧 Adding ComfyUI-compatible metadata...")
    for layer_name, layer_config in quant_layers.items():
        comfy_quant_key = f"{layer_name}.comfy_quant"
        comfy_quant_value = json.dumps(layer_config).encode('utf-8')
        state_dict[comfy_quant_key] = torch.tensor(list(comfy_quant_value), dtype=torch.uint8)

    # Build metadata
    metadata = {
        "_quantization_metadata": json.dumps({
            "layers": quant_layers,
            "format": quant_format,
            "quant_algo": quant_algo,
            "version": "1.0",
            "model_type": model_type
        })
    }

    # Save as safetensors
    print(f"\n💾 Saving to SafeTensors: {output_path}")
    save_file(state_dict, str(output_path), metadata=metadata)

    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"✅ SUCCESS! Saved to: {output_path}")
        print(f"   File size: {size_mb:.2f} MB")
        print(f"\n📊 Quantization Summary:")
        print(f"   Format: {quant_format}")
        print(f"   Algorithm: {quant_algo}")
        print(f"   Quantized layers: {len(quant_layers)}")
        print(f"   Model type: {model_type}")
        print(f"\n✨ Your model is now ready for ComfyUI!")
        return True
    else:
        print("❌ ERROR: Failed to save SafeTensors file!")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert NVIDIA ModelOpt checkpoint to SafeTensors with ComfyUI metadata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert FLUX model
  python convert_mto_to_safetensors.py model_NVFP4.pt output.safetensors --model-type flux-dev

  # Convert SD3 model with explicit format
  python convert_mto_to_safetensors.py model.pt output.safetensors --model-type sd3-medium --format nvfp4 --algo svdquant
        """
    )

    parser.add_argument("input", type=str, help="Input ModelOpt .pt checkpoint")
    parser.add_argument("output", type=str, help="Output .safetensors file")
    parser.add_argument("--model-type", type=str, required=True,
                        choices=["flux-dev", "flux-schnell", "sd3-medium", "sd3.5-medium"],
                        help="Model architecture type")
    parser.add_argument("--format", type=str, default="nvfp4",
                        choices=["nvfp4", "float8_e4m3fn"],
                        help="Quantization format")
    parser.add_argument("--algo", type=str, default="svdquant",
                        help="Quantization algorithm")

    args = parser.parse_args()

    print("=" * 80)
    print("  CONVERT NVIDIA MODELOPT TO SAFETENSORS")
    print("=" * 80)
    print()

    try:
        success = convert_mto_to_safetensors(
            args.input,
            args.output,
            args.format,
            args.algo,
            args.model_type
        )

        if success:
            print("\n" + "=" * 80)
            sys.exit(0)
        else:
            print("\n" + "=" * 80)
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
