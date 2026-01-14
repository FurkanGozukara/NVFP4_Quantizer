#!/usr/bin/env python3
"""
Fix Quantization Metadata for ComfyUI Compatibility
This script adds proper quantization metadata to existing NVFP4/FP8 models
so they can be correctly loaded by ComfyUI.
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import torch
    from safetensors.torch import load_file, save_file
except ImportError as e:
    print(f"Error: Missing required libraries: {e}")
    print("Please install: pip install torch safetensors")
    sys.exit(1)


def detect_quantization_format(state_dict):
    """
    Auto-detect the quantization format from the state dict.

    Returns:
        str: 'nvfp4', 'float8_e4m3fn', or 'unknown'
    """
    # Check for NVFP4 indicators (weight_scale_2 is unique to NVFP4)
    for key in state_dict.keys():
        if '.weight_scale_2' in key:
            return 'nvfp4'

    # Check for FP8 by examining weight dtypes
    for key, tensor in state_dict.items():
        if 'weight' in key and not key.endswith('_scale'):
            if tensor.dtype == torch.float8_e4m3fn:
                return 'float8_e4m3fn'
            elif tensor.dtype == torch.uint8 and '.weight_scale' in str(list(state_dict.keys())):
                # uint8 storage with weight_scale suggests NVFP4
                return 'nvfp4'

    return 'unknown'


def fix_quantization_metadata(input_path, output_path=None, quant_format=None, quant_algo="svdquant"):
    """
    Add ComfyUI-compatible quantization metadata to a safetensors file.

    Args:
        input_path: Path to input .safetensors file
        output_path: Path to output .safetensors file (defaults to input_path)
        quant_format: Quantization format ('nvfp4' or 'float8_e4m3fn', auto-detect if None)
        quant_algo: Quantization algorithm used (default: 'svdquant')
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if output_path is None:
        output_path = input_path
    else:
        output_path = Path(output_path)

    print(f"📂 Loading model from: {input_path}")
    state_dict = load_file(str(input_path))

    # Auto-detect format if not specified
    if quant_format is None:
        quant_format = detect_quantization_format(state_dict)
        print(f"🔍 Auto-detected quantization format: {quant_format}")

        if quant_format == 'unknown':
            print("⚠️  WARNING: Could not auto-detect quantization format!")
            print("   Please specify format manually with --format")
            return False
    else:
        print(f"📋 Using specified quantization format: {quant_format}")

    # Build quantization layer metadata
    print(f"🔧 Building quantization metadata...")
    quant_layers = {}

    # Scan for quantized layers
    for key in state_dict.keys():
        if key.endswith('.weight_scale') or key.endswith('.input_scale'):
            layer_name = key.rsplit('.', 1)[0]
            if layer_name not in quant_layers:
                quant_layers[layer_name] = {"format": quant_format}

    if not quant_layers:
        print("❌ ERROR: No quantized layers found in model!")
        print("   This model may not be quantized or uses an unsupported format.")
        return False

    print(f"✅ Found {len(quant_layers)} quantized layers")

    # Add .comfy_quant keys for each quantized layer
    print(f"📝 Adding .comfy_quant metadata keys...")
    for layer_name, layer_config in quant_layers.items():
        comfy_quant_key = f"{layer_name}.comfy_quant"
        comfy_quant_value = json.dumps(layer_config).encode('utf-8')
        state_dict[comfy_quant_key] = torch.tensor(list(comfy_quant_value), dtype=torch.uint8)

    # Build metadata for safetensors file header
    metadata = {
        "_quantization_metadata": json.dumps({
            "layers": quant_layers,
            "format": quant_format,
            "quant_algo": quant_algo,
            "version": "1.0"
        })
    }

    # Save fixed model
    print(f"💾 Saving fixed model to: {output_path}")
    save_file(state_dict, str(output_path), metadata=metadata)

    # Verify file was created
    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"✅ SUCCESS! Model saved ({size_mb:.2f} MB)")
        print(f"\n📊 Quantization Metadata Summary:")
        print(f"   Format: {quant_format}")
        print(f"   Algorithm: {quant_algo}")
        print(f"   Quantized Layers: {len(quant_layers)}")
        print(f"\n✨ Your model is now compatible with ComfyUI!")
        return True
    else:
        print("❌ ERROR: Failed to save fixed model!")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Fix quantization metadata for ComfyUI compatibility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect format and fix in-place
  python fix_quantization_metadata.py model_NVFP4_custom.safetensors

  # Specify format explicitly
  python fix_quantization_metadata.py model.safetensors --format nvfp4

  # Save to different output file
  python fix_quantization_metadata.py input.safetensors --output fixed.safetensors

  # Specify quantization algorithm
  python fix_quantization_metadata.py model.safetensors --format nvfp4 --algo svdquant
        """
    )

    parser.add_argument(
        "input",
        type=str,
        help="Input .safetensors file to fix"
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output .safetensors file (defaults to overwriting input)"
    )

    parser.add_argument(
        "-f", "--format",
        type=str,
        choices=["nvfp4", "float8_e4m3fn"],
        default=None,
        help="Quantization format (auto-detect if not specified)"
    )

    parser.add_argument(
        "-a", "--algo",
        type=str,
        default="svdquant",
        help="Quantization algorithm used (default: svdquant)"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("  FIX QUANTIZATION METADATA FOR COMFYUI")
    print("=" * 80)
    print()

    try:
        success = fix_quantization_metadata(
            args.input,
            args.output,
            args.format,
            args.algo
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
