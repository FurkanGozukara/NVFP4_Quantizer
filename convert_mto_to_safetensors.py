#!/usr/bin/env python3
"""
Convert NVIDIA ModelOpt checkpoint to SafeTensors with ComfyUI metadata.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

try:
    import torch
    import modelopt.torch.opt as mto
except ImportError as e:
    print(f"Error: Missing required libraries: {e}")
    print("Please install: pip install torch safetensors nvidia-modelopt")
    sys.exit(1)


def _load_export_helper():
    repo_root = Path(__file__).resolve().parent
    helper_dir = repo_root / "Model-Optimizer" / "examples" / "diffusers" / "quantization"
    if not helper_dir.exists():
        raise FileNotFoundError(f"Export helper not found: {helper_dir}")
    sys.path.insert(0, str(helper_dir))
    from save_quantized_safetensors import save_quantized_safetensors
    return save_quantized_safetensors


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("convert_mto_to_safetensors")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", "%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def convert_mto_to_safetensors(
    mto_checkpoint_path: str,
    output_safetensors_path: str,
    quant_format: str = "nvfp4",
    quant_algo: str = "svdquant",
    model_type: str = "flux-dev",
) -> bool:
    """
    Convert NVIDIA ModelOpt .pt checkpoint to SafeTensors with ComfyUI metadata.
    """
    logger = _setup_logger()
    mto_path = Path(mto_checkpoint_path)
    if not mto_path.exists():
        raise FileNotFoundError(f"ModelOpt checkpoint not found: {mto_path}")

    output_path = Path(output_safetensors_path)

    logger.info("Loading ModelOpt checkpoint: %s", mto_path)
    logger.info("Format: %s | Algorithm: %s | Model type: %s", quant_format, quant_algo, model_type)

    logger.info("Preparing base transformer architecture...")
    from diffusers import FluxTransformer2DModel, StableDiffusion3Transformer2DModel

    if model_type in ["flux-dev", "flux-schnell"]:
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
        backbone = StableDiffusion3Transformer2DModel.from_pretrained(
            "stabilityai/stable-diffusion-3-medium",
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    logger.info("Restoring quantized weights from ModelOpt checkpoint...")
    mto.restore(backbone, str(mto_path))
    logger.info("Quantized weights restored.")

    save_quantized_safetensors = _load_export_helper()
    success = save_quantized_safetensors(
        backbone,
        output_path,
        quant_format=quant_format,
        quant_algo=quant_algo,
        logger=logger,
    )
    return success


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert NVIDIA ModelOpt checkpoint to SafeTensors with ComfyUI metadata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python convert_mto_to_safetensors.py model_NVFP4.pt output.safetensors --model-type flux-dev
  python convert_mto_to_safetensors.py model.pt output.safetensors --model-type sd3-medium --format nvfp4 --algo svdquant
        """,
    )

    parser.add_argument("input", type=str, help="Input ModelOpt .pt checkpoint")
    parser.add_argument("output", type=str, help="Output .safetensors file")
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["flux-dev", "flux-schnell", "sd3-medium", "sd3.5-medium"],
        help="Model architecture type",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="nvfp4",
        choices=["nvfp4", "float8_e4m3fn"],
        help="Quantization format",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="svdquant",
        help="Quantization algorithm",
    )

    args = parser.parse_args()

    try:
        success = convert_mto_to_safetensors(
            args.input,
            args.output,
            args.format,
            args.algo,
            args.model_type,
        )
        sys.exit(0 if success else 1)
    except Exception as exc:
        print(f"ERROR: {exc}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
