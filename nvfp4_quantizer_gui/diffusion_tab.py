"""
Diffusion Models Quantization Tab
For FLUX, SD3, SDXL models
"""

import json
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path
import gradio as gr
from typing import Tuple

from .common_utils import (
    get_file_path,
    get_folder_path,
    check_venv_exists,
    DIFFUSION_SCRIPT,
    VENV_PYTHON,
    get_model_size,
    get_next_run_log_path,
)

try:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file as safetensors_save
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

# BEST PRESETS FOR DIFFUSION MODELS
# Based on NVIDIA Model-Optimizer configs
DIFFUSION_PRESETS = {
    "Best Quality (Mixed Dev)": {
        "quant_format": "fp4_mixed_dev",
        "quant_algo": "max",
        "calib_size": 512,
        "n_steps": 30,
        "quantize_mha": True,
        "svd_lowrank": 64,
        "compress": False,
        "description": "Flux Dev only: mixed FP4/FP8 layer map for higher quality (larger file).",
    },
    "🚀 Best Quality": {
        "quant_format": "fp4_mixed_dev",
        "quant_algo": "max",
        "calib_size": 512,
        "n_steps": 30,
        "quantize_mha": True,
        "svd_lowrank": 64,
        "compress": False,
        "description": "Maximum quality with mixed FP4/FP8 and MAX algorithm. Best for production use.",
    },
    "⚡ Balanced (Recommended)": {
        "quant_format": "fp4_mixed_dev",
        "quant_algo": "max",
        "calib_size": 256,
        "n_steps": 20,
        "quantize_mha": True,
        "svd_lowrank": 32,
        "compress": False,
        "description": "Best balance using mixed FP4/FP8 with optimal calibration. Recommended for most users.",
    },
    "💨 Fast": {
        "quant_format": "fp4_mixed_dev",
        "quant_algo": "max",
        "calib_size": 128,
        "n_steps": 20,
        "quantize_mha": True,
        "svd_lowrank": 32,
        "compress": False,
        "description": "Quick quantization using mixed FP4/FP8 with faster calibration.",
    },
    "🏃 Ultra Fast Test": {
        "quant_format": "fp4_mixed_dev",
        "quant_algo": "max",
        "calib_size": 64,
        "n_steps": 20,
        "quantize_mha": True,
        "svd_lowrank": 16,
        "compress": False,
        "description": "Quick test using mixed FP4/FP8 with minimal calibration.",
    },
}

DIFFUSION_MODELS = {
    "Auto Detect": "auto",
    "FLUX Dev": "flux-dev",
    "FLUX Schnell": "flux-schnell",
    "SD3 Medium": "sd3-medium",
    "SD3.5 Medium": "sd3.5-medium",
    "SDXL Base": "sdxl-1.0",
    "SDXL Turbo": "sdxl-turbo",
}


def is_valid_quantized_safetensors(path: Path) -> bool:
    if not SAFETENSORS_AVAILABLE or not path.exists():
        return False
    try:
        with safe_open(str(path), framework="pt", device="cpu") as f:
            metadata = f.metadata() or {}
            if "_quantization_metadata" in metadata:
                try:
                    layers = json.loads(metadata["_quantization_metadata"]).get("layers", {})
                    if layers:
                        return True
                except Exception:
                    pass
            for key in f.keys():
                if key.endswith(".weight_scale") or key.endswith(".input_scale"):
                    return True
    except Exception:
        return False
    return False


def auto_detect_model_type(path: str) -> str:
    """Auto-detect model type from filename."""
    filename = Path(path).name.lower()

    if "flux" in filename and "schnell" not in filename:
        return "FLUX Dev"
    elif "schnell" in filename:
        return "FLUX Schnell"
    elif "sd3.5" in filename or "sd35" in filename:
        return "SD3.5 Medium"
    elif "sd3" in filename:
        return "SD3 Medium"
    elif "turbo" in filename:
        return "SDXL Turbo"
    elif "sdxl" in filename or "xl" in filename:
        return "SDXL Base"

    return "FLUX Dev"


def run_diffusion_quantization(
    model_path: str,
    model_type_ui: str,
    output_folder: str,
    quant_format: str,
    quant_algo: str,
    calib_size: int,
    n_steps: int,
    svd_lowrank: int,
    quantize_mha: bool,
    compress: bool,
    convert_to_safetensors: bool,
    vae_path: str = "",
    text_encoder_path: str = "",
    text_encoder_2_path: str = "",
    progress=gr.Progress(),
) -> Tuple[str, str]:
    """
    Run diffusion model quantization.

    Returns:
        Tuple of (log_output, output_file_path)
    """

    log_file = None
    try:
        # Validate inputs
        if not model_path or not model_path.strip():
            return "❌ Please select a model file", ""

        if not os.path.exists(model_path):
            return f"❌ Model file not found: {model_path}", ""

        # Check environment
        venv_ok, msg = check_venv_exists()
        if not venv_ok:
            return msg, ""

        if not DIFFUSION_SCRIPT.exists():
            return f"❌ Quantization script not found: {DIFFUSION_SCRIPT}", ""

        progress(0.1, desc="🔍 Analyzing model...")

        # Detect model type
        if model_type_ui == "Auto Detect":
            detected_type = auto_detect_model_type(model_path)
        else:
            detected_type = model_type_ui

        model_type = DIFFUSION_MODELS[detected_type]

        # Generate output path
        input_path = Path(model_path).expanduser()
        if not input_path.is_absolute():
            input_path = Path(os.path.abspath(str(input_path)))
        if quant_format == "fp4_mixed_dev":
            suffix = "_NVFP4_mixed_custom"
        elif quant_format == "fp4":
            suffix = "_NVFP4_custom"
        else:
            suffix = "_FP8_custom"

        # Use custom output path if provided (file or folder), otherwise save in same folder
        output_path_input = output_folder.strip() if output_folder else ""
        output_dir = None
        safetensors_target_path = None
        if output_path_input:
            output_candidate = Path(output_path_input).expanduser()
            if not output_candidate.is_absolute():
                output_candidate = Path(os.path.abspath(str(output_candidate)))

            is_existing_dir = output_candidate.exists() and output_candidate.is_dir()
            has_suffix = bool(output_candidate.suffix)
            if not is_existing_dir and has_suffix:
                output_dir = output_candidate.parent
                if not output_dir.exists():
                    output_dir.mkdir(parents=True, exist_ok=True)

                if output_candidate.suffix.lower() == ".safetensors":
                    safetensors_target_path = output_candidate
                    output_path = output_candidate.with_suffix(".pt")
                    if not convert_to_safetensors:
                        convert_to_safetensors = True
                else:
                    output_path = output_candidate
            else:
                output_dir = output_candidate
                if not output_dir.exists():
                    output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"{input_path.stem}{suffix}.pt"
        else:
            output_dir = input_path.parent
            output_path = output_dir / f"{input_path.stem}{suffix}.pt"

        output_path = Path(os.path.abspath(str(output_path)))
        if safetensors_target_path is not None:
            safetensors_target_path = Path(os.path.abspath(str(safetensors_target_path)))

        log_path = get_next_run_log_path()
        log_error = None
        try:
            log_file = open(log_path, "wb")
        except Exception as e:
            log_error = str(e)

        progress(0.2, desc="⚙️ Configuring parameters...")

        # Build command based on Model-Optimizer pipeline
        cmd = [
            str(VENV_PYTHON),
            str(DIFFUSION_SCRIPT),
            "--model", model_type,
            "--format", quant_format,
            "--quant-algo", quant_algo,
            "--model-dtype", "BFloat16",
            "--trt-high-precision-dtype", "BFloat16",
            "--calib-size", str(int(calib_size)),
            "--batch-size", "2",
            "--n-steps", str(int(n_steps)),
            "--override-model-path", str(input_path),
            "--quantized-torch-ckpt-save-path", str(output_path),
        ]

        # Add MHA quantization flag
        if quantize_mha:
            cmd.append("--quantize-mha")

        # Add lowrank parameter for SVDQuant
        if quant_algo == "svdquant":
            cmd.extend(["--lowrank", str(int(svd_lowrank))])

        # Add compression flag if enabled
        if compress:
            cmd.append("--compress")

        # Add local component paths if provided
        if vae_path and vae_path.strip():
            cmd.extend(["--vae-path", str(vae_path)])
        if text_encoder_path and text_encoder_path.strip():
            cmd.extend(["--text-encoder-path", str(text_encoder_path)])
        if text_encoder_2_path and text_encoder_2_path.strip():
            cmd.extend(["--text-encoder-2-path", str(text_encoder_2_path)])

        progress(0.25, desc="🚀 Starting quantization...")

        # Format output header with DETAILED LOGGING
        orig_size = get_model_size(model_path)

        result = "╔" + "═" * 78 + "╗\n"
        result += "║" + " " * 22 + "FLUX/DIFFUSION QUANTIZATION" + " " * 29 + "║\n"
        result += "╚" + "═" * 78 + "╝\n\n"

        result += f"📁 Input Configuration:\n"
        result += f"   Model Path: {model_path}\n"
        result += f"   Model Name: {input_path.name}\n"
        result += f"   Model Size: {orig_size}\n"
        result += f"   Model Type: {detected_type}\n\n"

        result += f"💾 Output Configuration:\n"
        result += f"   User Provided Output Path: {output_path_input if output_path_input else 'NO (using input folder)'}\n"
        result += f"   Resolved Output Path (.pt): {output_path}\n"
        if safetensors_target_path is not None:
            result += f"   SafeTensors Target Path: {safetensors_target_path}\n"
        result += f"   Output File Name: {output_path.name}\n"
        result += f"   Output Folder: {output_path.parent}\n"
        result += f"   Convert to SafeTensors: {'✅ YES' if convert_to_safetensors else '❌ NO'}\n\n"
        if log_file is not None:
            result += f"   Log File: {log_path}\n\n"
        else:
            log_status = f"Unavailable ({log_error})" if log_error else "Unavailable"
            result += f"   Log File: {log_status}\n\n"

        result += f"🔢 Quantization Settings:\n"
        result += f"   Format: {quant_format.upper()}\n"
        result += f"   Algorithm: {quant_algo.upper()}\n"
        result += f"   MHA Quantization: {'Enabled' if quantize_mha else 'Disabled'}\n"
        result += f"   Compression: {'Enabled' if compress else 'Disabled'}\n\n"

        result += f"📊 Calibration Settings:\n"
        result += f"   Calibration Samples: {calib_size}\n"
        result += f"   Denoising Steps: {n_steps}\n"
        if quant_algo == "svdquant":
            result += f"   SVD Lowrank: {svd_lowrank}\n"
        result += "\n"

        result += "ℹ️  Note: Calibration runs in 2 passes for optimal quality:\n"
        result += "   Pass 1: Transformer blocks calibration\n"
        result += "   Pass 2: Additional components (MHA, normalization)\n"
        result += "   This is normal and will complete automatically.\n\n"

        result += "═" * 80 + "\n\n"

        # Log the full command for debugging
        result += "🔍 Command Being Executed:\n"
        cmd_str = " ".join([f'"{arg}"' if " " in str(arg) else str(arg) for arg in cmd])
        # Truncate extremely long commands
        if len(cmd_str) > 500:
            result += f"   {cmd_str[:500]}...\n"
            result += f"   (command truncated - total length: {len(cmd_str)} chars)\n"
        else:
            result += f"   {cmd_str}\n"
        result += "\n"
        if log_file is not None:
            log_file.write(f"Command: {cmd_str}\n\n".encode("utf-8", errors="replace"))
            log_file.flush()
        result += f"🔍 Key Arguments Check:\n"
        if "--quantized-torch-ckpt-save-path" in cmd:
            idx = cmd.index("--quantized-torch-ckpt-save-path")
            result += f"   ✅ --quantized-torch-ckpt-save-path: {cmd[idx+1]}\n"
        else:
            result += f"   ❌ --quantized-torch-ckpt-save-path: NOT FOUND IN COMMAND!\n"
        result += "\n"

        result += "═" * 80 + "\n\n"

        progress(0.3, desc="⚡ Quantizing model (this will take a while)...")

        # Execute quantization
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
            cwd=str(DIFFUSION_SCRIPT.parent),
            env=env,
        )

        # Stream output with progress tracking
        output_lines = []
        progress_value = 0.3
        buffer = ""

        while True:
            chunk = process.stdout.read(64)
            if not chunk:
                break

            try:
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
            except Exception:
                sys.stdout.write(chunk.decode(errors="replace"))
                sys.stdout.flush()
            if log_file is not None:
                log_file.write(chunk)
                log_file.flush()

            text_chunk = chunk.decode(errors="replace")
            buffer += text_chunk

            if "\n" in buffer or "\r" in buffer:
                parts = re.split(r"[\r\n]", buffer)
                buffer = parts.pop()
                for line in parts:
                    if not line:
                        continue
                    output_lines.append(line + "\n")

                    # Update progress based on keywords
                    line_lower = line.lower()
                    if "loading" in line or "load" in line_lower:
                        progress_value = max(progress_value, 0.4)
                        progress(progress_value, desc="Loading model...")
                    elif "quantizing" in line or "quantize" in line_lower:
                        progress_value = max(progress_value, 0.55)
                        progress(progress_value, desc="Quantizing layers...")
                    elif "calibrating" in line or "calibration" in line_lower:
                        progress_value = max(progress_value, 0.7)
                        progress(progress_value, desc="Calibrating...")
                    elif "step" in line_lower:
                        progress_value = min(progress_value + 0.001, 0.85)
                        progress(progress_value, desc="Processing calibration...")
                    elif "saving" in line or "save" in line_lower:
                        progress_value = 0.92
                        progress(progress_value, desc="Saving quantized model...")

        if buffer.strip():
            output_lines.append(buffer + "\n")

        process.wait()

        # Append logs (last 80 lines to keep it manageable)
        result += "📝 Quantization Logs (last 80 lines):\n"
        result += "─" * 80 + "\n"
        result += "".join(output_lines[-80:])
        result += "─" * 80 + "\n\n"

        progress(0.95, desc="✨ Finalizing...")

        # Debug: Check if file was created and if save log was seen
        result += f"\n🔍 Post-Quantization File Check:\n"
        result += f"   Expected Path: {output_path}\n"
        result += f"   File Exists: {'✅ YES' if output_path.exists() else '❌ NO'}\n"
        if output_path.exists():
            result += f"   File Size: {get_model_size(str(output_path))}\n"

        # Check if the NVIDIA script logged that it saved the file
        all_logs = "".join(output_lines)
        if "Saving quantized checkpoint" in all_logs:
            result += f"   Save Log Found: ✅ YES - Script attempted to save\n"
        else:
            result += f"   Save Log Found: ❌ NO - Script did not log save attempt!\n"

        if "Checkpoint saved successfully" in all_logs:
            result += f"   Save Success Log: ✅ YES - Script confirmed save\n"
        else:
            result += f"   Save Success Log: ❌ NO - Script did not confirm save\n"
        result += "\n"

        # Check success
        if process.returncode == 0 and output_path.exists():
            output_size = get_model_size(str(output_path))
            final_output_path = output_path

            # Convert to safetensors if requested
            result += "\n🔄 SafeTensors Conversion Process:\n"
            result += f"   Conversion Requested: {'✅ YES' if convert_to_safetensors else '❌ NO'}\n"
            result += f"   SafeTensors Library Available: {'✅ YES' if SAFETENSORS_AVAILABLE else '❌ NO'}\n\n"

            if convert_to_safetensors and SAFETENSORS_AVAILABLE:
                try:
                    progress(0.96, desc="Converting to SafeTensors...")
                    result += "   Starting conversion...\n"

                    safetensors_path = safetensors_target_path or output_path.with_suffix('.safetensors')
                    result += f"   Target Path: {safetensors_path}\n\n"

                    if safetensors_path.exists() and is_valid_quantized_safetensors(safetensors_path):
                        result += "   Existing SafeTensors looks valid; skipping conversion.\n"
                        safetensors_size = get_model_size(str(safetensors_path))
                        final_output_path = safetensors_path
                        output_size = safetensors_size
                    else:
                        converter_script = Path(__file__).resolve().parents[1] / "convert_mto_to_safetensors.py"
                        if not converter_script.exists():
                            raise FileNotFoundError(f"Converter not found: {converter_script}")

                        if quant_format in ("fp4", "fp4_mixed_dev"):
                            quant_format_str = "nvfp4"
                        else:
                            quant_format_str = "float8_e4m3fn"
                        cmd_conv = [
                            str(VENV_PYTHON),
                            str(converter_script),
                            str(output_path),
                            str(safetensors_path),
                            "--model-type",
                            model_type,
                            "--format",
                            quant_format_str,
                            "--algo",
                            quant_algo,
                        ]
                        result += f"   Running converter: {' '.join(cmd_conv)}\n"
                        conv = subprocess.run(cmd_conv, capture_output=True, text=True)

                        if conv.stdout:
                            result += conv.stdout[-4000:] + "\n"
                        if conv.stderr:
                            result += conv.stderr[-4000:] + "\n"

                        if safetensors_path.exists() and is_valid_quantized_safetensors(safetensors_path):
                            safetensors_size = get_model_size(str(safetensors_path))
                            result += "   Conversion OK\n"
                            result += f"   SafeTensors File: {safetensors_path.name}\n"
                            result += f"   Size: {safetensors_size}\n\n"
                            final_output_path = safetensors_path
                            output_size = safetensors_size
                        else:
                            result += "   Conversion failed or output is invalid.\n"
                            result += "   Using .pt file instead\n\n"

                except Exception as e:
                    result += f"   Conversion error: {str(e)}\n"
                    result += "   Using .pt file instead\n\n"
            elif convert_to_safetensors and not SAFETENSORS_AVAILABLE:
                result += "   ⚠️ SafeTensors library not installed!\n"
                result += "   ℹ️  Install with: pip install safetensors\n"
                result += "   Using .pt format\n\n"
            else:
                result += "   ℹ️  Skipping conversion (not requested)\n"
                result += "   Using .pt format\n\n"

            result += "\n" + "╔" + "═" * 78 + "╗\n"
            result += "║" + " " * 32 + "✅ SUCCESS!" + " " * 33 + "║\n"
            result += "╚" + "═" * 78 + "╝\n\n"

            result += f"📦 Quantized Model Saved:\n"
            result += f"   {final_output_path}\n\n"

            result += f"📊 File Sizes:\n"
            result += f"   Original: {orig_size}\n"
            result += f"   Quantized: {output_size}\n\n"

            result += f"✨ Your {'SafeTensors' if final_output_path.suffix == '.safetensors' else 'NVFP4'} model is ready!\n\n"

            result += f"📝 Next Steps:\n"
            result += f"   • Test with sample generations\n"
            result += f"   • Compare quality with original\n"
            if final_output_path.suffix != '.safetensors':
                result += f"   • Convert to .safetensors if needed for ComfyUI\n"

            progress(1.0, desc="✅ Complete!")
            return result, str(final_output_path)

        else:
            result += "\n" + "═" * 80 + "\n"

            if process.returncode == 0 and not output_path.exists():
                result += f"❌ QUANTIZATION COMPLETED BUT FILE NOT SAVED!\n\n"
                result += f"The quantization script finished successfully (exit code 0),\n"
                result += f"but the expected output file was not created:\n"
                result += f"   {output_path}\n\n"
                result += f"This usually means the --quantized-torch-ckpt-save-path argument\n"
                result += f"was not processed correctly by the script.\n\n"
                result += f"💡 Check the logs above for any 'Saving checkpoint' messages.\n"
            else:
                result += f"❌ QUANTIZATION FAILED (Exit Code: {process.returncode})\n\n"
                result += "Please check the logs above for error details.\n\n"

            if "CUDA out of memory" in "".join(output_lines):
                result += "💡 TIP: Out of memory error detected.\n"
                result += "   Try 'Ultra Fast Test' preset or close other GPU applications.\n"

            return result, ""

    except Exception as e:
        error_msg = f"❌ Unexpected Error:\n\n{str(e)}\n\n{traceback.format_exc()}"
        return error_msg, ""
    finally:
        if log_file is not None:
            log_file.close()


def diffusion_quantization_tab(headless: bool = False):
    """
    Create the diffusion models quantization tab.

    Args:
        headless: Whether running in headless mode
    """

    gr.Markdown(
        """
        ## 🎨 FLUX / Diffusion Model Quantization

        **For FLUX-SRPO-bf16.safetensors and other diffusion models**

        Supports: FLUX Dev/Schnell, SD3/SD3.5, SDXL, SDXL Turbo
        """,
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📁 Step 1: Select Your Model")

            with gr.Row():
                model_path = gr.Textbox(
                    label="Model File",
                    value="/workspace/FLUX-SRPO-bf16.safetensors",
                    placeholder="Click 📂 to browse for your .safetensors file",
                    scale=5,
                    info="Select FLUX, SD3, or SDXL model file"
                )
                model_btn = gr.Button("📂", scale=1, size="lg")

            with gr.Row():
                output_folder = gr.Textbox(
                    label="Output Path (Optional)",
                    placeholder="Leave empty to save in same folder as input model",
                    scale=5,
                    info="Provide a folder or a filename (.pt or .safetensors). If empty, saves next to input with _NVFP4 suffix (_NVFP4_mixed for mixed dev)."
                )
                output_folder_btn = gr.Button("📂", scale=1, size="lg")

            model_type = gr.Dropdown(
                label="Model Type",
                choices=list(DIFFUSION_MODELS.keys()),
                value="FLUX Dev",
                info="Auto-detect from filename or select manually"
            )

            gr.Markdown("---")
            gr.Markdown("### 🔧 Optional: Local VAE & Text Encoders")
            gr.Markdown("*Leave empty to use HuggingFace defaults. Only needed for fully offline operation.*")

            with gr.Row():
                vae_path = gr.Textbox(
                    label="VAE Model (Optional)",
                    value="/workspace/ae.safetensors",
                    placeholder="e.g., ae.safetensors",
                    scale=5,
                    info="Local VAE model path. Leave empty to download from HuggingFace."
                )
                vae_btn = gr.Button("📂", scale=1, size="lg")

            with gr.Row():
                text_encoder_path = gr.Textbox(
                    label="CLIP Text Encoder (Optional)",
                    value="/workspace/clip_l.safetensors",
                    placeholder="e.g., clip_l.safetensors",
                    scale=5,
                    info="Local CLIP text encoder path. Leave empty to download from HuggingFace."
                )
                text_encoder_btn = gr.Button("📂", scale=1, size="lg")

            with gr.Row():
                text_encoder_2_path = gr.Textbox(
                    label="T5 Text Encoder (Optional)",
                    value="/workspace/t5xxl_fp16.safetensors",
                    placeholder="e.g., t5xxl_fp16.safetensors",
                    scale=5,
                    info="Local T5 text encoder for FLUX/SD3. Leave empty to download from HuggingFace."
                )
                text_encoder_2_btn = gr.Button("📂", scale=1, size="lg")

    gr.Markdown("---")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Step 2: Choose Quality Preset or Customize")

            preset = gr.Radio(
                choices=list(DIFFUSION_PRESETS.keys()),
                value="⚡ Balanced (Recommended)",
                label="Quantization Preset",
                info="Select a preset to auto-fill parameters below, then customize if needed"
            )

            preset_info = gr.Markdown(
                DIFFUSION_PRESETS["⚡ Balanced (Recommended)"]["description"],
                elem_id="preset-info"
            )

    gr.Markdown("---")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🔧 Advanced Parameters (Customizable)")
            gr.Markdown("*These are auto-filled from preset. You can modify any parameter below.*")

            with gr.Row():
                quant_format = gr.Dropdown(
                    choices=["fp4", "fp4_mixed_dev", "fp8"],
                    value="fp4_mixed_dev",
                    label="Quantization Format",
                    info="FP4 (NVFP4) = 75% compression with 4-bit precision | FP4 mixed dev = NVFP4 with FP8 for select FLUX layers (larger file, higher quality) | FP8 = 50% compression with 8-bit precision."
                )

                quant_algo = gr.Dropdown(
                    choices=["svdquant", "max"],
                    value="max",
                    label="Quantization Algorithm",
                    info="MAX = Faster calibration using max absolute values (recommended) | SVDQuant = Uses matrix decomposition for superior quality but slower"
                )

            with gr.Row():
                calib_size = gr.Number(
                    value=256,
                    label="Calibration Samples",
                    info="Number of prompts used to measure activation ranges during calibration. More samples = better statistics = higher quality. 64=fast, 256=balanced, 512=best quality. Diminishing returns beyond 512.",
                    precision=0
                )

                n_steps = gr.Number(
                    value=20,
                    label="Denoising Steps",
                    info="Number of diffusion denoising iterations per calibration sample. More steps capture fuller range of model behavior. 20=sufficient, 28-30=maximum quality. Each sample runs through this many denoising steps.",
                    precision=0
                )

            with gr.Row():
                svd_lowrank = gr.Number(
                    value=32,
                    label="SVD Lowrank",
                    info="For SVDQuant only: Controls matrix decomposition rank. Higher rank = more singular values kept = better quality but slower. 16=ultra-fast, 32=balanced (recommended), 64=maximum quality. Ignored for MAX algorithm.",
                    precision=0
                )

                quantize_mha = gr.Checkbox(
                    value=True,
                    label="Quantize Multi-Head Attention",
                    info="Quantizes attention operations (Q,K,V projections) to FP8 for better compression. Preserves attention patterns while reducing memory. Minimal quality loss. Recommended: Enable for all models."
                )

            with gr.Row():
                compress = gr.Checkbox(
                    value=False,
                    label="Enable Compression",
                    info="Applies additional compression beyond quantization for 10-20% extra size reduction. EXPERIMENTAL - may cause compatibility issues. Only for FP4/FP8. Recommended: Keep disabled unless maximum compression is critical."
                )

                convert_to_safetensors = gr.Checkbox(
                    value=True,
                    label="Auto-Convert to SafeTensors",
                    info="✅ Enabled by default - Converts .pt output to .safetensors format for ComfyUI compatibility. SafeTensors = safer, faster loading. Disable only if using PyTorch directly."
                )

    gr.Markdown("---")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🚀 Step 3: Start Quantization")

            quantize_btn = gr.Button(
                "⚡ START QUANTIZATION",
                variant="primary",
                size="lg",
            )

            gr.Markdown("*Duration varies based on GPU, model size, and preset*")

    gr.Markdown("---")

    with gr.Row():
        output_logs = gr.Textbox(
            label="📊 Quantization Progress & Logs",
            lines=28,
            max_lines=50,
            interactive=False,
            placeholder="Click 'START QUANTIZATION' to begin...\n\nOutput file will use _NVFP4 suffix (_NVFP4_mixed for mixed dev) unless you provide a filename.\nIf no output path is specified, it will be saved in the same folder as your input model.",
        )

    with gr.Row():
        output_file = gr.Textbox(
            label="💾 Quantized Model Output Path",
            interactive=False,
            placeholder="Output file path will appear here after completion",
        )

    # Update preset info and parameters on change
    def update_preset_params(preset_name: str):
        params = DIFFUSION_PRESETS[preset_name]
        return (
            params["description"],  # preset_info
            params["quant_format"],  # quant_format
            params["quant_algo"],  # quant_algo
            params["calib_size"],  # calib_size
            params["n_steps"],  # n_steps
            params["svd_lowrank"],  # svd_lowrank
            params["quantize_mha"],  # quantize_mha
            params["compress"],  # compress
        )

    preset.change(
        fn=update_preset_params,
        inputs=[preset],
        outputs=[preset_info, quant_format, quant_algo, calib_size, n_steps, svd_lowrank, quantize_mha, compress],
        show_progress=False,
    )

    # File browser
    model_btn.click(
        fn=lambda: get_file_path("", "*.safetensors;*.pt;*.pth", "Select Model File"),
        outputs=[model_path],
        show_progress=False,
    )

    # Output folder browser
    output_folder_btn.click(
        fn=lambda: get_folder_path("", "Select Output Folder"),
        outputs=[output_folder],
        show_progress=False,
    )

    # VAE browser
    vae_btn.click(
        fn=lambda: get_file_path("", "*.safetensors;*.pt;*.pth", "Select VAE Model"),
        outputs=[vae_path],
        show_progress=False,
    )

    # CLIP text encoder browser
    text_encoder_btn.click(
        fn=lambda: get_file_path("", "*.safetensors;*.pt;*.pth", "Select CLIP Text Encoder"),
        outputs=[text_encoder_path],
        show_progress=False,
    )

    # T5 text encoder browser
    text_encoder_2_btn.click(
        fn=lambda: get_file_path("", "*.safetensors;*.pt;*.pth", "Select T5 Text Encoder"),
        outputs=[text_encoder_2_path],
        show_progress=False,
    )

    # Main quantization
    quantize_btn.click(
        fn=run_diffusion_quantization,
        inputs=[
            model_path,
            model_type,
            output_folder,
            quant_format,
            quant_algo,
            calib_size,
            n_steps,
            svd_lowrank,
            quantize_mha,
            compress,
            convert_to_safetensors,
            vae_path,
            text_encoder_path,
            text_encoder_2_path
        ],
        outputs=[output_logs, output_file],
        show_progress=True,
    )
