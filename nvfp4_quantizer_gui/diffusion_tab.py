"""
Diffusion Models Quantization Tab
For FLUX, SD3, SDXL models
"""

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
)

# BEST PRESETS FOR DIFFUSION MODELS
# Based on NVIDIA Model-Optimizer configs
DIFFUSION_PRESETS = {
    "🚀 Best Quality (30-40 min)": {
        "quant_format": "fp4",
        "quant_algo": "svdquant",
        "calib_size": 512,
        "n_steps": 30,
        "quantize_mha": True,
        "svd_lowrank": 64,
        "compress": False,
        "description": "Maximum quality with SVDQuant algorithm. Perfect for production use.",
    },
    "⚡ Balanced (Recommended, 15-20 min)": {
        "quant_format": "fp4",
        "quant_algo": "svdquant",
        "calib_size": 256,
        "n_steps": 20,
        "quantize_mha": True,
        "svd_lowrank": 32,
        "compress": False,
        "description": "Best balance using SVDQuant with optimal calibration. Recommended for most users.",
    },
    "💨 Fast (10-15 min)": {
        "quant_format": "fp4",
        "quant_algo": "max",
        "calib_size": 128,
        "n_steps": 20,
        "quantize_mha": True,
        "svd_lowrank": 32,
        "compress": False,
        "description": "Quick quantization with MAX algorithm for faster calibration.",
    },
    "🏃 Ultra Fast Test (5-10 min)": {
        "quant_format": "fp8",
        "quant_algo": "max",
        "calib_size": 64,
        "n_steps": 20,
        "quantize_mha": True,
        "svd_lowrank": 16,
        "compress": False,
        "description": "Quick test with FP8 format (50% compression vs NVFP4's 75%).",
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
    preset_name: str,
    model_type_ui: str,
    output_folder: str,
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

        # Get preset parameters
        params = DIFFUSION_PRESETS[preset_name]

        # Generate output path
        input_path = Path(model_path)
        suffix = "_NVFP4_HQ" if "Best" in preset_name else "_NVFP4"

        # Use custom output folder if provided, otherwise save in same folder
        if output_folder and output_folder.strip():
            output_dir = Path(output_folder)
            if not output_dir.exists():
                output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{input_path.stem}{suffix}.pt"
        else:
            output_path = input_path.parent / f"{input_path.stem}{suffix}.pt"

        progress(0.2, desc="⚙️ Configuring parameters...")

        # Build command based on Model-Optimizer pipeline
        cmd = [
            str(VENV_PYTHON),
            str(DIFFUSION_SCRIPT),
            "--model", model_type,
            "--format", params["quant_format"],
            "--quant-algo", params["quant_algo"],
            "--model-dtype", "BFloat16",
            "--trt-high-precision-dtype", "BFloat16",
            "--calib-size", str(params["calib_size"]),
            "--batch-size", "2",
            "--n-steps", str(params["n_steps"]),
            "--override-model-path", str(model_path),
            "--quantized-torch-ckpt-save-path", str(output_path),
        ]

        # Add MHA quantization flag
        if params["quantize_mha"]:
            cmd.append("--quantize-mha")

        # Add lowrank parameter for SVDQuant
        if params["quant_algo"] == "svdquant":
            cmd.extend(["--lowrank", str(params["svd_lowrank"])])

        # Add compression flag if enabled
        if params.get("compress", False):
            cmd.append("--compress")

        # Add local component paths if provided
        if vae_path and vae_path.strip():
            cmd.extend(["--vae-path", str(vae_path)])
        if text_encoder_path and text_encoder_path.strip():
            cmd.extend(["--text-encoder-path", str(text_encoder_path)])
        if text_encoder_2_path and text_encoder_2_path.strip():
            cmd.extend(["--text-encoder-2-path", str(text_encoder_2_path)])

        progress(0.25, desc="🚀 Starting quantization...")

        # Format output header
        orig_size = get_model_size(model_path)

        result = "╔" + "═" * 78 + "╗\n"
        result += "║" + " " * 22 + "FLUX/DIFFUSION QUANTIZATION" + " " * 29 + "║\n"
        result += "╚" + "═" * 78 + "╝\n\n"

        result += f"📁 Input File:\n"
        result += f"   {input_path.name}\n"
        result += f"   Size: {orig_size}\n\n"

        result += f"💾 Output File:\n"
        result += f"   {output_path.name}\n\n"

        result += f"🎯 Model Type: {detected_type}\n"
        result += f"⚙️  Preset: {preset_name}\n"
        result += f"   {params['description']}\n\n"

        result += f"🔢 Quantization:\n"
        result += f"   Format: {params['quant_format'].upper()}\n"
        result += f"   Algorithm: {params['quant_algo'].upper()}\n"
        result += f"   MHA: {'Enabled' if params['quantize_mha'] else 'Disabled'}\n\n"

        result += f"📊 Calibration:\n"
        result += f"   Samples: {params['calib_size']}\n"
        result += f"   Steps: {params['n_steps']}\n\n"

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
        result += "📝 Quantization Logs:\n"
        result += "─" * 80 + "\n"
        result += "".join(output_lines[-80:])
        result += "─" * 80 + "\n\n"

        progress(0.95, desc="✨ Finalizing...")

        # Check success
        if process.returncode == 0 and output_path.exists():
            output_size = get_model_size(str(output_path))

            result += "\n" + "╔" + "═" * 78 + "╗\n"
            result += "║" + " " * 32 + "✅ SUCCESS!" + " " * 33 + "║\n"
            result += "╚" + "═" * 78 + "╝\n\n"

            result += f"📦 Quantized Model Saved:\n"
            result += f"   {output_path}\n\n"

            result += f"📊 File Sizes:\n"
            result += f"   Original: {orig_size}\n"
            result += f"   Quantized: {output_size}\n\n"

            result += f"✨ Your NVFP4 model is ready!\n\n"

            result += f"📝 Next Steps:\n"
            result += f"   • Test with sample generations\n"
            result += f"   • Compare quality with original\n"
            result += f"   • Convert to .safetensors if needed for ComfyUI\n"

            progress(1.0, desc="✅ Complete!")
            return result, str(output_path)

        else:
            result += "\n" + "═" * 80 + "\n"
            result += f"❌ QUANTIZATION FAILED (Exit Code: {process.returncode})\n\n"
            result += "Please check the logs above for error details.\n\n"

            if "CUDA out of memory" in "".join(output_lines):
                result += "💡 TIP: Out of memory error detected.\n"
                result += "   Try 'Ultra Fast Test' preset or close other GPU applications.\n"

            return result, ""

    except Exception as e:
        error_msg = f"❌ Unexpected Error:\n\n{str(e)}\n\n{traceback.format_exc()}"
        return error_msg, ""


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
                    label="Output Folder (Optional)",
                    placeholder="Leave empty to save in same folder as input model",
                    scale=5,
                    info="Custom folder to save quantized model. If not provided, saves in same folder with _NVFP4 suffix"
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
            gr.Markdown("### ⚙️ Step 2: Choose Quality Preset")

            preset = gr.Radio(
                choices=list(DIFFUSION_PRESETS.keys()),
                value="⚡ Balanced (Recommended, 15-20 min)",
                label="Quantization Preset",
                info="All parameters pre-configured for optimal results"
            )

            preset_info = gr.Markdown(
                DIFFUSION_PRESETS["⚡ Balanced (Recommended, 15-20 min)"]["description"],
                elem_id="preset-info"
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

            gr.Markdown("*Estimated time: 15-40 minutes depending on preset and GPU*")

    gr.Markdown("---")

    with gr.Row():
        output_logs = gr.Textbox(
            label="📊 Quantization Progress & Logs",
            lines=28,
            max_lines=50,
            interactive=False,
            placeholder="Click 'START QUANTIZATION' to begin...\n\nOutput file will be saved with _NVFP4 suffix.\nIf no output folder is specified, it will be saved in the same folder as your input model.",
        )

    with gr.Row():
        output_file = gr.Textbox(
            label="💾 Quantized Model Output Path",
            interactive=False,
            placeholder="Output file path will appear here after completion",
        )

    # Update preset info on change
    def update_preset_info(preset_name: str):
        return DIFFUSION_PRESETS[preset_name]["description"]

    preset.change(
        fn=update_preset_info,
        inputs=[preset],
        outputs=[preset_info],
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
        inputs=[model_path, preset, model_type, output_folder, vae_path, text_encoder_path, text_encoder_2_path],
        outputs=[output_logs, output_file],
        show_progress=True,
    )
