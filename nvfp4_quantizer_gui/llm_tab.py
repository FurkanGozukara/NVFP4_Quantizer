"""
LLM Quantization Tab
For Llama, Mistral, Qwen, and other large language models
"""

import os
import subprocess
import traceback
from pathlib import Path
import gradio as gr
from typing import Tuple

from .common_utils import (
    get_file_path,
    get_folder_path,
    check_venv_exists,
    LLM_SCRIPT,
    VENV_PYTHON,
)

# LLM Quantization formats
LLM_QUANT_FORMATS = {
    "NVFP4 (Default)": "nvfp4",
    "NVFP4 AWQ": "nvfp4_awq",
    "NVFP4 MLP Only": "nvfp4_mlp_only",
    "W4A8 NVFP4 FP8 (Mixed)": "w4a8_nvfp4_fp8",
}

# Calibration datasets
DATASETS = [
    "cnn_dailymail",
    "pileval",
    "wikipedia",
    "c4",
]

# KV Cache formats
KV_CACHE_FORMATS = {
    "None": "none",
    "FP8": "fp8",
    "NVFP4": "nvfp4",
}


def run_llm_quantization(
    model_path: str,
    output_dir: str,
    quant_format_ui: str,
    kv_cache_ui: str,
    dataset: str,
    calib_size: int,
    batch_size: int,
    max_seq_len: int,
    progress=gr.Progress(),
) -> Tuple[str, str]:
    """
    Run LLM quantization.

    Returns:
        Tuple of (log_output, output_directory_path)
    """

    try:
        # Validate inputs
        if not model_path or not model_path.strip():
            return "❌ Please specify model path (HuggingFace ID or local path)", ""

        if not output_dir or not output_dir.strip():
            return "❌ Please specify output directory", ""

        # Check environment
        venv_ok, msg = check_venv_exists()
        if not venv_ok:
            return msg, ""

        if not LLM_SCRIPT.exists():
            return f"❌ Quantization script not found: {LLM_SCRIPT}", ""

        progress(0.1, desc="⚙️ Configuring quantization...")

        # Map UI selections
        quant_format = LLM_QUANT_FORMATS[quant_format_ui]
        kv_cache = KV_CACHE_FORMATS[kv_cache_ui]

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Build command
        cmd = [
            str(VENV_PYTHON),
            str(LLM_SCRIPT),
            "--model_dir", model_path,
            "--qformat", quant_format,
            "--dataset", dataset,
            "--calib_size", str(calib_size),
            "--batch_size", str(batch_size),
            "--tp_size", "1",  # Single GPU
            "--max_seq_len", str(max_seq_len),
            "--export_fmt", "hf",
            "--hf_ckpt_path", output_dir,
        ]

        # Add KV cache if not none
        if kv_cache != "none":
            cmd.extend(["--kv_cache_dtype", kv_cache])

        progress(0.2, desc="🚀 Starting LLM quantization...")

        # Format output header
        result = "╔" + "═" * 78 + "╗\n"
        result += "║" + " " * 30 + "LLM QUANTIZATION" + " " * 32 + "║\n"
        result += "╚" + "═" * 78 + "╝\n\n"

        result += f"🤖 Model:\n"
        result += f"   {model_path}\n\n"

        result += f"💾 Output Directory:\n"
        result += f"   {output_dir}\n\n"

        result += f"🔢 Quantization:\n"
        result += f"   Format: {quant_format_ui}\n"
        result += f"   KV Cache: {kv_cache_ui}\n\n"

        result += f"📊 Calibration:\n"
        result += f"   Dataset: {dataset}\n"
        result += f"   Samples: {calib_size}\n"
        result += f"   Batch Size: {batch_size}\n"
        result += f"   Max Sequence Length: {max_seq_len}\n\n"

        result += "═" * 80 + "\n\n"

        progress(0.3, desc="⚡ Quantizing model (may take 10-60 minutes)...")

        # Execute quantization
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            cwd=str(LLM_SCRIPT.parent)
        )

        # Stream output with progress tracking
        output_lines = []
        progress_value = 0.3

        for line in process.stdout:
            output_lines.append(line)

            # Update progress based on keywords
            if "Loading" in line or "load" in line.lower():
                progress_value = max(progress_value, 0.4)
                progress(progress_value, desc="📥 Loading model...")
            elif "Quantizing" in line or "quantize" in line.lower():
                progress_value = max(progress_value, 0.55)
                progress(progress_value, desc="🔢 Quantizing...")
            elif "Calibrating" in line or "calibration" in line.lower():
                progress_value = max(progress_value, 0.7)
                progress(progress_value, desc="📊 Calibrating...")
            elif "Exporting" in line or "export" in line.lower():
                progress_value = 0.9
                progress(progress_value, desc="💾 Exporting...")
            elif "step" in line.lower() or "batch" in line.lower():
                progress_value = min(progress_value + 0.001, 0.85)
                progress(progress_value, desc="🔄 Processing...")

        process.wait()

        # Append logs
        result += "📝 Quantization Logs:\n"
        result += "─" * 80 + "\n"
        result += "".join(output_lines[-80:])
        result += "─" * 80 + "\n\n"

        progress(0.95, desc="✨ Finalizing...")

        # Check success
        if process.returncode == 0:
            result += "\n" + "╔" + "═" * 78 + "╗\n"
            result += "║" + " " * 32 + "✅ SUCCESS!" + " " * 33 + "║\n"
            result += "╚" + "═" * 78 + "╝\n\n"

            result += f"📦 Quantized Model Saved:\n"
            result += f"   {output_dir}\n\n"

            result += f"🔢 Format: {quant_format_ui}\n"
            result += f"   KV Cache: {kv_cache_ui}\n\n"

            result += f"✨ Your quantized LLM is ready!\n\n"

            result += f"📝 Usage:\n"
            result += f"   Load with: transformers.AutoModelForCausalLM.from_pretrained('{output_dir}')\n\n"

            result += f"💡 Next Steps:\n"
            result += f"   • Test with inference\n"
            result += f"   • Evaluate perplexity/accuracy\n"
            result += f"   • Deploy to TensorRT-LLM if needed\n"

            progress(1.0, desc="✅ Complete!")
            return result, output_dir

        else:
            result += "\n" + "═" * 80 + "\n"
            result += f"❌ QUANTIZATION FAILED (Exit Code: {process.returncode})\n\n"
            result += "Please check the logs above for error details.\n"

            return result, ""

    except Exception as e:
        error_msg = f"❌ Unexpected Error:\n\n{str(e)}\n\n{traceback.format_exc()}"
        return error_msg, ""


def llm_quantization_tab(headless: bool = False):
    """
    Create the LLM quantization tab.

    Args:
        headless: Whether running in headless mode
    """

    gr.Markdown(
        """
        ## 🤖 Large Language Model Quantization

        **For Llama, Mistral, Qwen, Phi, Gemma, and other LLMs**

        Supports any HuggingFace causal language model
        """,
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📁 Model Configuration")

            with gr.Row():
                model_path = gr.Textbox(
                    label="Model Path",
                    placeholder="HuggingFace model ID (e.g., meta-llama/Llama-2-7b-hf) or local path",
                    scale=5,
                    info="Enter HuggingFace model ID or browse to local model folder"
                )
                model_btn = gr.Button("📂", scale=1, size="lg")

            with gr.Row():
                output_dir = gr.Textbox(
                    label="Output Directory",
                    placeholder="Where to save the quantized model",
                    scale=5,
                    info="Directory where quantized model will be saved"
                )
                output_btn = gr.Button("📂", scale=1, size="lg")

    gr.Markdown("---")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### ⚙️ Quantization Settings")

            quant_format = gr.Dropdown(
                label="Quantization Format",
                choices=list(LLM_QUANT_FORMATS.keys()),
                value="NVFP4 (Default)",
                info="NVFP4 provides ~4x compression with minimal quality loss"
            )

            kv_cache = gr.Dropdown(
                label="KV Cache Quantization",
                choices=list(KV_CACHE_FORMATS.keys()),
                value="None",
                info="Quantize KV cache for lower inference memory usage"
            )

        with gr.Column():
            gr.Markdown("### 📊 Calibration Settings")

            dataset = gr.Dropdown(
                label="Calibration Dataset",
                choices=DATASETS,
                value="cnn_dailymail",
                info="Dataset used for quantization calibration"
            )

            calib_size = gr.Slider(
                label="Calibration Samples",
                minimum=32,
                maximum=2048,
                value=512,
                step=32,
                info="More samples = better quality, slower (512 recommended)"
            )

            batch_size = gr.Slider(
                label="Batch Size",
                minimum=1,
                maximum=32,
                value=1,
                step=1,
                info="Adjust based on GPU memory (1 recommended for safety)"
            )

            max_seq_len = gr.Slider(
                label="Max Sequence Length",
                minimum=128,
                maximum=8192,
                value=2048,
                step=128,
                info="Maximum sequence length for calibration"
            )

    gr.Markdown("---")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🚀 Start Quantization")

            quantize_btn = gr.Button(
                "⚡ START LLM QUANTIZATION",
                variant="primary",
                size="lg",
            )

            gr.Markdown("*Estimated time: 10-60 minutes depending on model size*")

    gr.Markdown("---")

    with gr.Row():
        output_logs = gr.Textbox(
            label="📊 Quantization Progress & Logs",
            lines=28,
            max_lines=50,
            interactive=False,
            placeholder="Click 'START LLM QUANTIZATION' to begin...\n\nNote: First time may download model from HuggingFace if not local.",
        )

    with gr.Row():
        output_path = gr.Textbox(
            label="💾 Quantized Model Output Directory",
            interactive=False,
            placeholder="Output directory path will appear here after completion",
        )

    # File/folder browsers
    model_btn.click(
        fn=lambda: get_folder_path("", "Select Model Directory"),
        outputs=[model_path],
        show_progress=False,
    )

    output_btn.click(
        fn=lambda: get_folder_path("", "Select Output Directory"),
        outputs=[output_dir],
        show_progress=False,
    )

    # Main quantization
    quantize_btn.click(
        fn=run_llm_quantization,
        inputs=[model_path, output_dir, quant_format, kv_cache, dataset, calib_size, batch_size, max_seq_len],
        outputs=[output_logs, output_path],
        show_progress=True,
    )
