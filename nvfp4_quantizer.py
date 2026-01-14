#!/usr/bin/env python3
"""
NVFP4 Model Quantizer - Main Application
Modular GUI for quantizing diffusion models and LLMs to NVFP4 format
"""

import os
import sys
import argparse
import gradio as gr
from pathlib import Path

# Add GUI module to path
sys.path.insert(0, str(Path(__file__).parent))

from nvfp4_quantizer_gui.diffusion_tab import diffusion_quantization_tab
from nvfp4_quantizer_gui.llm_tab import llm_quantization_tab
from nvfp4_quantizer_gui.common_utils import check_venv_exists


def create_ui(headless: bool = False):
    """Create the main Gradio UI with tabs."""

    # Check environment status
    venv_ok, venv_msg = check_venv_exists()

    # Create main Blocks interface
    with gr.Blocks(
        title="NVFP4 Model Quantizer",
    ) as app:

        # Header
        gr.Markdown(
            """
            # ⚡ NVFP4 Model Quantizer V1 : https://www.patreon.com/posts/148217625 
            ### All-in-One Tool for Quantizing AI Models to NVFP4 Format

            Reduce model size by 2-4x while maintaining quality using NVIDIA ModelOpt.
            """,
        )

        # Status indicator
        with gr.Row():
            with gr.Column():
                status_msg = gr.Markdown(
                    f"**🔧 Environment Status:** {venv_msg}",
                    elem_classes="status-ok" if venv_ok else "status-error"
                )

        gr.Markdown("---")

        # Create tabs
        with gr.Tabs(elem_classes="tab-nav"):

            # Diffusion models tab
            with gr.Tab("🎨 FLUX / Diffusion Models"):
                diffusion_quantization_tab(headless=headless)

            # LLM tab
            with gr.Tab("🤖 Large Language Models"):
                llm_quantization_tab(headless=headless)

            # Help tab
            with gr.Tab("💡 Help & Tips"):
                gr.Markdown(
                    """
                    ## Quick Start Guide

                    ### For FLUX-SRPO-bf16.safetensors:

                    1. Go to **"FLUX / Diffusion Models"** tab
                    2. Click 📂 and select your FLUX-SRPO-bf16.safetensors file
                    3. Keep "Auto Detect" for model type
                    4. Choose **"Balanced"** preset (recommended)
                    5. Click **"START QUANTIZATION"**
                    6. Wait for completion (duration varies by GPU)
                    7. Get your quantized model (~75% smaller!)

                    ---

                    ## Diffusion Models Tab

                    ### Supported Models:
                    - ✅ FLUX Dev & Schnell
                    - ✅ Stable Diffusion 3 & 3.5
                    - ✅ SDXL & SDXL Turbo

                    ### Presets Explained:

                    - **🚀 Best Quality**
                      - Uses MAX algorithm for maximum quality
                      - 512 calibration samples
                      - Perfect for production deployment
                      - ~75% size reduction

                    - **⚡ Balanced** ⭐ Recommended
                      - Best balance of quality and speed
                      - 256 calibration samples
                      - Great for most use cases
                      - ~75% size reduction

                    - **💨 Fast**
                      - Quick quantization
                      - 128 calibration samples
                      - Good for testing
                      - ~75% size reduction

                    - **🏃 Ultra Fast Test**
                      - FP8 format (faster)
                      - 64 calibration samples
                      - Quick quality check
                      - ~50% size reduction (FP8 vs NVFP4's 75%)

                    ### Expected Results:

                    **FLUX Dev:**
                    - Original: ~24 GB
                    - NVFP4: ~6-8 GB
                    - Quality Loss: <2%

                    **SDXL Base:**
                    - Original: ~7 GB
                    - NVFP4: ~2 GB
                    - Quality Loss: <2%

                    ---

                    ## LLM Tab

                    ### Supported Models:
                    - ✅ Llama 2/3
                    - ✅ Mistral
                    - ✅ Mixtral
                    - ✅ Qwen
                    - ✅ Phi
                    - ✅ Gemma
                    - ✅ Any HuggingFace causal LM

                    ### Recommended Settings:

                    - **Format:** NVFP4 (Default)
                    - **Calibration:** 512 samples from cnn_dailymail
                    - **Batch Size:** 1 (safest for memory)
                    - **KV Cache:** None (or FP8 for lower inference memory)

                    ### Example Usage:

                    **Model Path:** `meta-llama/Llama-2-7b-hf`
                    **Output Dir:** `./quantized_models/llama2-7b-nvfp4`

                    Results in ~4x smaller model with <2% perplexity increase.

                    ---

                    ## Troubleshooting

                    ### "Virtual environment not found"
                    - Ensure venv exists in the project root directory
                    - Run the Model-Optimizer installation script first

                    ### "CUDA out of memory"
                    - **For Diffusion:** Try "Ultra Fast Test" preset (uses FP8)
                    - **For LLM:** Reduce batch size to 1, calibration samples to 256
                    - Close other GPU applications
                    - Check GPU memory with Task Manager/nvidia-smi

                    ### Quantization takes time
                    - This is normal! NVFP4 quantization requires calibration
                    - Duration varies based on GPU, model size, and preset
                    - Use "Fast" or "Ultra Fast" presets for quicker results

                    ### Poor quality after quantization
                    - **For Diffusion:** Try "Best Quality" preset
                    - **For LLM:** Increase calibration samples to 1024
                    - Ensure model type auto-detection is correct
                    - Try different calibration dataset

                    ### Model won't load
                    - **HuggingFace models:** Check internet connection
                    - **Local models:** Verify file paths are correct
                    - Check logs for specific error messages

                    ---

                    ## Output Formats

                    ### Diffusion Models:
                    - Saves as PyTorch checkpoint (.pt)
                    - Contains quantized weights and config
                    - Can convert to .safetensors for ComfyUI:
                      ```python
                      import torch
                      from safetensors.torch import save_file

                      checkpoint = torch.load("model_NVFP4.pt")
                      save_file(checkpoint, "model_NVFP4.safetensors")
                      ```

                    ### LLMs:
                    - Saves as HuggingFace format
                    - Load with: `AutoModelForCausalLM.from_pretrained(output_dir)`
                    - Compatible with transformers library
                    - Can export to TensorRT-LLM for deployment

                    ---

                    ## Tips for Best Results

                    1. **Use Representative Data**
                       - Calibration data should match your use case
                       - For FLUX: default prompts work well
                       - For LLMs: choose dataset similar to your task

                    2. **Monitor GPU Usage**
                       - Keep GPU memory <90% during quantization
                       - Reduce batch size if getting close to limits

                    3. **Test Before Deploying**
                       - Always validate quality with test generations
                       - Compare side-by-side with original model
                       - Check for artifacts or quality degradation

                    4. **Start Conservative**
                       - Use "Balanced" preset first
                       - Upgrade to "Best Quality" if needed
                       - Don't skip quality testing!

                    ---

                    ## System Requirements

                    ### Minimum:
                    - NVIDIA GPU with 24 GB VRAM
                    - CUDA 11.8 or later
                    - 32 GB system RAM

                    ### Recommended:
                    - NVIDIA RTX 4090 / A6000 / H100
                    - 32+ GB VRAM
                    - 64 GB system RAM
                    - NVMe SSD for fast I/O

                    ---

                    ## Performance Expectations

                    - Duration varies significantly based on GPU, model size, and preset
                    - Higher-end GPUs (RTX 4090, A6000, H100) will be faster
                    - More calibration samples = longer processing time
                    - Lower-end GPUs may need to reduce calibration samples
                    - Consider "Fast" or "Ultra Fast" presets for quicker results

                    ---

                    ## Additional Resources

                    - [NVIDIA ModelOpt Documentation](https://github.com/NVIDIA/TensorRT-Model-Optimizer)
                    - [Diffusers Examples](https://github.com/NVIDIA/TensorRT-Model-Optimizer/tree/main/examples/diffusers)
                    - [LLM PTQ Examples](https://github.com/NVIDIA/TensorRT-Model-Optimizer/tree/main/examples/llm_ptq)

                    ---

                    **Need Help?** Check the logs in the quantization output for detailed error messages.
                    """
                )

        # Footer
        gr.Markdown(
            """
            ---
            **NVFP4 Model Quantizer** | Powered by NVIDIA ModelOpt | Version 1.0
            """,
        )

    return app


def main():
    """Main entry point."""

    # Parse arguments
    parser = argparse.ArgumentParser(description="NVFP4 Model Quantizer")
    parser.add_argument("--listen", type=str, default="127.0.0.1", help="IP to listen on")
    parser.add_argument("--port", type=int, default=7860, help="Port to run on")
    parser.add_argument("--share", action="store_true", help="Share via Gradio link")
    parser.add_argument("--headless", action="store_true", help="Headless mode")
    parser.add_argument("--debug", action="store_true", help="Debug mode")

    args = parser.parse_args()

    # Create UI
    app = create_ui(headless=args.headless)

    # Extract theme and css from the create_ui function
    theme = gr.themes.Soft(
        primary_hue="purple",
        secondary_hue="blue",
        neutral_hue="slate",
    )

    css = """
    .primary-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        font-size: 18px !important;
        font-weight: bold !important;
        padding: 15px 30px !important;
        border: none !important;
        color: white !important;
    }
    .status-ok {
        color: #28a745;
        font-weight: bold;
        font-size: 16px;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
        font-size: 16px;
    }
    .tab-nav button {
        font-size: 16px !important;
        font-weight: 600 !important;
        padding: 12px 24px !important;
    }
    """

    print("\n" + "="*60)
    print("  NVFP4 MODEL QUANTIZER")
    print("="*60)
    print(f"  Opening browser at http://{args.listen}:{args.port}")
    print("  Press Ctrl+C to stop")
    print("="*60 + "\n")

    app.launch(
        share=args.share,
        inbrowser=not args.headless,
        show_error=True,
        debug=args.debug,
        theme=theme,
        css=css,
    )


if __name__ == "__main__":
    main()
