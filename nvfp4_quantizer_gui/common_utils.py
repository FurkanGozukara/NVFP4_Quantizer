"""
Common utilities for NVFP4 Quantizer GUI
"""

import os
import sys
from pathlib import Path
from typing import Tuple

try:
    from tkinter import filedialog, Tk
except ImportError:
    filedialog = None
    Tk = None

# Path constants
MODEL_OPTIMIZER_DIR = Path(__file__).parent.parent / "Model-Optimizer"
VENV_DIR = MODEL_OPTIMIZER_DIR / "venv"
VENV_PYTHON = VENV_DIR / "Scripts" / "python.exe" if sys.platform == "win32" else VENV_DIR / "bin" / "python"

# Scripts
DIFFUSION_SCRIPT = MODEL_OPTIMIZER_DIR / "examples" / "diffusers" / "quantization" / "quantize.py"
LLM_SCRIPT = MODEL_OPTIMIZER_DIR / "examples" / "llm_ptq" / "hf_ptq.py"


def check_venv_exists() -> Tuple[bool, str]:
    """Check if Model-Optimizer venv exists and is valid."""
    if not VENV_DIR.exists():
        return False, f"❌ Virtual environment not found at {VENV_DIR}"

    if not VENV_PYTHON.exists():
        return False, f"❌ Python executable not found at {VENV_PYTHON}"

    return True, f"✅ Environment Ready"


def is_display_available() -> bool:
    """Check if display is available for file dialogs."""
    env_exclusion = ["COLAB_GPU", "RUNPOD_POD_ID"]
    if any(var in os.environ for var in env_exclusion):
        return False

    if sys.platform == "darwin":
        return False

    if sys.platform.startswith("linux") or sys.platform == "posix":
        if "DISPLAY" not in os.environ and "WAYLAND_DISPLAY" not in os.environ:
            return False

    return True


def get_file_path(file_path: str = "", extensions: str = "*.*", title: str = "Select File") -> str:
    """
    Open file dialog to select a file.

    Args:
        file_path: Initial file path
        extensions: File extension filter (e.g., "*.safetensors")
        title: Dialog title

    Returns:
        Selected file path or empty string
    """
    if filedialog is None or Tk is None or not is_display_available():
        return file_path

    try:
        root = Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        # Parse extensions
        if extensions == "*.*":
            filetypes = [("All files", "*.*")]
        elif ";" in extensions:
            # Multiple extensions like "*.safetensors;*.pt"
            exts = extensions.split(";")
            filetypes = [("Model files", extensions)]
            for ext in exts:
                name = ext.replace("*.", "").upper()
                filetypes.append((f"{name} files", ext))
            filetypes.append(("All files", "*.*"))
        else:
            name = extensions.replace("*.", "").upper()
            filetypes = [(f"{name} files", extensions), ("All files", "*.*")]

        selected_path = filedialog.askopenfilename(
            title=title,
            filetypes=filetypes,
            initialfile=file_path,
        )

        root.destroy()
        return selected_path if selected_path else file_path

    except Exception as e:
        print(f"Error opening file dialog: {e}")
        return file_path


def get_folder_path(folder_path: str = "", title: str = "Select Folder") -> str:
    """
    Open folder dialog to select a directory.

    Args:
        folder_path: Initial folder path
        title: Dialog title

    Returns:
        Selected folder path or empty string
    """
    if filedialog is None or Tk is None or not is_display_available():
        return folder_path

    try:
        root = Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        selected_path = filedialog.askdirectory(
            title=title,
            initialdir=folder_path,
        )

        root.destroy()
        return selected_path if selected_path else folder_path

    except Exception as e:
        print(f"Error opening folder dialog: {e}")
        return folder_path


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def get_model_size(model_path: str) -> str:
    """Get model file size in human-readable format."""
    try:
        if os.path.exists(model_path):
            size = os.path.getsize(model_path)
            return format_file_size(size)
    except:
        pass
    return "Unknown"
