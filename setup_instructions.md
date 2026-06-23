# Environment Setup Instructions

This workshop runs entirely in your browser using Google Colab. **No local installation is required.** However, if you prefer to run the notebook locally, Part 2 provides instructions for setting up a local environment using Miniforge.

---

## Part 1 — Google Colab (Recommended)

Google Colab provides free access to a cloud-based Python environment with limited GPU support. You only need a Google account. The free tier includes a daily GPU quota — please follow the instructions below carefully to avoid running out of compute time before the workshop.

### Step 1: Sign in to Google Colab

1. Go to [https://colab.research.google.com](https://colab.research.google.com)
2. Sign in with your Google account
3. Click **"New notebook"** to verify that Colab opens correctly

### Step 2: Enable GPU

> ⚠️ **Do this before running any cells.** Changing the runtime type after cells have already been executed will restart the session and clear all variables.

1. In the menu bar, go to **Runtime → Change runtime type**
2. Under **Hardware accelerator**, select **T4 GPU**
3. Click **Save**

### Step 3: Test your environment

Copy and run the following code in a Colab cell to verify everything is working:

```python
# Test Colab environment
import sys
print(f"Python version: {sys.version}")

# Install required packages
!pip install -q mne wfdb scikit-learn

import torch
import mne
import wfdb
import sklearn
import numpy as np
import matplotlib

print(f"PyTorch   : {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MNE       : {mne.__version__}")
print(f"wfdb      : {wfdb.__version__}")
print(f"scikit-learn: {sklearn.__version__}")
print(f"NumPy     : {np.__version__}")
print(f"Matplotlib: {matplotlib.__version__}")
print("\nAll packages installed successfully!")
```

### Expected output

```
Python version: 3.x.x
PyTorch   : 2.x.x
CUDA available: True
MNE       : 1.x.x
wfdb      : 4.x.x
scikit-learn: 1.x.x
NumPy     : 2.x.x
Matplotlib: 3.x.x

All packages installed successfully!
```

> **Note:** If `CUDA available` shows `False`, please go back to Step 2 and confirm that T4 GPU is selected.

> **Note on dependency warnings:** After running `pip install`, you may see messages such as `ERROR: pip's dependency resolver does not currently take into account all the packages...` followed by conflicts involving `google-colab`, `gradio`, `moviepy`, or `ipython`. These are warnings about Colab's own pre-installed packages and **will not affect the workshop in any way.** As long as the package versions print correctly and you see "All packages installed successfully!", you are good to go.

> ⚠️ **Important — GPU quota:** Google Colab's free tier has a limited GPU quota per day. Once you have confirmed that the test cell runs successfully, **please disconnect and delete the runtime immediately** (Runtime → Disconnect and delete runtime). This will release the GPU and preserve your quota for use during the workshop. If your quota runs out mid-session, the notebook will fall back to CPU, which will be significantly slower.  If needed, additional compute units can be purchased through [Google Colab Pro](https://colab.research.google.com/signup).

---

## Part 2 — Local Environment with Miniforge (Optional)

If you prefer to run the notebook on your own machine, follow the instructions below. Miniforge is a lightweight conda-compatible package manager that works on all platforms.

---

### Step 1: Install Miniforge

#### macOS

```bash
# Download installer
curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-$(uname -m).sh

# Run installer
bash Miniforge3-MacOSX-$(uname -m).sh
```
Follow the prompts and allow the installer to initialize conda. Then restart your terminal.

> **Note:** The `$(uname -m)` part automatically selects the correct version for your Mac (Intel or Apple Silicon).

#### Windows

1. Download the installer from:  
   [https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe)
2. Run the `.exe` installer
3. When prompted, select **"Add Miniforge3 to my PATH environment variable"**
4. After installation, open **Miniforge Prompt** from the Start menu (not the standard Command Prompt)

#### Linux

```bash
# Download installer
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh

# Run installer
bash Miniforge3-Linux-x86_64.sh
```
Follow the prompts and allow the installer to initialize conda. Then restart your terminal.

---

### Step 2: Create the workshop environment

Run the following commands in your terminal (macOS/Linux) or Miniforge Prompt (Windows):

```bash
# Create a new environment named 'rbd_workshop' with Python 3.11
conda create -n ifcn_workshop python=3.11 -y

# Activate the environment
conda activate ifcn_workshop
```

Then install PyTorch according to your hardware:

#### Option A — NVIDIA GPU (CUDA)
For laptops and desktops with an NVIDIA GPU. Choose the command that matches your GPU generation:

**Most users — GTX 10xx / RTX 20xx / 30xx / 40xx series (Maxwell through Ada Lovelace):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

**Newer high-end GPUs — Hopper (H100) or Blackwell (B100/B200/RTX 50xx):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
```

> **Not sure which GPU you have?** Run `nvidia-smi` in your terminal. The output shows your GPU model and the maximum CUDA version your driver supports. If the CUDA version shown is 12.x, use `cu126`. If it is 13.x, use `cu130`.  
> PyTorch is backward compatible — a `cu126` build will run correctly on a machine with a newer driver, just without the latest optimizations.

#### Option B — AMD GPU (ROCm) — Linux only
For desktops or workstations with an AMD GPU (e.g. RX 6700/6800/7800/7900 XT, Radeon Pro W6800, Instinct MI series). ROCm is currently only supported on Linux.
```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm7.2
```

> **Note:** ROCm support on Windows is limited and not recommended for this workshop. Windows users with AMD GPUs should use the CPU option below.

#### Option C — CPU only
For all other cases, including:
- MacBooks with Intel processors
- MacBooks with Apple Silicon (M1/M2/M3/M4) — PyTorch uses the MPS backend instead of CUDA
- Windows or Linux laptops without a dedicated GPU
- Virtual machines

```bash
pip install torch
```

> **Apple Silicon note:** PyTorch on Apple Silicon (M1/M2/M3/M4) uses the built-in MPS (Metal Performance Shaders) backend for GPU acceleration automatically. No special installation is needed — the standard CPU install above will enable MPS if available. You can verify with `torch.backends.mps.is_available()`.

Finally, install the remaining packages:

```bash
pip install mne wfdb scikit-learn jupyter notebook numpy scipy matplotlib
```

> **Note for Windows users:** If you encounter a permissions error during `pip install`, try running Miniforge Prompt as Administrator.

---

### Step 3: Launch Jupyter Notebook

```bash
# Make sure the environment is activated
conda activate ifcn_workshop

# Launch Jupyter
jupyter notebook
```

Your browser should open automatically at `http://localhost:8888`. If it does not, copy the URL printed in the terminal and paste it into your browser.

---

### Step 4: Test your environment

In Jupyter, create a new notebook and run the following cell:

```python
import sys
print(f"Python version: {sys.version}")

import torch
import mne
import wfdb
import sklearn
import numpy as np
import matplotlib

print(f"PyTorch      : {torch.__version__}")
print(f"CUDA available : {torch.cuda.is_available()}")
print(f"ROCm available : {torch.version.hip is not None}")
print(f"MPS available  : {torch.backends.mps.is_available()}")
print(f"MNE          : {mne.__version__}")
print(f"wfdb         : {wfdb.__version__}")
print(f"scikit-learn : {sklearn.__version__}")
print(f"NumPy        : {np.__version__}")
print(f"Matplotlib   : {matplotlib.__version__}")
print("\nAll packages installed successfully!")
```

Expected output depending on your hardware:

| Hardware | CUDA | ROCm | MPS |
|---|---|---|---|
| NVIDIA GPU | `True` | `False` | `False` |
| AMD GPU (Linux) | `False` | `True` | `False` |
| Apple Silicon (M1/M2/M3/M4) | `False` | `False` | `True` |
| CPU only / Intel Mac | `False` | `False` | `False` |

> All configurations will work for this workshop. GPU acceleration (CUDA/ROCm/MPS) will make model training faster, but the models are small enough to run on CPU within a reasonable time.

---

## Troubleshooting

**Colab disconnects during the session**  
Colab sessions time out after periods of inactivity. If this happens, go to **Runtime → Run all** to re-execute the notebook from the top.

**`conda` command not found after installation**  
Close and reopen your terminal (macOS/Linux) or open a new Miniforge Prompt (Windows).

**Package version conflicts**  
If you encounter conflicts during `pip install`, try installing packages one at a time to identify the issue, or create a fresh environment and reinstall.

**Cannot import torch after installation**  
Make sure the `ifcn_workshop` environment is activated (`conda activate ifcn_workshop`) before launching Jupyter.

---

## Getting Help

If you run into any issues not covered above, please report them via one of the following:

**Preferred — Submit a GitHub Issue**  
Reporting issues on GitHub is preferred because other participants may have the same problem and can benefit from the solution. To submit an issue:

1. Go to [https://github.com/Po-yuLin/AOC-IFCN_workshop/issues](https://github.com/Po-yuLin/AOC-IFCN_workshop/issues)
2. Click **"New issue"**
3. Describe your problem — include your operating system, the error message (copy and paste the full text), and which step you were on
4. Click **"Submit new issue"**

**Alternative — Email**  
If you are not familiar with GitHub, you are also welcome to email the workshop organizer directly:  
📧 p88124019@gs.ncku.edu.tw

Please include your operating system, a description of the problem, and the full error message if applicable.
