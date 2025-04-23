# PyTorch Multi-GPU Demo

This demo shows how to use `torch.nn.DataParallel` to train any PyTorch model across multiple GPUs (or fall back to single-GPU/CPU).

### Prerequisites

- Python 3.7+
- PyTorch 1.8+
- At least one CUDA-compatible GPU (multiple GPUs optional)

### Setup

1. Create a virtualenv (optional) and install:
   ```bash
   pip install -r requirements.txt
