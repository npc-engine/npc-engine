"""Utility methods for model handling."""
import numpy as np


# Map from onnx type string to numpy dtype
DTYPE_MAP = {
    "tensor(int64)": np.int64,
    "tensor(float)": np.float32,
    "tensor(double)": np.float64,
    "tensor(int32)": np.int32,
}
