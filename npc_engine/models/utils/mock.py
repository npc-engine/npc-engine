"""Utility script to mock models for testing."""
from typing import Dict
from torch.onnx import export
import torch
import onnxruntime as rt
import numpy as np
from npc_engine.models.utils import DTYPE_MAP


def create_stub_onnx_model(onnx_model_path: str, output_path: str):
    """Create stub onnx model for tests with correct input and output shapes and names.

    Args:
        onnx_model_path: Path to the onnx model.
    """
    model = rt.InferenceSession(onnx_model_path)

    named_shapes = _get_named_shapes(model.get_inputs())
    dynamic_axes = _get_dynamic_axes(model.get_inputs(), model.get_outputs())
    input_names = [inp.name for inp in model.get_inputs()]
    inputs = {
        inp.name: torch.Tensor(
            np.random.randn(*[named_shapes.get(dim, dim) for dim in inp.shape]).astype(
                DTYPE_MAP[inp.type]
            )
        )
        for inp in model.get_inputs()
    }
    outputs = {
        out.name: torch.Tensor(
            np.random.randn(*[named_shapes.get(dim, dim) for dim in out.shape]).astype(
                DTYPE_MAP[out.type]
            )
        )
        for out in model.get_outputs()
    }
    output_names = [out.name for out in model.get_outputs()]

    class DummyModule(torch.nn.Module):
        def forward(self, *inputs):
            return outputs

    model = DummyModule()
    export(
        model,
        inputs,
        output_path,
        output_names,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )


def _get_named_shapes(inputs) -> Dict[str, int]:
    """Get named shapes from onnx model inputs.

    Args:
        inputs: List of onnx model inputs.
    """
    named_shapes = {}
    for inp in inputs:
        for dim in inp.shape:
            if isinstance(dim, str):
                named_shapes[dim] = 1
    return named_shapes


def _get_dynamic_axes(inputs, outputs) -> Dict[str, Dict[str, int]]:
    """Get dynamic axes from onnx model inputs and outputs.

    Args:
        inputs: List of onnx model inputs.
        outputs: List of onnx model outputs.
    """
    dynamic_axes = {}
    for inp in inputs:
        for i, dim in enumerate(inp.shape):
            if isinstance(dim, str):
                dynamic_axes[inp.name] = {**dynamic_axes.get(inp.name, {}), dim: i}
    for out in outputs:
        for i, dim in enumerate(out.shape):
            if isinstance(dim, str):
                dynamic_axes[out.name] = {**dynamic_axes.get(out.name, {}), dim: i}
    return dynamic_axes
