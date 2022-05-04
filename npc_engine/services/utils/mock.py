"""Utility script to mock models for testing."""
from typing import List
import sclblonnx as so
from onnx import helper as xhelp
import sclblonnx._globals as glob
import numpy as np
import click


def create_stub_onnx_model(
    onnx_model_path: str, output_path: str, dynamic_shape_values: dict
):
    """Create stub onnx model for tests with correct input and output shapes and names.

    Args:
        onnx_model_path: Path to the onnx model.
        output_path: Path to the output mock model.
        dynamic_shape_values: Map to specify dynamic dimension values.
    """
    onnx_model = so.graph_from_file(onnx_model_path)
    inputs = onnx_model.input
    outputs = onnx_model.output

    mock_graph = so.empty_graph()
    inverse_data_dict = {value: key for key, value in glob.DATA_TYPES.items()}

    dynamic_shape_map = {}

    for input_ in inputs:
        so.add_input(
            mock_graph,
            name=input_.name,
            dimensions=[
                dim.dim_param if dim.dim_param != "" else dim.dim_value
                for dim in input_.type.tensor_type.shape.dim
            ],
            data_type=inverse_data_dict[input_.type.tensor_type.elem_type],
        )
        for i, dim in enumerate(input_.type.tensor_type.shape.dim):
            if dim.dim_param != "":
                dynamic_shape_map[dim.dim_param] = (input_.name, i)

    for output in outputs:
        so.add_output(
            mock_graph,
            name=output.name,
            dimensions=[
                dim.dim_param if dim.dim_param != "" else dim.dim_value
                for dim in output.type.tensor_type.shape.dim
            ],
            data_type=inverse_data_dict[output.type.tensor_type.elem_type],
        )
        input_names = [inp.name for inp in inputs]
        if output.name not in input_names:
            build_output_shape_tensor_(
                mock_graph,
                output,
                dynamic_shape_map,
                f"dynamic_shape_{output.name}",
                override=dynamic_shape_values,
            )
            node = so.node(
                "ConstantOfShape",
                inputs=[f"dynamic_shape_{output.name}"],
                outputs=[output.name],
                value=xhelp.make_tensor(
                    name=f"dynamic_shape_{output.name}_value",
                    data_type=output.type.tensor_type.elem_type,
                    dims=[1],
                    vals=[0],
                ),
                name=f"ConstantOfShape_{output.name}",
            )
            so.add_node(mock_graph, node)
    so.graph_to_file(mock_graph, output_path, onnx_opset_version=15)


def build_output_shape_tensor_(
    graph, output, dynamic_shape_map, shape_name="dynamic_shape", override={}
):
    """Build output shape tensor for dynamic shape models.

    Args:
        graph: Graph to add the output shape tensor to.
        output: Model output.
        dynamic_shape_map: Map of input names to their dynamic shape indices.
        shape_name: Name of the output shape tensor.
    """
    dimensions_retrieved = []
    for i, dim in enumerate(output.type.tensor_type.shape.dim):
        if " + " in dim.dim_param:
            dim1, dim2 = dim.dim_param.split(" + ")
            create_dim_variable_(
                graph,
                shape_name,
                dim1,
                override.get(dim1, dim.dim_value),
                i,
                dynamic_shape_map,
                postfix="1",
            )
            create_dim_variable_(
                graph,
                shape_name,
                dim2,
                override.get(dim2, dim.dim_value),
                i,
                dynamic_shape_map,
                postfix="2",
            )
            so.add_node(
                graph,
                so.node(
                    "Add",
                    inputs=[f"{shape_name}_{i}_1", f"{shape_name}_{i}_2"],
                    outputs=[f"{shape_name}_{i}"],
                ),
            )
        else:
            create_dim_variable_(
                graph,
                shape_name,
                dim.dim_param,
                override.get(dim.dim_param, dim.dim_value),
                i,
                dynamic_shape_map,
            )
        dimensions_retrieved.append(f"{shape_name}_{i}")
    node = so.node(
        "Concat",
        inputs=dimensions_retrieved,
        outputs=[f"{shape_name}"],
        axis=0,
    )
    so.add_node(graph, node)


def create_dim_variable_(
    graph, shape_name, dim_param, dim_value, dim_id, dynamic_shape_map, postfix=None
):
    """Create a dimension variable for a dynamic shape model.

    Args:
        graph: Graph to add the dimension variable to.
        shape_name: Name of the output shape tensor.
        dim_param: Dimension parameter name.
        dim_value: Dimension value.
        dim_id: Index of the dimension variable.
        dynamic_shape_map: Map of dynamic axes names to their inputs indices.
    """
    if dim_param != "" and dim_param in dynamic_shape_map:
        node1 = so.node(
            "Shape",
            inputs=[dynamic_shape_map[dim_param][0]],
            outputs=[
                f"{shape_name}_{dim_id}"
                if postfix is None
                else f"{shape_name}_{dim_id}_{postfix}"
            ],
            start=dynamic_shape_map[dim_param][1],
            end=dynamic_shape_map[dim_param][1] + 1,
        )
        so.add_node(graph, node1)
    else:
        so.add_constant(
            graph,
            f"{shape_name}_{dim_id}"
            if postfix is None
            else f"{shape_name}_{dim_id}_{postfix}",
            np.array(
                [dim_value if dim_value != 0 else 1],
                dtype=np.int64,
            ),
            data_type="INT64",
        )


@click.command()
@click.option(
    "-d",
    "--dynamic-dim",
    type=str,
    multiple=True,
    help="Specify dynamic dimension in format `<dim name>:<dim value>`.",
)
@click.option("-m", "--onnx-model-path", required=True, type=str)
@click.option("-o", "--output-path", required=True, type=str)
def main(onnx_model_path: str, output_path: str, dynamic_dim: List[str]):
    """Create stub onnx model for tests with correct input and output shapes and names."""
    dynamic_dims_map = {}
    for dim in dynamic_dim:
        dim_name, dim_value = dim.split(":")
        dynamic_dims_map[dim_name] = int(dim_value)
    create_stub_onnx_model(onnx_model_path, output_path, dynamic_dims_map)


if __name__ == "__main__":
    main()
