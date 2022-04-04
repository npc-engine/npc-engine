"""Utility script to mock models for testing."""
import sclblonnx as so
from onnx import helper as xhelp
import sclblonnx._globals as glob
import numpy as np
import click


def create_stub_onnx_model(onnx_model_path: str, output_path: str):
    """Create stub onnx model for tests with correct input and output shapes and names.

    Args:
        onnx_model_path: Path to the onnx model.
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
        build_output_shape_tensor_(
            mock_graph, output, dynamic_shape_map, f"dynamic_shape_{output.name}",
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
    graph, output, dynamic_shape_map, shape_name="dynamic_shape"
):
    """Build output shape tensor for dynamic shape models.

    Args:
        graph: Graph to add the output shape tensor to.
        output_name: Name of the output.
        dynamic_shape_map: Map of input names to their dynamic shape indices.
        shape_name: Name of the output shape tensor.
    """
    dimensions_retrieved = []
    for i, dim in enumerate(output.type.tensor_type.shape.dim):
        if dim.dim_param != "":
            if " + " in dim.dim_param:
                dim1, dim2 = dim.dim_param.split(" + ")
                if dim1 in dynamic_shape_map:
                    node1 = so.node(
                        "Shape",
                        inputs=[dynamic_shape_map[dim1][0]],
                        outputs=[f"{shape_name}_{i}_1"],
                        start=dynamic_shape_map[dim1][1],
                        end=dynamic_shape_map[dim1][1] + 1,
                    )
                    so.add_node(graph, node1)
                else:
                    so.add_constant(
                        graph,
                        f"{shape_name}_{i}_1",
                        np.array([1], dtype=np.int64),
                        data_type="INT64",
                    )
                if dim2 in dynamic_shape_map:
                    node2 = so.node(
                        "Shape",
                        inputs=[dynamic_shape_map[dim2][0]],
                        outputs=[f"{shape_name}_{i}_2"],
                        start=dynamic_shape_map[dim2][1],
                        end=dynamic_shape_map[dim2][1] + 1,
                    )
                    so.add_node(graph, node2)
                else:
                    so.add_constant(
                        graph,
                        f"{shape_name}_{i}_2",
                        np.array([1], dtype=np.int64),
                        data_type="INT64",
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
                node = so.node(
                    "Shape",
                    inputs=[dynamic_shape_map[dim.dim_param][0]],
                    outputs=[f"{shape_name}_{i}"],
                    start=dynamic_shape_map[dim.dim_param][1],
                    end=dynamic_shape_map[dim.dim_param][1] + 1,
                )
                so.add_node(graph, node)
        else:
            so.add_constant(
                graph,
                f"{shape_name}_{i}",
                np.array([dim.dim_value], dtype=np.int64),
                data_type="INT64",
            )
        dimensions_retrieved.append(f"{shape_name}_{i}")
    node = so.node(
        "Concat", inputs=dimensions_retrieved, outputs=[f"{shape_name}"], axis=0,
    )
    so.add_node(graph, node)


@click.command()
@click.option("-m", "--onnx-model-path", required=True, type=str)
@click.option("-o", "--output-path", required=True, type=str)
def main(onnx_model_path: str, output_path: str):
    """Create stub onnx model for tests with correct input and output shapes and names."""
    create_stub_onnx_model(onnx_model_path, output_path)


if __name__ == "__main__":
    main()
