"""Shared test fixtures for MLASIC tests.

Builds synthetic ONNX models programmatically — no network dependency.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper


def _make_dense_layer(
    prefix: str,
    input_name: str,
    in_dim: int,
    out_dim: int,
    has_bn: bool = True,
    has_relu: bool = True,
    rng: np.random.RandomState | None = None,
) -> tuple[list, list, list, str]:
    """Build one dense layer: MatMul -> Add [-> BN] [-> ReLU].

    Returns (nodes, initializers, value_infos, output_name).
    """
    if rng is None:
        rng = np.random.RandomState(42)

    nodes = []
    initializers = []
    value_infos = []

    # Weight: [in_dim, out_dim]
    w_name = f"{prefix}_weight"
    w_data = rng.randn(in_dim, out_dim).astype(np.float32) * 0.1
    initializers.append(numpy_helper.from_array(w_data, name=w_name))

    # MatMul
    mm_out = f"{prefix}_matmul_out"
    mm_node = helper.make_node("MatMul", [input_name, w_name], [mm_out], name=f"{prefix}_MatMul")
    nodes.append(mm_node)
    value_infos.append(helper.make_tensor_value_info(mm_out, TensorProto.FLOAT, [1, out_dim]))

    # Bias: [out_dim]
    b_name = f"{prefix}_bias"
    b_data = rng.randn(out_dim).astype(np.float32) * 0.01
    initializers.append(numpy_helper.from_array(b_data, name=b_name))

    # Add
    add_out = f"{prefix}_add_out"
    nodes.append(helper.make_node("Add", [mm_out, b_name], [add_out], name=f"{prefix}_Add"))
    value_infos.append(helper.make_tensor_value_info(add_out, TensorProto.FLOAT, [1, out_dim]))

    current_out = add_out

    # BatchNorm (optional)
    if has_bn:
        bn_scale = np.ones(out_dim, dtype=np.float32)
        bn_bias = np.zeros(out_dim, dtype=np.float32)
        bn_mean = np.zeros(out_dim, dtype=np.float32)
        bn_var = np.ones(out_dim, dtype=np.float32)

        scale_name = f"{prefix}_bn_scale"
        bias_name = f"{prefix}_bn_bias"
        mean_name = f"{prefix}_bn_mean"
        var_name = f"{prefix}_bn_var"

        initializers.extend(
            [
                numpy_helper.from_array(bn_scale, name=scale_name),
                numpy_helper.from_array(bn_bias, name=bias_name),
                numpy_helper.from_array(bn_mean, name=mean_name),
                numpy_helper.from_array(bn_var, name=var_name),
            ]
        )

        bn_out = f"{prefix}_bn_out"
        nodes.append(
            helper.make_node(
                "BatchNormalization",
                [current_out, scale_name, bias_name, mean_name, var_name],
                [bn_out],
                name=f"{prefix}_BN",
                epsilon=1e-5,
            )
        )
        value_infos.append(helper.make_tensor_value_info(bn_out, TensorProto.FLOAT, [1, out_dim]))
        current_out = bn_out

    # ReLU (optional)
    if has_relu:
        relu_out = f"{prefix}_relu_out"
        nodes.append(helper.make_node("Relu", [current_out], [relu_out], name=f"{prefix}_Relu"))
        value_infos.append(helper.make_tensor_value_info(relu_out, TensorProto.FLOAT, [1, out_dim]))
        current_out = relu_out

    return nodes, initializers, value_infos, current_out


def build_ad_model() -> onnx.ModelProto:
    """Build a synthetic AD model: 640->128->128->128->640 MLP with BN+ReLU.

    Mimics the MLPerf Tiny Anomaly Detection model structure.
    """
    rng = np.random.RandomState(42)
    all_nodes = []
    all_inits = []
    all_vis = []

    # Layer dimensions: input_dim -> output_dim
    layers = [
        (640, 128, True, True),  # Layer 0: 640->128, BN+ReLU
        (128, 128, True, True),  # Layer 1: 128->128, BN+ReLU
        (128, 128, True, True),  # Layer 2: 128->128, BN+ReLU
        (128, 640, False, False),  # Layer 3: 128->640, no BN, no ReLU (output)
    ]

    input_name = "input"
    current = input_name

    for i, (in_dim, out_dim, has_bn, has_relu) in enumerate(layers):
        nodes, inits, vis, current = _make_dense_layer(
            prefix=f"layer{i}",
            input_name=current,
            in_dim=in_dim,
            out_dim=out_dim,
            has_bn=has_bn,
            has_relu=has_relu,
            rng=rng,
        )
        all_nodes.extend(nodes)
        all_inits.extend(inits)
        all_vis.extend(vis)

    output_name = current

    # Graph input/output
    graph_input = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [1, 640])
    graph_output = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [1, 640])

    graph = helper.make_graph(
        nodes=all_nodes,
        name="ad_model",
        inputs=[graph_input],
        outputs=[graph_output],
        initializer=all_inits,
        value_info=all_vis,
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def build_unsupported_op_model() -> onnx.ModelProto:
    """Build a model with LpNormalization (unsupported) for rejection testing."""
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])

    node = helper.make_node(
        "LpNormalization",
        inputs=["input"],
        outputs=["output"],
        name="lpnorm_0",
        p=2,
    )

    graph = helper.make_graph([node], "unsupported_model", [inp], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


@pytest.fixture
def ad_model_path(tmp_path: Path) -> Path:
    """Save synthetic AD model to a temp file and return its path."""
    model = build_ad_model()
    path = tmp_path / "ad_model.onnx"
    onnx.save(model, str(path))
    return path


@pytest.fixture
def ad_model() -> onnx.ModelProto:
    """Return the synthetic AD model proto."""
    return build_ad_model()


@pytest.fixture
def unsupported_model_path(tmp_path: Path) -> Path:
    """Save unsupported-op model to a temp file and return its path."""
    model = build_unsupported_op_model()
    path = tmp_path / "unsupported_model.onnx"
    onnx.save(model, str(path))
    return path


@pytest.fixture
def test_vectors() -> np.ndarray:
    """100 random test vectors for the AD model, shape (100, 1, 640)."""
    rng = np.random.RandomState(42)
    return rng.randn(100, 1, 640).astype(np.float32)


def build_simple_cnn() -> onnx.ModelProto:
    """Build a synthetic CNN: Conv→BN→ReLU→MaxPool→Conv→BN→ReLU→GlobalAvgPool→Flatten→MatMul→Add.

    Input: (1, 1, 8, 8), Output: (1, 4)
    """
    rng = np.random.RandomState(123)
    nodes = []
    initializers = []
    value_infos = []

    # Conv1: 1→4 channels, 3x3 kernel, pad=1
    conv1_w = rng.randn(4, 1, 3, 3).astype(np.float32) * 0.1
    conv1_b = rng.randn(4).astype(np.float32) * 0.01
    initializers.append(numpy_helper.from_array(conv1_w, name="conv1_w"))
    initializers.append(numpy_helper.from_array(conv1_b, name="conv1_b"))
    nodes.append(
        helper.make_node(
            "Conv",
            ["input", "conv1_w", "conv1_b"],
            ["conv1_out"],
            name="Conv1",
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
        )
    )
    value_infos.append(helper.make_tensor_value_info("conv1_out", TensorProto.FLOAT, [1, 4, 8, 8]))

    # BN1
    bn1_scale = np.ones(4, dtype=np.float32)
    bn1_bias = np.zeros(4, dtype=np.float32)
    bn1_mean = np.zeros(4, dtype=np.float32)
    bn1_var = np.ones(4, dtype=np.float32)
    initializers.extend(
        [
            numpy_helper.from_array(bn1_scale, name="bn1_scale"),
            numpy_helper.from_array(bn1_bias, name="bn1_bias"),
            numpy_helper.from_array(bn1_mean, name="bn1_mean"),
            numpy_helper.from_array(bn1_var, name="bn1_var"),
        ]
    )
    nodes.append(
        helper.make_node(
            "BatchNormalization",
            ["conv1_out", "bn1_scale", "bn1_bias", "bn1_mean", "bn1_var"],
            ["bn1_out"],
            name="BN1",
            epsilon=1e-5,
        )
    )
    value_infos.append(helper.make_tensor_value_info("bn1_out", TensorProto.FLOAT, [1, 4, 8, 8]))

    # ReLU1
    nodes.append(helper.make_node("Relu", ["bn1_out"], ["relu1_out"], name="Relu1"))
    value_infos.append(helper.make_tensor_value_info("relu1_out", TensorProto.FLOAT, [1, 4, 8, 8]))

    # MaxPool: 2x2, stride 2
    nodes.append(
        helper.make_node(
            "MaxPool",
            ["relu1_out"],
            ["pool1_out"],
            name="MaxPool1",
            kernel_shape=[2, 2],
            strides=[2, 2],
        )
    )
    value_infos.append(helper.make_tensor_value_info("pool1_out", TensorProto.FLOAT, [1, 4, 4, 4]))

    # Conv2: 4→8 channels, 3x3 kernel, pad=1
    conv2_w = rng.randn(8, 4, 3, 3).astype(np.float32) * 0.1
    conv2_b = rng.randn(8).astype(np.float32) * 0.01
    initializers.append(numpy_helper.from_array(conv2_w, name="conv2_w"))
    initializers.append(numpy_helper.from_array(conv2_b, name="conv2_b"))
    nodes.append(
        helper.make_node(
            "Conv",
            ["pool1_out", "conv2_w", "conv2_b"],
            ["conv2_out"],
            name="Conv2",
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
        )
    )
    value_infos.append(helper.make_tensor_value_info("conv2_out", TensorProto.FLOAT, [1, 8, 4, 4]))

    # BN2
    bn2_scale = np.ones(8, dtype=np.float32)
    bn2_bias = np.zeros(8, dtype=np.float32)
    bn2_mean = np.zeros(8, dtype=np.float32)
    bn2_var = np.ones(8, dtype=np.float32)
    initializers.extend(
        [
            numpy_helper.from_array(bn2_scale, name="bn2_scale"),
            numpy_helper.from_array(bn2_bias, name="bn2_bias"),
            numpy_helper.from_array(bn2_mean, name="bn2_mean"),
            numpy_helper.from_array(bn2_var, name="bn2_var"),
        ]
    )
    nodes.append(
        helper.make_node(
            "BatchNormalization",
            ["conv2_out", "bn2_scale", "bn2_bias", "bn2_mean", "bn2_var"],
            ["bn2_out"],
            name="BN2",
            epsilon=1e-5,
        )
    )
    value_infos.append(helper.make_tensor_value_info("bn2_out", TensorProto.FLOAT, [1, 8, 4, 4]))

    # ReLU2
    nodes.append(helper.make_node("Relu", ["bn2_out"], ["relu2_out"], name="Relu2"))
    value_infos.append(helper.make_tensor_value_info("relu2_out", TensorProto.FLOAT, [1, 8, 4, 4]))

    # GlobalAveragePool
    nodes.append(
        helper.make_node("GlobalAveragePool", ["relu2_out"], ["gap_out"], name="GlobalAvgPool")
    )
    value_infos.append(helper.make_tensor_value_info("gap_out", TensorProto.FLOAT, [1, 8, 1, 1]))

    # Flatten
    nodes.append(helper.make_node("Flatten", ["gap_out"], ["flat_out"], name="Flatten1", axis=1))
    value_infos.append(helper.make_tensor_value_info("flat_out", TensorProto.FLOAT, [1, 8]))

    # Dense: MatMul + Add → 4 classes
    fc_w = rng.randn(8, 4).astype(np.float32) * 0.1
    fc_b = rng.randn(4).astype(np.float32) * 0.01
    initializers.append(numpy_helper.from_array(fc_w, name="fc_w"))
    initializers.append(numpy_helper.from_array(fc_b, name="fc_b"))
    nodes.append(helper.make_node("MatMul", ["flat_out", "fc_w"], ["mm_out"], name="FC_MatMul"))
    value_infos.append(helper.make_tensor_value_info("mm_out", TensorProto.FLOAT, [1, 4]))
    nodes.append(helper.make_node("Add", ["mm_out", "fc_b"], ["output"], name="FC_Add"))

    graph_input = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 8, 8])
    graph_output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])

    graph = helper.make_graph(
        nodes=nodes,
        name="simple_cnn",
        inputs=[graph_input],
        outputs=[graph_output],
        initializer=initializers,
        value_info=value_infos,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def build_simple_transformer_encoder() -> onnx.ModelProto:
    """Build a synthetic transformer encoder block.

    Gather(embedding) → Q/K/V projections → scaled dot-product attention →
    Softmax → context → LayerNorm → FFN (2 layers with GELU) → LayerNorm

    Input: indices (1, 4) int64, Output: (1, 4, 16)
    seq_len=4, d_model=16, d_ff=32
    """
    rng = np.random.RandomState(456)
    nodes = []
    initializers = []
    value_infos = []

    seq_len = 4
    d_model = 16
    d_ff = 32
    vocab_size = 32

    # Embedding table
    embed_table = rng.randn(vocab_size, d_model).astype(np.float32) * 0.1
    initializers.append(numpy_helper.from_array(embed_table, name="embed_table"))

    # Gather (embedding lookup)
    nodes.append(
        helper.make_node("Gather", ["embed_table", "input_ids"], ["embed_out"], name="Gather_Embed")
    )
    value_infos.append(
        helper.make_tensor_value_info("embed_out", TensorProto.FLOAT, [1, seq_len, d_model])
    )

    # Q, K, V projections (MatMul)
    for name in ["Q", "K", "V"]:
        w = rng.randn(d_model, d_model).astype(np.float32) * 0.1
        initializers.append(numpy_helper.from_array(w, name=f"W_{name}"))
        nodes.append(
            helper.make_node(
                "MatMul",
                ["embed_out", f"W_{name}"],
                [f"{name}_out"],
                name=f"MatMul_{name}",
            )
        )
        value_infos.append(
            helper.make_tensor_value_info(f"{name}_out", TensorProto.FLOAT, [1, seq_len, d_model])
        )

    # Transpose K for attention: (1, 4, 16) -> (1, 16, 4)
    nodes.append(
        helper.make_node("Transpose", ["K_out"], ["K_T"], name="Transpose_K", perm=[0, 2, 1])
    )
    value_infos.append(
        helper.make_tensor_value_info("K_T", TensorProto.FLOAT, [1, d_model, seq_len])
    )

    # Score = Q @ K^T
    nodes.append(helper.make_node("MatMul", ["Q_out", "K_T"], ["score_raw"], name="MatMul_Score"))
    value_infos.append(
        helper.make_tensor_value_info("score_raw", TensorProto.FLOAT, [1, seq_len, seq_len])
    )

    # Scale: score / sqrt(d_model)
    scale_val = np.array(1.0 / np.sqrt(d_model), dtype=np.float32)
    initializers.append(numpy_helper.from_array(scale_val, name="scale_val"))
    nodes.append(
        helper.make_node("Mul", ["score_raw", "scale_val"], ["score_scaled"], name="Mul_Scale")
    )
    value_infos.append(
        helper.make_tensor_value_info("score_scaled", TensorProto.FLOAT, [1, seq_len, seq_len])
    )

    # Softmax
    nodes.append(
        helper.make_node(
            "Softmax", ["score_scaled"], ["attn_weights"], name="Softmax_Attn", axis=-1
        )
    )
    value_infos.append(
        helper.make_tensor_value_info("attn_weights", TensorProto.FLOAT, [1, seq_len, seq_len])
    )

    # Context = attn_weights @ V
    nodes.append(
        helper.make_node("MatMul", ["attn_weights", "V_out"], ["context"], name="MatMul_Context")
    )
    value_infos.append(
        helper.make_tensor_value_info("context", TensorProto.FLOAT, [1, seq_len, d_model])
    )

    # Residual add + LayerNorm1
    nodes.append(
        helper.make_node("Add", ["embed_out", "context"], ["residual1"], name="Add_Residual1")
    )
    value_infos.append(
        helper.make_tensor_value_info("residual1", TensorProto.FLOAT, [1, seq_len, d_model])
    )

    ln1_scale = np.ones(d_model, dtype=np.float32)
    ln1_bias = np.zeros(d_model, dtype=np.float32)
    initializers.append(numpy_helper.from_array(ln1_scale, name="ln1_scale"))
    initializers.append(numpy_helper.from_array(ln1_bias, name="ln1_bias"))
    nodes.append(
        helper.make_node(
            "LayerNormalization",
            ["residual1", "ln1_scale", "ln1_bias"],
            ["ln1_out"],
            name="LayerNorm1",
            axis=-1,
            epsilon=1e-5,
        )
    )
    value_infos.append(
        helper.make_tensor_value_info("ln1_out", TensorProto.FLOAT, [1, seq_len, d_model])
    )

    # FFN: MatMul → GELU (Erf-based) → MatMul
    ffn1_w = rng.randn(d_model, d_ff).astype(np.float32) * 0.1
    ffn1_b = rng.randn(d_ff).astype(np.float32) * 0.01
    initializers.append(numpy_helper.from_array(ffn1_w, name="ffn1_w"))
    initializers.append(numpy_helper.from_array(ffn1_b, name="ffn1_b"))
    nodes.append(helper.make_node("MatMul", ["ln1_out", "ffn1_w"], ["ffn1_mm"], name="FFN1_MatMul"))
    value_infos.append(
        helper.make_tensor_value_info("ffn1_mm", TensorProto.FLOAT, [1, seq_len, d_ff])
    )
    nodes.append(helper.make_node("Add", ["ffn1_mm", "ffn1_b"], ["ffn1_out"], name="FFN1_Add"))
    value_infos.append(
        helper.make_tensor_value_info("ffn1_out", TensorProto.FLOAT, [1, seq_len, d_ff])
    )

    # GELU = 0.5 * x * (1 + erf(x / sqrt(2)))
    sqrt2_inv = np.array(1.0 / np.sqrt(2.0), dtype=np.float32)
    half_const = np.array(0.5, dtype=np.float32)
    one_const = np.array(1.0, dtype=np.float32)
    initializers.append(numpy_helper.from_array(sqrt2_inv, name="sqrt2_inv"))
    initializers.append(numpy_helper.from_array(half_const, name="half_const"))
    initializers.append(numpy_helper.from_array(one_const, name="one_const"))

    nodes.append(
        helper.make_node("Mul", ["ffn1_out", "sqrt2_inv"], ["gelu_scaled"], name="GELU_scale")
    )
    value_infos.append(
        helper.make_tensor_value_info("gelu_scaled", TensorProto.FLOAT, [1, seq_len, d_ff])
    )
    nodes.append(helper.make_node("Erf", ["gelu_scaled"], ["gelu_erf"], name="GELU_erf"))
    value_infos.append(
        helper.make_tensor_value_info("gelu_erf", TensorProto.FLOAT, [1, seq_len, d_ff])
    )
    nodes.append(helper.make_node("Add", ["gelu_erf", "one_const"], ["gelu_add"], name="GELU_add"))
    value_infos.append(
        helper.make_tensor_value_info("gelu_add", TensorProto.FLOAT, [1, seq_len, d_ff])
    )
    nodes.append(helper.make_node("Mul", ["ffn1_out", "gelu_add"], ["gelu_mul1"], name="GELU_mul1"))
    value_infos.append(
        helper.make_tensor_value_info("gelu_mul1", TensorProto.FLOAT, [1, seq_len, d_ff])
    )
    nodes.append(
        helper.make_node("Mul", ["gelu_mul1", "half_const"], ["gelu_out"], name="GELU_mul2")
    )
    value_infos.append(
        helper.make_tensor_value_info("gelu_out", TensorProto.FLOAT, [1, seq_len, d_ff])
    )

    # FFN2
    ffn2_w = rng.randn(d_ff, d_model).astype(np.float32) * 0.1
    ffn2_b = rng.randn(d_model).astype(np.float32) * 0.01
    initializers.append(numpy_helper.from_array(ffn2_w, name="ffn2_w"))
    initializers.append(numpy_helper.from_array(ffn2_b, name="ffn2_b"))
    nodes.append(
        helper.make_node("MatMul", ["gelu_out", "ffn2_w"], ["ffn2_mm"], name="FFN2_MatMul")
    )
    value_infos.append(
        helper.make_tensor_value_info("ffn2_mm", TensorProto.FLOAT, [1, seq_len, d_model])
    )
    nodes.append(helper.make_node("Add", ["ffn2_mm", "ffn2_b"], ["ffn2_out"], name="FFN2_Add"))
    value_infos.append(
        helper.make_tensor_value_info("ffn2_out", TensorProto.FLOAT, [1, seq_len, d_model])
    )

    # Residual add + LayerNorm2
    nodes.append(
        helper.make_node("Add", ["ln1_out", "ffn2_out"], ["residual2"], name="Add_Residual2")
    )
    value_infos.append(
        helper.make_tensor_value_info("residual2", TensorProto.FLOAT, [1, seq_len, d_model])
    )

    ln2_scale = np.ones(d_model, dtype=np.float32)
    ln2_bias = np.zeros(d_model, dtype=np.float32)
    initializers.append(numpy_helper.from_array(ln2_scale, name="ln2_scale"))
    initializers.append(numpy_helper.from_array(ln2_bias, name="ln2_bias"))
    nodes.append(
        helper.make_node(
            "LayerNormalization",
            ["residual2", "ln2_scale", "ln2_bias"],
            ["output"],
            name="LayerNorm2",
            axis=-1,
            epsilon=1e-5,
        )
    )

    graph_input = helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, seq_len])
    graph_output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, seq_len, d_model])

    graph = helper.make_graph(
        nodes=nodes,
        name="simple_transformer",
        inputs=[graph_input],
        outputs=[graph_output],
        initializer=initializers,
        value_info=value_infos,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def build_resnet_block() -> onnx.ModelProto:
    """Build a residual block: Conv(3x3)→BN→ReLU→Conv(3x3)→BN→Add(skip)→ReLU.

    Input: (1, 16, 8, 8), Output: (1, 16, 8, 8).
    Tests skip connection scheduling.
    """
    rng = np.random.RandomState(789)
    nodes = []
    initializers = []
    value_infos = []

    C = 16  # channels throughout (no projection needed)

    # Conv1: 16→16 channels, 3x3 kernel, pad=1
    conv1_w = rng.randn(C, C, 3, 3).astype(np.float32) * 0.1
    conv1_b = rng.randn(C).astype(np.float32) * 0.01
    initializers.append(numpy_helper.from_array(conv1_w, name="res_conv1_w"))
    initializers.append(numpy_helper.from_array(conv1_b, name="res_conv1_b"))
    nodes.append(
        helper.make_node(
            "Conv",
            ["input", "res_conv1_w", "res_conv1_b"],
            ["res_conv1_out"],
            name="ResConv1",
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
        )
    )
    value_infos.append(
        helper.make_tensor_value_info("res_conv1_out", TensorProto.FLOAT, [1, C, 8, 8])
    )

    # BN1
    bn1_scale = np.ones(C, dtype=np.float32)
    bn1_bias = np.zeros(C, dtype=np.float32)
    bn1_mean = np.zeros(C, dtype=np.float32)
    bn1_var = np.ones(C, dtype=np.float32)
    initializers.extend(
        [
            numpy_helper.from_array(bn1_scale, name="res_bn1_scale"),
            numpy_helper.from_array(bn1_bias, name="res_bn1_bias"),
            numpy_helper.from_array(bn1_mean, name="res_bn1_mean"),
            numpy_helper.from_array(bn1_var, name="res_bn1_var"),
        ]
    )
    nodes.append(
        helper.make_node(
            "BatchNormalization",
            ["res_conv1_out", "res_bn1_scale", "res_bn1_bias", "res_bn1_mean", "res_bn1_var"],
            ["res_bn1_out"],
            name="ResBN1",
            epsilon=1e-5,
        )
    )
    value_infos.append(
        helper.make_tensor_value_info("res_bn1_out", TensorProto.FLOAT, [1, C, 8, 8])
    )

    # ReLU1
    nodes.append(helper.make_node("Relu", ["res_bn1_out"], ["res_relu1_out"], name="ResReLU1"))
    value_infos.append(
        helper.make_tensor_value_info("res_relu1_out", TensorProto.FLOAT, [1, C, 8, 8])
    )

    # Conv2: 16→16 channels, 3x3 kernel, pad=1
    conv2_w = rng.randn(C, C, 3, 3).astype(np.float32) * 0.1
    conv2_b = rng.randn(C).astype(np.float32) * 0.01
    initializers.append(numpy_helper.from_array(conv2_w, name="res_conv2_w"))
    initializers.append(numpy_helper.from_array(conv2_b, name="res_conv2_b"))
    nodes.append(
        helper.make_node(
            "Conv",
            ["res_relu1_out", "res_conv2_w", "res_conv2_b"],
            ["res_conv2_out"],
            name="ResConv2",
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
        )
    )
    value_infos.append(
        helper.make_tensor_value_info("res_conv2_out", TensorProto.FLOAT, [1, C, 8, 8])
    )

    # BN2
    bn2_scale = np.ones(C, dtype=np.float32)
    bn2_bias = np.zeros(C, dtype=np.float32)
    bn2_mean = np.zeros(C, dtype=np.float32)
    bn2_var = np.ones(C, dtype=np.float32)
    initializers.extend(
        [
            numpy_helper.from_array(bn2_scale, name="res_bn2_scale"),
            numpy_helper.from_array(bn2_bias, name="res_bn2_bias"),
            numpy_helper.from_array(bn2_mean, name="res_bn2_mean"),
            numpy_helper.from_array(bn2_var, name="res_bn2_var"),
        ]
    )
    nodes.append(
        helper.make_node(
            "BatchNormalization",
            ["res_conv2_out", "res_bn2_scale", "res_bn2_bias", "res_bn2_mean", "res_bn2_var"],
            ["res_bn2_out"],
            name="ResBN2",
            epsilon=1e-5,
        )
    )
    value_infos.append(
        helper.make_tensor_value_info("res_bn2_out", TensorProto.FLOAT, [1, C, 8, 8])
    )

    # Skip connection: Add(input, bn2_out)
    nodes.append(helper.make_node("Add", ["input", "res_bn2_out"], ["res_add_out"], name="ResAdd"))
    value_infos.append(
        helper.make_tensor_value_info("res_add_out", TensorProto.FLOAT, [1, C, 8, 8])
    )

    # Final ReLU
    nodes.append(helper.make_node("Relu", ["res_add_out"], ["output"], name="ResFinalReLU"))

    graph_input = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, C, 8, 8])
    graph_output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, C, 8, 8])

    graph = helper.make_graph(
        nodes=nodes,
        name="resnet_block",
        inputs=[graph_input],
        outputs=[graph_output],
        initializer=initializers,
        value_info=value_infos,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


@pytest.fixture
def resnet_block_model_path(tmp_path: Path) -> Path:
    """Save synthetic ResNet block to a temp file and return its path."""
    model = build_resnet_block()
    path = tmp_path / "resnet_block.onnx"
    onnx.save(model, str(path))
    return path


@pytest.fixture
def cnn_model_path(tmp_path: Path) -> Path:
    """Save synthetic CNN model to a temp file and return its path."""
    model = build_simple_cnn()
    path = tmp_path / "cnn_model.onnx"
    onnx.save(model, str(path))
    return path


@pytest.fixture
def transformer_model_path(tmp_path: Path) -> Path:
    """Save synthetic transformer model to a temp file and return its path."""
    model = build_simple_transformer_encoder()
    path = tmp_path / "transformer_model.onnx"
    onnx.save(model, str(path))
    return path
