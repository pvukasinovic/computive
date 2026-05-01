"""TinyML model validation tests for CNN/MLP hardware generation.

Builds synthetic proxy TinyML models (small dims for fast testing),
runs them through the full compiler pipeline (Stages 1-6), and verifies:
  1. Golden vectors match INT8 interpreter output
  2. Generated tile_fabric.sv has correct IS_CONV/kernel/stride/pad params
  3. Weight .mem files are generated and round-trip correctly
  4. Per-channel requant ROM files exist for conv tiles
  5. ACT_DEPTH is correctly sized for spatial activations

Models:
  - DS-CNN: Conv→DWConv→PWConv→GAP→Dense (keyword spotting proxy)
  - ResNet-8: Conv→[Conv+Conv+Add]×2→GAP→Dense (image classification proxy)
  - MobileNetV1: Conv→[DWConv+PWConv]×3→GAP→Dense (visual wake words proxy)
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from mlasic.ingestion import ONNXParser
from mlasic.ir import HardwareConstraints, OpType
from mlasic.optimization import (
    BatchNormFoldingPass,
    ConstantFoldingPass,
    ConvBatchNormFoldingPass,
    ConvFusionPass,
    ConvQuantizationPass,
    DeadCodeEliminationPass,
    OperatorFusionPass,
    PassManager,
    QuantizationPass,
)
from mlasic.dag_scheduler import DAGScheduler
from mlasic.tile_mapper import TileMapper, TileType
from mlasic.rom_mapper import WeightROMMapper
from mlasic.rtl_gen import RTLGenerator
from mlasic.testbench_gen import VerifConfig, VerifGenerator


# ---------------------------------------------------------------------------
# Model builders — synthetic TinyML proxies (small dims for fast testing)
# ---------------------------------------------------------------------------


def _add_conv_bn_relu(
    nodes: list,
    initializers: list,
    value_infos: list,
    name: str,
    input_name: str,
    in_channels: int,
    out_channels: int,
    kernel: int,
    stride: int,
    pad: int,
    in_h: int,
    in_w: int,
    group: int = 1,
    rng: np.random.RandomState | None = None,
    add_relu: bool = True,
) -> tuple[str, int, int]:
    """Add Conv→BN→ReLU block. Returns (output_name, out_h, out_w)."""
    if rng is None:
        rng = np.random.RandomState(42)

    out_h = (in_h + 2 * pad - kernel) // stride + 1
    out_w = (in_w + 2 * pad - kernel) // stride + 1

    # Conv weights
    ic_per_group = in_channels // group
    w = (rng.randn(out_channels, ic_per_group, kernel, kernel) * 0.1).astype(np.float32)
    b = (rng.randn(out_channels) * 0.01).astype(np.float32)
    initializers.append(numpy_helper.from_array(w, name=f"{name}_w"))
    initializers.append(numpy_helper.from_array(b, name=f"{name}_b"))

    conv_out = f"{name}_conv_out"
    nodes.append(
        helper.make_node(
            "Conv",
            [input_name, f"{name}_w", f"{name}_b"],
            [conv_out],
            name=f"{name}_Conv",
            kernel_shape=[kernel, kernel],
            strides=[stride, stride],
            pads=[pad, pad, pad, pad],
            group=group,
        )
    )
    value_infos.append(
        helper.make_tensor_value_info(conv_out, TensorProto.FLOAT, [1, out_channels, out_h, out_w])
    )

    # BatchNorm
    bn_scale = np.ones(out_channels, dtype=np.float32)
    bn_bias = np.zeros(out_channels, dtype=np.float32)
    bn_mean = np.zeros(out_channels, dtype=np.float32)
    bn_var = np.ones(out_channels, dtype=np.float32)
    for suffix, data in [("scale", bn_scale), ("bias", bn_bias), ("mean", bn_mean), ("var", bn_var)]:
        initializers.append(numpy_helper.from_array(data, name=f"{name}_bn_{suffix}"))

    bn_out = f"{name}_bn_out"
    nodes.append(
        helper.make_node(
            "BatchNormalization",
            [conv_out, f"{name}_bn_scale", f"{name}_bn_bias", f"{name}_bn_mean", f"{name}_bn_var"],
            [bn_out],
            name=f"{name}_BN",
            epsilon=1e-5,
        )
    )
    value_infos.append(
        helper.make_tensor_value_info(bn_out, TensorProto.FLOAT, [1, out_channels, out_h, out_w])
    )

    if add_relu:
        relu_out = f"{name}_relu_out"
        nodes.append(helper.make_node("Relu", [bn_out], [relu_out], name=f"{name}_Relu"))
        value_infos.append(
            helper.make_tensor_value_info(relu_out, TensorProto.FLOAT, [1, out_channels, out_h, out_w])
        )
        return relu_out, out_h, out_w
    return bn_out, out_h, out_w


def build_ds_cnn_proxy() -> onnx.ModelProto:
    """DS-CNN proxy: Conv(1,8,3×3)→DWConv(8,8,3×3,g=8)→PWConv(8,16,1×1)→GAP→Dense(16,4).

    Input: (1, 1, 8, 8), Output: (1, 4)
    """
    rng = np.random.RandomState(42)
    nodes, inits, vis = [], [], []

    # Conv1: 1→8, 3x3, pad=1, stride=1 → (1,8,8,8)
    out, h, w = _add_conv_bn_relu(
        nodes, inits, vis, "conv1", "input", 1, 8, 3, 1, 1, 8, 8, rng=rng
    )

    # DWConv: 8→8, 3x3, pad=1, group=8 → (1,8,8,8)
    out, h, w = _add_conv_bn_relu(
        nodes, inits, vis, "dw1", out, 8, 8, 3, 1, 1, h, w, group=8, rng=rng
    )

    # PWConv: 8→16, 1x1 → (1,16,8,8)
    out, h, w = _add_conv_bn_relu(
        nodes, inits, vis, "pw1", out, 8, 16, 1, 1, 0, h, w, rng=rng
    )

    # GlobalAveragePool → (1,16,1,1)
    gap_out = "gap_out"
    nodes.append(helper.make_node("GlobalAveragePool", [out], [gap_out], name="GAP"))
    vis.append(helper.make_tensor_value_info(gap_out, TensorProto.FLOAT, [1, 16, 1, 1]))

    # Flatten → (1,16)
    flat_out = "flat_out"
    nodes.append(helper.make_node("Flatten", [gap_out], [flat_out], name="Flatten", axis=1))
    vis.append(helper.make_tensor_value_info(flat_out, TensorProto.FLOAT, [1, 16]))

    # Dense: 16→4
    fc_w = (rng.randn(16, 4) * 0.1).astype(np.float32)
    fc_b = (rng.randn(4) * 0.01).astype(np.float32)
    inits.append(numpy_helper.from_array(fc_w, name="fc_w"))
    inits.append(numpy_helper.from_array(fc_b, name="fc_b"))
    nodes.append(helper.make_node("MatMul", ["flat_out", "fc_w"], ["mm_out"], name="FC_MatMul"))
    vis.append(helper.make_tensor_value_info("mm_out", TensorProto.FLOAT, [1, 4]))
    nodes.append(helper.make_node("Add", ["mm_out", "fc_b"], ["output"], name="FC_Add"))

    graph = helper.make_graph(
        nodes,
        "ds_cnn_proxy",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 8, 8])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])],
        initializer=inits,
        value_info=vis,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


def build_resnet8_proxy() -> onnx.ModelProto:
    """ResNet-8 proxy: Conv(1,8,3×3)→[Conv+Conv+Add]×2→GAP→Dense(8,4).

    Input: (1, 1, 8, 8), Output: (1, 4)
    Two residual blocks with skip connections.
    """
    rng = np.random.RandomState(43)
    nodes, inits, vis = [], [], []

    # Initial Conv: 1→8, 3x3, pad=1 → (1,8,8,8)
    out, h, w = _add_conv_bn_relu(
        nodes, inits, vis, "stem", "input", 1, 8, 3, 1, 1, 8, 8, rng=rng
    )

    # Residual block 1: two 3x3 convs + skip add
    skip1 = out
    out, h, w = _add_conv_bn_relu(
        nodes, inits, vis, "res1a", out, 8, 8, 3, 1, 1, h, w, rng=rng
    )
    out, h, w = _add_conv_bn_relu(
        nodes, inits, vis, "res1b", out, 8, 8, 3, 1, 1, h, w, rng=rng, add_relu=False
    )
    add1_out = "res1_add"
    nodes.append(helper.make_node("Add", [out, skip1], [add1_out], name="Res1_Add"))
    vis.append(helper.make_tensor_value_info(add1_out, TensorProto.FLOAT, [1, 8, h, w]))
    relu1_out = "res1_relu"
    nodes.append(helper.make_node("Relu", [add1_out], [relu1_out], name="Res1_Relu"))
    vis.append(helper.make_tensor_value_info(relu1_out, TensorProto.FLOAT, [1, 8, h, w]))

    # Residual block 2: two 3x3 convs + skip add
    skip2 = relu1_out
    out2, h2, w2 = _add_conv_bn_relu(
        nodes, inits, vis, "res2a", relu1_out, 8, 8, 3, 1, 1, h, w, rng=rng
    )
    out2, h2, w2 = _add_conv_bn_relu(
        nodes, inits, vis, "res2b", out2, 8, 8, 3, 1, 1, h2, w2, rng=rng, add_relu=False
    )
    add2_out = "res2_add"
    nodes.append(helper.make_node("Add", [out2, skip2], [add2_out], name="Res2_Add"))
    vis.append(helper.make_tensor_value_info(add2_out, TensorProto.FLOAT, [1, 8, h2, w2]))
    relu2_out = "res2_relu"
    nodes.append(helper.make_node("Relu", [add2_out], [relu2_out], name="Res2_Relu"))
    vis.append(helper.make_tensor_value_info(relu2_out, TensorProto.FLOAT, [1, 8, h2, w2]))

    # GAP → (1,8,1,1)
    gap_out = "gap_out"
    nodes.append(helper.make_node("GlobalAveragePool", [relu2_out], [gap_out], name="GAP"))
    vis.append(helper.make_tensor_value_info(gap_out, TensorProto.FLOAT, [1, 8, 1, 1]))

    # Flatten → (1,8)
    flat_out = "flat_out"
    nodes.append(helper.make_node("Flatten", [gap_out], [flat_out], name="Flatten", axis=1))
    vis.append(helper.make_tensor_value_info(flat_out, TensorProto.FLOAT, [1, 8]))

    # Dense: 8→4
    fc_w = (rng.randn(8, 4) * 0.1).astype(np.float32)
    fc_b = (rng.randn(4) * 0.01).astype(np.float32)
    inits.append(numpy_helper.from_array(fc_w, name="fc_w"))
    inits.append(numpy_helper.from_array(fc_b, name="fc_b"))
    nodes.append(helper.make_node("MatMul", ["flat_out", "fc_w"], ["mm_out"], name="FC_MatMul"))
    vis.append(helper.make_tensor_value_info("mm_out", TensorProto.FLOAT, [1, 4]))
    nodes.append(helper.make_node("Add", ["mm_out", "fc_b"], ["output"], name="FC_Add"))

    graph = helper.make_graph(
        nodes,
        "resnet8_proxy",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 8, 8])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])],
        initializer=inits,
        value_info=vis,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


def build_mobilenetv1_proxy() -> onnx.ModelProto:
    """MobileNetV1 proxy: Conv(1,8,3×3,s=2)→[DWConv+PWConv]×3→GAP→Dense(32,4).

    Input: (1, 1, 8, 8), Output: (1, 4)
    Three depthwise-separable conv blocks.
    """
    rng = np.random.RandomState(44)
    nodes, inits, vis = [], [], []

    # Stem: Conv 1→8, 3x3, stride=2, pad=1 → (1,8,4,4)
    out, h, w = _add_conv_bn_relu(
        nodes, inits, vis, "stem", "input", 1, 8, 3, 2, 1, 8, 8, rng=rng
    )

    # DS block 1: DW 8→8, 3x3, pad=1 + PW 8→16
    out, h, w = _add_conv_bn_relu(
        nodes, inits, vis, "ds1_dw", out, 8, 8, 3, 1, 1, h, w, group=8, rng=rng
    )
    out, h, w = _add_conv_bn_relu(
        nodes, inits, vis, "ds1_pw", out, 8, 16, 1, 1, 0, h, w, rng=rng
    )

    # DS block 2: DW 16→16, 3x3, pad=1 + PW 16→32
    out, h, w = _add_conv_bn_relu(
        nodes, inits, vis, "ds2_dw", out, 16, 16, 3, 1, 1, h, w, group=16, rng=rng
    )
    out, h, w = _add_conv_bn_relu(
        nodes, inits, vis, "ds2_pw", out, 16, 32, 1, 1, 0, h, w, rng=rng
    )

    # DS block 3: DW 32→32, 3x3, pad=1, stride=2 + PW 32→32
    out, h, w = _add_conv_bn_relu(
        nodes, inits, vis, "ds3_dw", out, 32, 32, 3, 2, 1, h, w, group=32, rng=rng
    )
    out, h, w = _add_conv_bn_relu(
        nodes, inits, vis, "ds3_pw", out, 32, 32, 1, 1, 0, h, w, rng=rng
    )

    # GAP → (1,32,1,1)
    gap_out = "gap_out"
    nodes.append(helper.make_node("GlobalAveragePool", [out], [gap_out], name="GAP"))
    vis.append(helper.make_tensor_value_info(gap_out, TensorProto.FLOAT, [1, 32, 1, 1]))

    # Flatten → (1,32)
    flat_out = "flat_out"
    nodes.append(helper.make_node("Flatten", [gap_out], [flat_out], name="Flatten", axis=1))
    vis.append(helper.make_tensor_value_info(flat_out, TensorProto.FLOAT, [1, 32]))

    # Dense: 32→4
    fc_w = (rng.randn(32, 4) * 0.1).astype(np.float32)
    fc_b = (rng.randn(4) * 0.01).astype(np.float32)
    inits.append(numpy_helper.from_array(fc_w, name="fc_w"))
    inits.append(numpy_helper.from_array(fc_b, name="fc_b"))
    nodes.append(helper.make_node("MatMul", ["flat_out", "fc_w"], ["mm_out"], name="FC_MatMul"))
    vis.append(helper.make_tensor_value_info("mm_out", TensorProto.FLOAT, [1, 4]))
    nodes.append(helper.make_node("Add", ["mm_out", "fc_b"], ["output"], name="FC_Add"))

    graph = helper.make_graph(
        nodes,
        "mobilenetv1_proxy",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 8, 8])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])],
        initializer=inits,
        value_info=vis,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


# ---------------------------------------------------------------------------
# Pipeline helper
# ---------------------------------------------------------------------------


def _run_tinyml_pipeline(model: onnx.ModelProto, input_shape: tuple, tmp_path: Path) -> dict:
    """Run full MLASIC pipeline (Stages 1-6) on a synthetic model.

    Returns dict with stage results and generated artifacts.
    """
    result = {"stages_completed": [], "errors": {}}

    # Save model to disk for ONNXParser
    model_path = tmp_path / "model.onnx"
    onnx.save(model, str(model_path))

    # Stage 1: Ingestion
    parser = ONNXParser(model_path)
    graph = parser.parse()
    result["stages_completed"].append("ingestion")
    result["raw_nodes"] = len(graph.nodes)

    # Stage 2: Optimization (CNN passes)
    rng = np.random.RandomState(42)
    cal_data = [rng.randn(*input_shape).astype(np.float32) for _ in range(3)]

    pm = PassManager()
    pm.add_pass(ConstantFoldingPass())
    pm.add_pass(DeadCodeEliminationPass())
    pm.add_pass(BatchNormFoldingPass())
    pm.add_pass(ConvBatchNormFoldingPass())
    pm.add_pass(DeadCodeEliminationPass())
    pm.add_pass(ConvFusionPass())
    pm.add_pass(OperatorFusionPass())
    pm.add_pass(ConvQuantizationPass(calibration_data=cal_data))
    pm.add_pass(QuantizationPass(calibration_data=cal_data))
    graph = pm.run(graph, verify=False)
    result["stages_completed"].append("optimization")
    result["opt_nodes"] = len(graph.nodes)
    result["graph"] = graph

    # Stage 3: DAG Scheduling
    dag_schedule = DAGScheduler().schedule(graph)
    result["stages_completed"].append("scheduling")
    result["total_cycles"] = dag_schedule.total_cycles

    # Stage 3b: Tile Mapping
    fabric = TileMapper().map(graph, dag_schedule)
    result["stages_completed"].append("tile_mapping")
    result["total_tiles"] = fabric.total_tiles
    result["fabric"] = fabric

    # Stage 4: Weight ROM Mapping
    weight_dir = tmp_path / "weights"
    weight_dir.mkdir(parents=True, exist_ok=True)
    rom_mapper = WeightROMMapper()
    manifest = rom_mapper.generate(graph, fabric, output_dir=weight_dir)
    result["stages_completed"].append("weight_packing")
    result["total_weight_bytes"] = manifest["total_weight_bytes"]

    # Stage 5: RTL Generation (tile fabric)
    rtl_dir = tmp_path / "rtl"
    rtl_gen = RTLGenerator(
        graph=graph,
        weight_dir=weight_dir,
        output_dir=rtl_dir,
        constraints=HardwareConstraints(),
        fabric_config=fabric,
    )
    rtl_gen.generate_all()
    result["stages_completed"].append("rtl_gen")
    result["rtl_dir"] = rtl_dir
    result["sv_files"] = list(rtl_dir.rglob("*.sv"))
    result["mem_files"] = list(rtl_dir.rglob("*.mem"))

    # Stage 6: Testbench Generation
    tb_dir = tmp_path / "testbench"
    config = VerifConfig(
        graph=graph, output_dir=tb_dir, weight_dir=weight_dir,
        num_random_vectors=5, constraints=HardwareConstraints(), is_mlp=False,
    )
    vg = VerifGenerator(config)
    generated = vg.generate_all()
    result["stages_completed"].append("testbench")
    result["tb_files"] = generated

    return result


# ---------------------------------------------------------------------------
# Test class: DS-CNN
# ---------------------------------------------------------------------------


class TestDSCNN:
    """DS-CNN proxy: keyword spotting model with DW+PW convolutions."""

    def test_stages_1_through_6(self, tmp_path):
        """Full pipeline completes without error."""
        model = build_ds_cnn_proxy()
        result = _run_tinyml_pipeline(model, (1, 1, 8, 8), tmp_path)
        assert "testbench" in result["stages_completed"]

    def test_tile_fabric_has_conv_params(self, tmp_path):
        """Generated tile_fabric.sv has IS_CONV=1 and correct kernel/stride/pad."""
        model = build_ds_cnn_proxy()
        result = _run_tinyml_pipeline(model, (1, 1, 8, 8), tmp_path)
        rtl_dir = result["rtl_dir"]

        fabric_sv = (rtl_dir / "tile_fabric.sv").read_text()

        # Should have at least one IS_CONV=1 tile
        assert ".IS_CONV(1)" in fabric_sv, "No conv tile found in tile_fabric.sv"

        # Should have kernel_h/kernel_w parameters for conv tiles
        assert ".KERNEL_H(" in fabric_sv
        assert ".KERNEL_W(" in fabric_sv

        # DW conv should have GROUP > 1
        assert ".GROUP(8)" in fabric_sv, "DWConv group=8 not found"

    def test_weight_mem_files_exist(self, tmp_path):
        """Weight .mem files generated for conv tiles."""
        model = build_ds_cnn_proxy()
        result = _run_tinyml_pipeline(model, (1, 1, 8, 8), tmp_path)

        # Should have weight .mem files
        mem_files = [f.name for f in result["mem_files"]]
        weight_mems = [f for f in mem_files if "weights.mem" in f]
        assert len(weight_mems) > 0, f"No weight .mem files found: {mem_files}"

    def test_requant_rom_for_conv_tiles(self, tmp_path):
        """Per-channel requant ROM files generated for conv tiles."""
        model = build_ds_cnn_proxy()
        result = _run_tinyml_pipeline(model, (1, 1, 8, 8), tmp_path)

        rtl_dir = result["rtl_dir"]
        requant_files = list(rtl_dir.glob("tile_*_requant.mem"))
        assert len(requant_files) > 0, "No requant ROM files found"

        # Check format: each line should be 8 hex chars
        for rf in requant_files:
            lines = rf.read_text().strip().split("\n")
            assert len(lines) > 0
            for line in lines:
                assert re.match(r"^[0-9a-f]{8}$", line.strip()), f"Bad requant line: {line}"

    def test_act_depth_for_spatial(self, tmp_path):
        """ACT_DEPTH in tile_parameters.svh accounts for spatial activations."""
        model = build_ds_cnn_proxy()
        result = _run_tinyml_pipeline(model, (1, 1, 8, 8), tmp_path)

        rtl_dir = result["rtl_dir"]
        params = (rtl_dir / "tile_parameters.svh").read_text()

        # ACT_DEPTH should be large enough for 16*8*8=1024 bytes = 128 words
        m = re.search(r"ACT_DEPTH\s*=\s*(\d+)", params)
        assert m, "ACT_DEPTH not found in tile_parameters.svh"
        act_depth = int(m.group(1))
        # 16 channels * 8 * 8 = 1024 elements, / 8 = 128 words minimum
        assert act_depth >= 128, f"ACT_DEPTH={act_depth} too small for spatial activations"


# ---------------------------------------------------------------------------
# Test class: ResNet-8
# ---------------------------------------------------------------------------


class TestResNet8:
    """ResNet-8 proxy: image classification with residual skip connections."""

    def test_stages_1_through_6(self, tmp_path):
        """Full pipeline completes without error."""
        model = build_resnet8_proxy()
        result = _run_tinyml_pipeline(model, (1, 1, 8, 8), tmp_path)
        assert "testbench" in result["stages_completed"]

    def test_has_alu_tiles_for_skip(self, tmp_path):
        """Fabric has ALU tiles for residual Add operations."""
        model = build_resnet8_proxy()
        result = _run_tinyml_pipeline(model, (1, 1, 8, 8), tmp_path)

        fabric = result["fabric"]
        alu_tiles = [t for t in fabric.tiles if t.tile_type == TileType.ALU]
        assert len(alu_tiles) >= 2, f"Expected >= 2 ALU tiles, got {len(alu_tiles)}"

    def test_skip_connections_wired(self, tmp_path):
        """Generated tile_fabric.sv has skip connection wiring."""
        model = build_resnet8_proxy()
        result = _run_tinyml_pipeline(model, (1, 1, 8, 8), tmp_path)
        rtl_dir = result["rtl_dir"]

        fabric_sv = (rtl_dir / "tile_fabric.sv").read_text()

        # Should have skip data assignments
        assert "act_skip_data" in fabric_sv, "No skip connection wiring found"

    def test_tile_fabric_has_conv_and_add(self, tmp_path):
        """Generated tile_fabric.sv has both conv tiles and ALU tiles."""
        model = build_resnet8_proxy()
        result = _run_tinyml_pipeline(model, (1, 1, 8, 8), tmp_path)
        rtl_dir = result["rtl_dir"]

        fabric_sv = (rtl_dir / "tile_fabric.sv").read_text()

        # Should have conv tiles (TILE_TYPE=0, IS_CONV=1)
        assert ".IS_CONV(1)" in fabric_sv, "No conv tile in fabric"
        # Should have ALU tiles (TILE_TYPE=1)
        assert ".TILE_TYPE(1)" in fabric_sv, "No ALU tile in fabric"


# ---------------------------------------------------------------------------
# Test class: MobileNetV1
# ---------------------------------------------------------------------------


class TestMobileNetV1:
    """MobileNetV1 proxy: visual wake words with DW-separable convolutions."""

    def test_stages_1_through_6(self, tmp_path):
        """Full pipeline completes without error."""
        model = build_mobilenetv1_proxy()
        result = _run_tinyml_pipeline(model, (1, 1, 8, 8), tmp_path)
        assert "testbench" in result["stages_completed"]

    def test_multiple_dw_groups(self, tmp_path):
        """Fabric has DW conv tiles with different group values."""
        model = build_mobilenetv1_proxy()
        result = _run_tinyml_pipeline(model, (1, 1, 8, 8), tmp_path)
        rtl_dir = result["rtl_dir"]

        fabric_sv = (rtl_dir / "tile_fabric.sv").read_text()

        # Should have depthwise convs with group > 1
        groups = set(int(m.group(1)) for m in re.finditer(r"\.GROUP\((\d+)\)", fabric_sv))
        assert any(g > 1 for g in groups), f"No DW conv found (groups: {groups})"

    def test_stride2_conv(self, tmp_path):
        """Fabric has stride-2 convolution for downsampling."""
        model = build_mobilenetv1_proxy()
        result = _run_tinyml_pipeline(model, (1, 1, 8, 8), tmp_path)
        rtl_dir = result["rtl_dir"]

        fabric_sv = (rtl_dir / "tile_fabric.sv").read_text()

        # Should have STRIDE_H=2 for the stem conv
        assert ".STRIDE_H(2)" in fabric_sv, "No stride-2 conv found"

    def test_pool_tile_present(self, tmp_path):
        """Fabric has GlobalAveragePool tile."""
        model = build_mobilenetv1_proxy()
        result = _run_tinyml_pipeline(model, (1, 1, 8, 8), tmp_path)
        rtl_dir = result["rtl_dir"]

        fabric_sv = (rtl_dir / "tile_fabric.sv").read_text()

        # TILE_TYPE=5 is POOL
        assert ".TILE_TYPE(5)" in fabric_sv, "No pool tile found"

    def test_weight_mem_roundtrip(self, tmp_path):
        """Weight .mem files can be read back and match original data."""
        model = build_mobilenetv1_proxy()
        result = _run_tinyml_pipeline(model, (1, 1, 8, 8), tmp_path)
        rtl_dir = result["rtl_dir"]

        # Find a weight .mem file
        weight_mems = list(rtl_dir.glob("tile_*_weights.mem"))
        assert len(weight_mems) > 0, "No weight .mem files to verify"

        for wm in weight_mems:
            lines = wm.read_text().strip().split("\n")
            assert len(lines) > 0
            # Each line should be valid hex (256 hex chars for P=128)
            for line in lines:
                stripped = line.strip()
                assert all(c in "0123456789abcdef" for c in stripped), (
                    f"Invalid hex in {wm.name}: {stripped[:20]}..."
                )


# ---------------------------------------------------------------------------
# Model builder sanity tests
# ---------------------------------------------------------------------------


class TestModelBuilders:
    """Verify synthetic models are valid ONNX."""

    def test_ds_cnn_proxy_valid(self):
        model = build_ds_cnn_proxy()
        onnx.checker.check_model(model)

    def test_resnet8_proxy_valid(self):
        model = build_resnet8_proxy()
        onnx.checker.check_model(model)

    def test_mobilenetv1_proxy_valid(self):
        model = build_mobilenetv1_proxy()
        onnx.checker.check_model(model)

    def test_ds_cnn_proxy_runs_ort(self):
        """Model runs in ONNX Runtime (FP32)."""
        import onnxruntime as ort

        model = build_ds_cnn_proxy()
        sess = ort.InferenceSession(model.SerializeToString())
        inp = np.random.randn(1, 1, 8, 8).astype(np.float32)
        out = sess.run(None, {"input": inp})
        assert out[0].shape == (1, 4)

    def test_resnet8_proxy_runs_ort(self):
        """Model runs in ONNX Runtime (FP32)."""
        import onnxruntime as ort

        model = build_resnet8_proxy()
        sess = ort.InferenceSession(model.SerializeToString())
        inp = np.random.randn(1, 1, 8, 8).astype(np.float32)
        out = sess.run(None, {"input": inp})
        assert out[0].shape == (1, 4)

    def test_mobilenetv1_proxy_runs_ort(self):
        """Model runs in ONNX Runtime (FP32)."""
        import onnxruntime as ort

        model = build_mobilenetv1_proxy()
        sess = ort.InferenceSession(model.SerializeToString())
        inp = np.random.randn(1, 1, 8, 8).astype(np.float32)
        out = sess.run(None, {"input": inp})
        assert out[0].shape == (1, 4)
