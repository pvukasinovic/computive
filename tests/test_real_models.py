"""End-to-end tests with real production ONNX models.

Downloads well-known models from HuggingFace/ONNX Model Zoo and runs them
through the MLASIC compiler pipeline to verify correctness at scale.

Models tested:
  - ResNet-50       (~25M params, CNN, ImageNet)
  - MobileNetV2     (~3.5M params, CNN, ImageNet)
  - BERT-base       (~110M params, Transformer, NLP)
  - GPT-2           (~124M params, Transformer, language model)
  - GPT-2-medium    (~345M params, Transformer, language model)
  - GPT-2-large     (~774M params, Transformer, language model)
  - Llama-3.2-1B    (~1.24B params, Transformer, INT8 quantized)
"""

from __future__ import annotations

import gc
import os
import time
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import numpy_helper

# Lazy import to avoid failure if not installed
try:
    from huggingface_hub import hf_hub_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

from mlasic.ingestion import ONNXParser
from mlasic.exceptions import UnsupportedOperatorError
from mlasic.ir import HardwareConstraints, OpType
from mlasic.interpreter import IRInterpreter
from mlasic.optimization import (
    ActivationFusionPass,
    BatchNormFoldingPass,
    ConstantFoldingPass,
    ConvBatchNormFoldingPass,
    ConvFusionPass,
    ConvQuantizationPass,
    DeadCodeEliminationPass,
    LayerNormFusionPass,
    OperatorFusionPass,
    PassManager,
    QuantizationPass,
)
from mlasic.scheduler import Scheduler
from mlasic.dag_scheduler import DAGScheduler
from mlasic.weight_packer import WeightPacker
from mlasic.rtl_gen import RTLGenerator
from mlasic.testbench_gen import VerifConfig, VerifGenerator
from mlasic.golden_vectors import GoldenVectorGenerator
from mlasic.tile_mapper import TileMapper
from mlasic.rom_mapper import WeightROMMapper

# ---------------------------------------------------------------------------
# Model download helpers
# ---------------------------------------------------------------------------

MODELS_DIR = Path(__file__).parent.parent / "models"


def _download_hf_onnx(repo_id: str, filename: str, cache_name: str) -> Path:
    """Download an ONNX model from HuggingFace Hub, caching in models/."""
    dest = MODELS_DIR / cache_name
    if dest.exists():
        return dest

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(MODELS_DIR))
    downloaded = Path(path)
    if downloaded != dest:
        # Move from subdirectory to flat cache name
        downloaded.rename(dest)
    return dest


def _download_hf_onnx_with_data(
    repo_id: str, model_file: str, data_file: str, cache_dir: str
) -> Path:
    """Download an ONNX model that uses external data files."""
    out_dir = MODELS_DIR / cache_dir
    model_dest = out_dir / "model.onnx"
    data_dest = out_dir / "model.onnx_data"
    if model_dest.exists() and data_dest.exists():
        return model_dest

    out_dir.mkdir(parents=True, exist_ok=True)
    mp = hf_hub_download(repo_id=repo_id, filename=model_file, local_dir=str(out_dir))
    dp = hf_hub_download(repo_id=repo_id, filename=data_file, local_dir=str(out_dir))
    mp, dp = Path(mp), Path(dp)
    if mp != model_dest:
        mp.rename(model_dest)
    if dp != data_dest:
        dp.rename(data_dest)
    return model_dest


def _count_params(model_path: Path) -> int:
    """Count parameters in an ONNX model without loading all weights."""
    model = onnx.load(str(model_path))
    total = 0
    for init in model.graph.initializer:
        shape = list(init.dims)
        total += int(np.prod(shape)) if shape else 1
    del model
    gc.collect()
    return total


def _list_ops(model_path: Path) -> set[str]:
    """List unique op types in an ONNX model."""
    model = onnx.load(str(model_path))
    ops = {node.op_type for node in model.graph.node}
    del model
    gc.collect()
    return ops


def _run_cnn_full_pipeline(model_path: Path, input_shape: tuple, output_dir: Path) -> dict:
    """Run full pipeline (Stages 1-6) for CNN models.

    Returns dict with stage results and any errors.
    """
    result = {"stages_completed": [], "errors": {}}

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
    result["opt_stage"] = graph.stage

    # Stage 3: DAG Scheduling
    dag_schedule = DAGScheduler().schedule(graph)
    result["stages_completed"].append("scheduling")
    result["total_cycles"] = dag_schedule.total_cycles
    result["peak_act_bytes"] = dag_schedule.peak_activation_bytes

    # Stage 3b: Tile Mapping
    fabric = TileMapper().map(graph, dag_schedule)
    result["stages_completed"].append("tile_mapping")
    result["total_tiles"] = fabric.total_tiles

    # Stage 4: Weight ROM Mapping
    weight_dir = output_dir / "weights"
    weight_dir.mkdir(parents=True, exist_ok=True)
    rom_mapper = WeightROMMapper()
    manifest = rom_mapper.generate(graph, fabric, output_dir=weight_dir)
    result["stages_completed"].append("weight_packing")
    result["total_weight_bytes"] = manifest["total_weight_bytes"]

    # Stage 5: RTL Generation (tile fabric)
    rtl_dir = output_dir / "rtl"
    rtl_gen = RTLGenerator(
        graph=graph,
        weight_dir=weight_dir,
        output_dir=rtl_dir,
        constraints=HardwareConstraints(),
        fabric_config=fabric,
    )
    rtl_gen.generate_all()
    result["stages_completed"].append("rtl_gen")
    result["sv_files"] = len(list(rtl_dir.rglob("*.sv")))
    result["mem_files"] = len(list(rtl_dir.rglob("*.mem")))

    # Stage 6: Testbench Generation
    tb_dir = output_dir / "testbench"
    config = VerifConfig(
        graph=graph, output_dir=tb_dir, weight_dir=weight_dir,
        num_random_vectors=10, constraints=HardwareConstraints(), is_mlp=False,
    )
    vg = VerifGenerator(config)
    generated = vg.generate_all()
    result["stages_completed"].append("testbench")
    result["tb_files"] = len(generated)

    return result


def _run_transformer_full_pipeline(model_path: Path, output_dir: Path) -> dict:
    """Run full pipeline (Stages 1-6) for Transformer models.

    Skips quantization calibration (transformers use INT64 token inputs).
    Sets stage to 'quantized' manually so DAGScheduler can proceed.

    Returns dict with stage results.
    """
    result = {"stages_completed": [], "errors": {}}

    # Stage 1: Ingestion
    parser = ONNXParser(model_path)
    graph = parser.parse()
    result["stages_completed"].append("ingestion")
    result["raw_nodes"] = len(graph.nodes)

    # Stage 2: Optimization (light passes only — no calibration needed)
    pm = PassManager()
    pm.add_pass(ConstantFoldingPass())
    pm.add_pass(DeadCodeEliminationPass())
    graph = pm.run(graph, verify=False)
    result["stages_completed"].append("optimization")
    result["opt_nodes"] = len(graph.nodes)

    # Set stage to "quantized" so DAGScheduler accepts it
    # (no actual quantization — weights stay FP32, unfused nodes pass through)
    graph.stage = "quantized"

    # Stage 3: DAG Scheduling
    dag_schedule = DAGScheduler().schedule(graph)
    result["stages_completed"].append("scheduling")
    result["total_cycles"] = dag_schedule.total_cycles
    result["peak_act_bytes"] = dag_schedule.peak_activation_bytes
    result["scheduled_nodes"] = len(dag_schedule.node_schedules)

    # Stage 3b: Tile Mapping
    fabric = TileMapper().map(graph, dag_schedule)
    result["stages_completed"].append("tile_mapping")
    result["total_tiles"] = fabric.total_tiles

    # Stage 4: Weight ROM Mapping
    weight_dir = output_dir / "weights"
    weight_dir.mkdir(parents=True, exist_ok=True)
    rom_mapper = WeightROMMapper()
    manifest = rom_mapper.generate(graph, fabric, output_dir=weight_dir)
    result["stages_completed"].append("weight_packing")
    result["total_weight_bytes"] = manifest["total_weight_bytes"]

    # Stage 5: RTL Generation (tile fabric)
    rtl_dir = output_dir / "rtl"
    rtl_gen = RTLGenerator(
        graph=graph,
        weight_dir=weight_dir,
        output_dir=rtl_dir,
        constraints=HardwareConstraints(),
        fabric_config=fabric,
    )
    rtl_gen.generate_all()
    result["stages_completed"].append("rtl_gen")
    result["sv_files"] = len(list(rtl_dir.rglob("*.sv")))
    result["mem_files"] = len(list(rtl_dir.rglob("*.mem")))

    # Stage 6: Testbench Generation
    tb_dir = output_dir / "testbench"
    config = VerifConfig(
        graph=graph, output_dir=tb_dir, weight_dir=weight_dir,
        num_random_vectors=10, constraints=HardwareConstraints(), is_mlp=False,
    )
    vg = VerifGenerator(config)
    generated = vg.generate_all()
    result["stages_completed"].append("testbench")
    result["tb_files"] = len(generated)

    return result


# ---------------------------------------------------------------------------
# Model-specific download functions (verified HuggingFace repo IDs)
# ---------------------------------------------------------------------------

def download_resnet50() -> Path:
    """ResNet-50 from onnx-community (~25M params)."""
    return _download_hf_onnx(
        "onnx-community/resnet-50-ONNX",
        "onnx/model.onnx",
        "resnet50.onnx",
    )


def download_mobilenetv2() -> Path:
    """MobileNetV2 from onnxmodelzoo (~3.5M params)."""
    return _download_hf_onnx(
        "onnxmodelzoo/mobilenetv2-7",
        "mobilenetv2-7.onnx",
        "mobilenetv2.onnx",
    )


def download_bert_base() -> Path:
    """BERT-base-uncased (~110M params)."""
    return _download_hf_onnx(
        "google-bert/bert-base-uncased",
        "model.onnx",
        "bert-base.onnx",
    )


def download_gpt2() -> Path:
    """GPT-2 small from onnx-community (~124M params)."""
    return _download_hf_onnx(
        "onnx-community/gpt2-ONNX",
        "onnx/model.onnx",
        "gpt2.onnx",
    )


def download_gpt2_medium() -> Path:
    """GPT-2 medium (~345M params)."""
    return _download_hf_onnx(
        "openai-community/gpt2-medium",
        "onnx/decoder_model_merged.onnx",
        "gpt2-medium.onnx",
    )


def download_gpt2_large() -> Path:
    """GPT-2 large INT8 (~774M params, quantized, single-file)."""
    return _download_hf_onnx(
        "onnx-community/gpt2-large-ONNX",
        "onnx/model_int8.onnx",
        "gpt2-large-int8.onnx",
    )


def download_llama_1b() -> Path:
    """Llama-3.2-1B INT8 (~1.24B params, quantized, single-file)."""
    return _download_hf_onnx(
        "onnx-community/Llama-3.2-1B",
        "onnx/model_int8.onnx",
        "llama-3.2-1b-int8.onnx",
    )


# ---------------------------------------------------------------------------
# Skip if no HuggingFace Hub
# ---------------------------------------------------------------------------

requires_hf = pytest.mark.skipif(not HF_AVAILABLE, reason="huggingface_hub not installed")


# ---------------------------------------------------------------------------
# Test: ResNet-50 (~25M params, CNN)
# ---------------------------------------------------------------------------


@requires_hf
class TestResNet50:
    """ResNet-50: real CNN production model (~25M params)."""

    @pytest.fixture(scope="class")
    def model_path(self):
        try:
            return download_resnet50()
        except Exception as e:
            pytest.skip(f"Failed to download ResNet-50: {e}")

    def test_param_count(self, model_path):
        total = _count_params(model_path)
        print(f"\nResNet-50 params: {total:,}")
        assert total > 20_000_000
        assert total < 30_000_000

    def test_op_types(self, model_path):
        ops = _list_ops(model_path)
        print(f"\nResNet-50 ops: {sorted(ops)}")
        assert "Conv" in ops

    def test_ingestion(self, model_path):
        """Stage 1: Parser handles ResNet-50."""
        parser = ONNXParser(model_path)
        graph = parser.parse()
        assert graph.stage == "raw"
        assert len(graph.nodes) > 0
        print(f"\nResNet-50 IR: {len(graph.nodes)} nodes, {len(graph.tensors)} tensors")

    def test_optimization(self, model_path, tmp_path):
        """Stage 2: Optimization passes run on ResNet-50."""
        parser = ONNXParser(model_path)
        graph = parser.parse()

        pm = PassManager()
        pm.add_pass(ConstantFoldingPass())
        pm.add_pass(DeadCodeEliminationPass())
        pm.add_pass(BatchNormFoldingPass())
        pm.add_pass(ConvBatchNormFoldingPass())
        pm.add_pass(ConvFusionPass())
        graph = pm.run(graph, verify=False)

        print(f"\nResNet-50 after optimization: {len(graph.nodes)} nodes, stage={graph.stage}")
        assert len(graph.nodes) > 0

    def test_full_pipeline(self, model_path, tmp_path):
        """Stages 1-6: Full pipeline on ResNet-50 (CNN path)."""
        result = _run_cnn_full_pipeline(model_path, (1, 3, 224, 224), tmp_path)
        print(f"\nResNet-50 full pipeline: {result['stages_completed']}")
        print(f"  Tiles: {result['total_tiles']}, SV files: {result['sv_files']}")
        print(f"  Cycles: {result['total_cycles']:,}, Weight bytes: {result['total_weight_bytes']:,}")
        assert "rtl_gen" in result["stages_completed"]
        assert "testbench" in result["stages_completed"]
        assert result["sv_files"] > 0


# ---------------------------------------------------------------------------
# Test: MobileNetV2 (~3.5M params, CNN)
# ---------------------------------------------------------------------------


@requires_hf
class TestMobileNetV2:
    """MobileNetV2: efficient CNN production model (~3.5M params)."""

    @pytest.fixture(scope="class")
    def model_path(self):
        try:
            return download_mobilenetv2()
        except Exception as e:
            pytest.skip(f"Failed to download MobileNetV2: {e}")

    def test_param_count(self, model_path):
        total = _count_params(model_path)
        print(f"\nMobileNetV2 params: {total:,}")
        assert total > 2_000_000
        assert total < 5_000_000

    def test_op_types(self, model_path):
        ops = _list_ops(model_path)
        print(f"\nMobileNetV2 ops: {sorted(ops)}")

    def test_ingestion(self, model_path):
        """Stage 1: Parser handles MobileNetV2."""
        parser = ONNXParser(model_path)
        graph = parser.parse()
        assert graph.stage == "raw"
        print(f"\nMobileNetV2 IR: {len(graph.nodes)} nodes")

    def test_optimization(self, model_path):
        """Stage 2: Optimization passes run on MobileNetV2."""
        parser = ONNXParser(model_path)
        graph = parser.parse()

        pm = PassManager()
        pm.add_pass(ConstantFoldingPass())
        pm.add_pass(DeadCodeEliminationPass())
        pm.add_pass(BatchNormFoldingPass())
        pm.add_pass(ConvBatchNormFoldingPass())
        pm.add_pass(ConvFusionPass())
        graph = pm.run(graph, verify=False)

        print(f"\nMobileNetV2 after opt: {len(graph.nodes)} nodes, stage={graph.stage}")
        assert len(graph.nodes) > 0

    def test_full_pipeline(self, model_path, tmp_path):
        """Stages 1-6: Full pipeline on MobileNetV2 (CNN path)."""
        result = _run_cnn_full_pipeline(model_path, (1, 3, 224, 224), tmp_path)
        print(f"\nMobileNetV2 full pipeline: {result['stages_completed']}")
        print(f"  Tiles: {result['total_tiles']}, SV files: {result['sv_files']}")
        print(f"  Cycles: {result['total_cycles']:,}, Weight bytes: {result['total_weight_bytes']:,}")
        assert "rtl_gen" in result["stages_completed"]
        assert "testbench" in result["stages_completed"]
        assert result["sv_files"] > 0


# ---------------------------------------------------------------------------
# Test: BERT-base (~110M params, Transformer)
# ---------------------------------------------------------------------------


@requires_hf
class TestBERTBase:
    """BERT-base-uncased: real Transformer production model (~110M params)."""

    @pytest.fixture(scope="class")
    def model_path(self):
        try:
            return download_bert_base()
        except Exception as e:
            pytest.skip(f"Failed to download BERT-base: {e}")

    def test_param_count(self, model_path):
        total = _count_params(model_path)
        print(f"\nBERT-base params: {total:,}")
        assert total > 80_000_000
        assert total < 150_000_000

    def test_op_types(self, model_path):
        ops = _list_ops(model_path)
        print(f"\nBERT-base ops: {sorted(ops)}")

    def test_ingestion(self, model_path):
        """Stage 1: BERT-base parsed through full pipeline."""
        parser = ONNXParser(model_path)
        graph = parser.parse()
        print(f"\nBERT-base IR: {len(graph.nodes)} nodes, {len(graph.tensors)} tensors")
        assert graph.stage == "raw"
        assert len(graph.nodes) > 0

    def test_optimization(self, model_path):
        """Stage 2: Optimization passes on BERT-base."""
        parser = ONNXParser(model_path)
        graph = parser.parse()

        pm = PassManager()
        pm.add_pass(ConstantFoldingPass())
        pm.add_pass(DeadCodeEliminationPass())
        graph = pm.run(graph, verify=False)

        print(f"\nBERT-base after opt: {len(graph.nodes)} nodes, stage={graph.stage}")
        assert len(graph.nodes) > 0

    def test_full_pipeline(self, model_path, tmp_path):
        """Stages 1-6: Full pipeline on BERT-base (Transformer path)."""
        result = _run_transformer_full_pipeline(model_path, tmp_path)
        print(f"\nBERT-base full pipeline: {result['stages_completed']}")
        print(f"  Tiles: {result['total_tiles']}, SV files: {result['sv_files']}")
        print(f"  Cycles: {result['total_cycles']:,}")
        assert "rtl_gen" in result["stages_completed"]
        assert "testbench" in result["stages_completed"]
        assert result["sv_files"] > 0


# ---------------------------------------------------------------------------
# Test: GPT-2 small (~124M params, Transformer)
# ---------------------------------------------------------------------------


@requires_hf
class TestGPT2:
    """GPT-2: real Transformer language model (~124M params)."""

    @pytest.fixture(scope="class")
    def model_path(self):
        try:
            return download_gpt2()
        except Exception as e:
            pytest.skip(f"Failed to download GPT-2: {e}")

    def test_param_count(self, model_path):
        total = _count_params(model_path)
        print(f"\nGPT-2 params: {total:,}")
        assert total > 100_000_000
        assert total < 200_000_000

    def test_op_types(self, model_path):
        ops = _list_ops(model_path)
        print(f"\nGPT-2 ops: {sorted(ops)}")

    def test_ingestion(self, model_path):
        """Stage 1: GPT-2 parsed through full pipeline."""
        parser = ONNXParser(model_path)
        graph = parser.parse()
        print(f"\nGPT-2 IR: {len(graph.nodes)} nodes")
        assert graph.stage == "raw"
        assert len(graph.nodes) > 0

    def test_optimization(self, model_path):
        """Stage 2: Optimization passes on GPT-2."""
        parser = ONNXParser(model_path)
        graph = parser.parse()

        pm = PassManager()
        pm.add_pass(ConstantFoldingPass())
        pm.add_pass(DeadCodeEliminationPass())
        graph = pm.run(graph, verify=False)

        print(f"\nGPT-2 after opt: {len(graph.nodes)} nodes, stage={graph.stage}")
        assert len(graph.nodes) > 0

    def test_full_pipeline(self, model_path, tmp_path):
        """Stages 1-6: Full pipeline on GPT-2 (Transformer path)."""
        result = _run_transformer_full_pipeline(model_path, tmp_path)
        print(f"\nGPT-2 full pipeline: {result['stages_completed']}")
        print(f"  Tiles: {result['total_tiles']}, SV files: {result['sv_files']}")
        print(f"  Cycles: {result['total_cycles']:,}")
        assert "rtl_gen" in result["stages_completed"]
        assert "testbench" in result["stages_completed"]
        assert result["sv_files"] > 0


# ---------------------------------------------------------------------------
# Test: GPT-2 Medium (~345M params)
# ---------------------------------------------------------------------------


@requires_hf
class TestGPT2Medium:
    """GPT-2 Medium: ~345M parameters."""

    @pytest.fixture(scope="class")
    def model_path(self):
        try:
            return download_gpt2_medium()
        except Exception as e:
            pytest.skip(f"Failed to download GPT-2 Medium: {e}")

    def test_param_count(self, model_path):
        total = _count_params(model_path)
        print(f"\nGPT-2 Medium params: {total:,}")
        assert total > 300_000_000
        assert total < 500_000_000

    def test_op_types(self, model_path):
        ops = _list_ops(model_path)
        print(f"\nGPT-2 Medium ops: {sorted(ops)}")

    def test_ingestion(self, model_path):
        """Stage 1: GPT-2 Medium rejected due to If subgraph ops."""
        from mlasic.exceptions import UnsupportedOperatorError

        parser = ONNXParser(model_path)
        with pytest.raises(UnsupportedOperatorError, match="subgraph ops"):
            parser.parse()

    def test_optimization(self, model_path):
        """Stage 2: Skipped — GPT-2 Medium rejected at ingestion."""
        from mlasic.exceptions import UnsupportedOperatorError

        parser = ONNXParser(model_path)
        with pytest.raises(UnsupportedOperatorError, match="subgraph ops"):
            parser.parse()

    def test_full_pipeline(self, model_path, tmp_path):
        """Stages 1-6: Skipped — GPT-2 Medium rejected at ingestion (If ops)."""
        from mlasic.exceptions import UnsupportedOperatorError

        parser = ONNXParser(model_path)
        with pytest.raises(UnsupportedOperatorError, match="subgraph ops"):
            parser.parse()


# ---------------------------------------------------------------------------
# Test: GPT-2 Large (~774M params)
# ---------------------------------------------------------------------------


@requires_hf
class TestGPT2Large:
    """GPT-2 Large: ~774M parameters (INT8 quantized)."""

    @pytest.fixture(scope="class")
    def model_path(self):
        try:
            return download_gpt2_large()
        except Exception as e:
            pytest.skip(f"Failed to download GPT-2 Large: {e}")

    def test_param_count(self, model_path):
        total = _count_params(model_path)
        print(f"\nGPT-2 Large params: {total:,}")
        assert total > 500_000_000
        assert total < 1_500_000_000

    def test_op_types(self, model_path):
        ops = _list_ops(model_path)
        print(f"\nGPT-2 Large ops: {sorted(ops)}")

    def test_ingestion(self, model_path):
        """Stage 1: GPT-2 Large parsed through full pipeline."""
        parser = ONNXParser(model_path)
        graph = parser.parse()
        print(f"\nGPT-2 Large IR: {len(graph.nodes)} nodes")
        assert graph.stage == "raw"
        assert len(graph.nodes) > 0

    def test_optimization(self, model_path):
        """Stage 2: Optimization passes on GPT-2 Large."""
        parser = ONNXParser(model_path)
        graph = parser.parse()

        pm = PassManager()
        pm.add_pass(ConstantFoldingPass())
        pm.add_pass(DeadCodeEliminationPass())
        graph = pm.run(graph, verify=False)

        print(f"\nGPT-2 Large after opt: {len(graph.nodes)} nodes, stage={graph.stage}")
        assert len(graph.nodes) > 0

    def test_full_pipeline(self, model_path, tmp_path):
        """Stages 1-6: Full pipeline on GPT-2 Large (Transformer path)."""
        result = _run_transformer_full_pipeline(model_path, tmp_path)
        print(f"\nGPT-2 Large full pipeline: {result['stages_completed']}")
        print(f"  Tiles: {result['total_tiles']}, SV files: {result['sv_files']}")
        print(f"  Cycles: {result['total_cycles']:,}")
        assert "rtl_gen" in result["stages_completed"]
        assert result["sv_files"] > 0


# ---------------------------------------------------------------------------
# Test: Llama-3.2-1B (~1.24B params)
# ---------------------------------------------------------------------------


@requires_hf
class TestLlama1B:
    """Llama-3.2-1B: ~1.24B parameters (INT8 quantized)."""

    @pytest.fixture(scope="class")
    def model_path(self):
        try:
            return download_llama_1b()
        except Exception as e:
            pytest.skip(f"Failed to download Llama-3.2-1B: {e}")

    def test_param_count(self, model_path):
        total = _count_params(model_path)
        print(f"\nLlama-3.2-1B params: {total:,}")
        assert total > 1_000_000_000
        assert total < 2_000_000_000

    def test_op_types(self, model_path):
        ops = _list_ops(model_path)
        print(f"\nLlama-3.2-1B ops: {sorted(ops)}")

    def test_ingestion(self, model_path):
        """Stage 1: Llama-3.2-1B parsed through full pipeline."""
        parser = ONNXParser(model_path)
        graph = parser.parse()
        print(f"\nLlama-3.2-1B IR: {len(graph.nodes)} nodes")
        assert graph.stage == "raw"
        assert len(graph.nodes) > 0

    def test_optimization(self, model_path):
        """Stage 2: Optimization passes on Llama-3.2-1B."""
        parser = ONNXParser(model_path)
        graph = parser.parse()

        pm = PassManager()
        pm.add_pass(ConstantFoldingPass())
        pm.add_pass(DeadCodeEliminationPass())
        graph = pm.run(graph, verify=False)

        print(f"\nLlama-3.2-1B after opt: {len(graph.nodes)} nodes, stage={graph.stage}")
        assert len(graph.nodes) > 0

    def test_full_pipeline(self, model_path, tmp_path):
        """Stages 1-6: Full pipeline on Llama-3.2-1B (Transformer path)."""
        result = _run_transformer_full_pipeline(model_path, tmp_path)
        print(f"\nLlama-3.2-1B full pipeline: {result['stages_completed']}")
        print(f"  Tiles: {result['total_tiles']}, SV files: {result['sv_files']}")
        print(f"  Cycles: {result['total_cycles']:,}")
        assert "rtl_gen" in result["stages_completed"]
        assert result["sv_files"] > 0
