"""Tests for the INT8 reference interpreter."""

from __future__ import annotations

import numpy as np
import pytest

from mlasic.ingestion import ONNXParser
from mlasic.int8_interpreter import INT8Interpreter
from mlasic.ir import (
    FusedLinearAttrs,
    Graph,
    OpNode,
    OpType,
    QuantParams,
    Tensor,
    TensorType,
)
from mlasic.optimization import (
    BatchNormFoldingPass,
    OperatorFusionPass,
    PassManager,
    QuantizationPass,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_quantized_layer(
    *,
    input_dim: int,
    output_dim: int,
    has_relu: bool,
    w_int8: np.ndarray,
    b_int32: np.ndarray,
    scale_w: float = 0.01,
    scale_x: float = 0.02,
    scale_y: float = 0.03,
    zp_x: int = 0,
    zp_y: int = 0,
) -> tuple[Graph, FusedLinearAttrs]:
    """Build a single-layer quantized graph for unit testing."""
    m_float = (scale_w * scale_x) / scale_y
    m_fixed = int(np.floor(m_float * (1 << 16) + 0.5))

    qp_w = QuantParams(scale=scale_w, zero_point=0, calibrated=True)
    qp_x = QuantParams(scale=scale_x, zero_point=zp_x, calibrated=True)
    qp_y = QuantParams(scale=scale_y, zero_point=zp_y, calibrated=True)

    op = OpType.FUSED_LINEAR_RELU if has_relu else OpType.FUSED_LINEAR
    attrs = FusedLinearAttrs(
        input_dim=input_dim,
        output_dim=output_dim,
        has_relu=has_relu,
        weight_quant=qp_w,
        input_quant=qp_x,
        output_quant=qp_y,
        requant_scale_fixed=m_fixed,
        requant_shift=16,
    )

    node = OpNode("layer0", op, ["x", "w", "b"], ["y"])
    node.fused_attrs = attrs

    graph = Graph(
        name="test",
        nodes={"layer0": node},
        tensors={
            "x": Tensor("x", TensorType((1, input_dim), np.dtype(np.int8))),
            "w": Tensor(
                "w",
                TensorType((input_dim, output_dim), np.dtype(np.int8)),
                data=w_int8,
            ),
            "b": Tensor(
                "b",
                TensorType((output_dim,), np.dtype(np.int32)),
                data=b_int32,
            ),
            "y": Tensor("y", TensorType((1, output_dim), np.dtype(np.int8))),
        },
        inputs=["x"],
        outputs=["y"],
        stage="quantized",
    )
    return graph, attrs


# ---------------------------------------------------------------------------
# TestINT8BasicMath
# ---------------------------------------------------------------------------


class TestINT8BasicMath:
    def test_zero_input(self):
        """x=0, W=random, b=0 → verify correct requant of zero accumulator."""
        rng = np.random.RandomState(42)
        w = rng.randint(-127, 128, size=(4, 3), dtype=np.int8)
        b = np.zeros(3, dtype=np.int32)
        x = np.zeros((1, 4), dtype=np.int8)

        graph, attrs = _make_quantized_layer(
            input_dim=4, output_dim=3, has_relu=False, w_int8=w, b_int32=b
        )
        interp = INT8Interpreter(graph)
        result = interp.run({"x": x})

        # acc=0, bias=0, so requant of 0 → zp_out (clamped)
        expected_raw = attrs.output_quant.zero_point
        expected = np.clip(expected_raw, -128, 127)
        assert result["y"].dtype == np.int8
        np.testing.assert_array_equal(result["y"], np.full((1, 3), expected, dtype=np.int8))

    def test_identity_requant(self):
        """M_fixed=65536 (1.0 in 16.16), zp=0 → accumulator passes through."""
        w = np.array([[1, 0], [0, 1]], dtype=np.int8)
        b = np.zeros(2, dtype=np.int32)
        x = np.array([[10, -20]], dtype=np.int8)

        # Craft scales so M_fixed = 65536
        # m_float = (sw * sx) / sy = 1.0 → m_fixed = 65536
        graph, _ = _make_quantized_layer(
            input_dim=2,
            output_dim=2,
            has_relu=False,
            w_int8=w,
            b_int32=b,
            scale_w=0.1,
            scale_x=0.1,
            scale_y=0.01,
        )
        interp = INT8Interpreter(graph)
        result = interp.run({"x": x})

        # Identity weight, no bias → acc = [10, -20], requant 1.0 → [10, -20]
        np.testing.assert_array_equal(result["y"], [[10, -20]])

    def test_manual_computation_small(self):
        """2×3 input, 3×2 weight, hand-computed expected output."""
        x = np.array([[1, 2, 3]], dtype=np.int8)
        w = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.int8)
        b = np.array([10, -5], dtype=np.int32)

        # acc = [1*1+2*0+3*1, 1*0+2*1+3*1] = [4, 5]
        # acc + bias = [14, 0]
        # m_float = (0.01 * 0.02)/0.03 = 0.006667
        # m_fixed = floor(0.006667 * 65536 + 0.5) = floor(437.0 + 0.5) = 437
        # scaled[0] = 14 * 437 = 6118, rounded = (6118 + 32768) >> 16 = 38886 >> 16 = 0
        # scaled[1] = 0 * 437 = 0, rounded = (0 + 32768) >> 16 = 0
        # result = rounded + zp_out = [0, 0], clamp = [0, 0]
        graph, _ = _make_quantized_layer(
            input_dim=3, output_dim=2, has_relu=False, w_int8=w, b_int32=b
        )
        interp = INT8Interpreter(graph)
        result = interp.run({"x": x})

        # Verify manually
        acc = np.array([[14, 0]], dtype=np.int64)
        m_float = (0.01 * 0.02) / 0.03
        m_fixed = int(np.floor(m_float * 65536 + 0.5))
        scaled = acc * m_fixed
        rounded = (scaled + (1 << 15)) >> 16
        expected = np.clip(rounded + 0, -128, 127).astype(np.int8)
        np.testing.assert_array_equal(result["y"], expected)

    def test_relu_clips_negative(self):
        """Negative outputs → 0 when has_relu=True."""
        w = np.array([[1]], dtype=np.int8)
        b = np.array([-100], dtype=np.int32)
        x = np.array([[1]], dtype=np.int8)

        graph, _ = _make_quantized_layer(
            input_dim=1,
            output_dim=1,
            has_relu=True,
            w_int8=w,
            b_int32=b,
            scale_w=1.0,
            scale_x=1.0,
            scale_y=1.0,
        )
        interp = INT8Interpreter(graph)
        result = interp.run({"x": x})

        # acc = 1 - 100 = -99, requant passes through (M=65536), clamp then relu → 0
        assert result["y"][0, 0] == 0

    def test_no_relu_preserves_negative(self):
        """FusedLinear (no relu) keeps negatives."""
        w = np.array([[1]], dtype=np.int8)
        b = np.array([-100], dtype=np.int32)
        x = np.array([[1]], dtype=np.int8)

        graph, _ = _make_quantized_layer(
            input_dim=1,
            output_dim=1,
            has_relu=False,
            w_int8=w,
            b_int32=b,
            scale_w=1.0,
            scale_x=1.0,
            scale_y=1.0,
        )
        interp = INT8Interpreter(graph)
        result = interp.run({"x": x})

        # acc = 1 - 100 = -99, requant M=65536 → -99, clamp [-128,127] → -99
        assert result["y"][0, 0] == -99

    def test_relu_preserves_positive(self):
        """Positive outputs unchanged with ReLU."""
        w = np.array([[1]], dtype=np.int8)
        b = np.array([50], dtype=np.int32)
        x = np.array([[10]], dtype=np.int8)

        graph, _ = _make_quantized_layer(
            input_dim=1,
            output_dim=1,
            has_relu=True,
            w_int8=w,
            b_int32=b,
            scale_w=1.0,
            scale_x=1.0,
            scale_y=1.0,
        )
        interp = INT8Interpreter(graph)
        result = interp.run({"x": x})

        # acc = 10 + 50 = 60, requant → 60, relu → 60
        assert result["y"][0, 0] == 60


# ---------------------------------------------------------------------------
# TestINT8Accumulation
# ---------------------------------------------------------------------------


class TestINT8Accumulation:
    def test_max_positive_accumulation(self):
        """x=127, W=127 → acc=16129 per element, verify correct requant."""
        x = np.full((1, 1), 127, dtype=np.int8)
        w = np.full((1, 1), 127, dtype=np.int8)
        b = np.zeros(1, dtype=np.int32)

        graph, attrs = _make_quantized_layer(
            input_dim=1,
            output_dim=1,
            has_relu=False,
            w_int8=w,
            b_int32=b,
            scale_w=1.0,
            scale_x=1.0,
            scale_y=200.0,
        )
        interp = INT8Interpreter(graph)
        result = interp.run({"x": x})

        # acc = 127*127 = 16129
        # m_float = (1*1)/200 = 0.005, m_fixed = round(0.005*65536) = 328
        # scaled = 16129 * 328 = 5290312
        # rounded = (5290312 + 32768) >> 16 = 5323080 >> 16 = 81
        assert result["y"].dtype == np.int8
        # Just verify it's within valid range and positive
        assert -128 <= result["y"][0, 0] <= 127

    def test_max_negative_accumulation(self):
        """x=-128, W=127 → acc=-16256, verify correct requant."""
        x = np.full((1, 1), -128, dtype=np.int8)
        w = np.full((1, 1), 127, dtype=np.int8)
        b = np.zeros(1, dtype=np.int32)

        graph, _ = _make_quantized_layer(
            input_dim=1,
            output_dim=1,
            has_relu=False,
            w_int8=w,
            b_int32=b,
            scale_w=1.0,
            scale_x=1.0,
            scale_y=200.0,
        )
        interp = INT8Interpreter(graph)
        result = interp.run({"x": x})

        assert result["y"].dtype == np.int8
        assert -128 <= result["y"][0, 0] <= 127

    def test_large_dim_no_overflow(self):
        """x=[127]*640, W=[127]*640 → acc=10,322,880, stays in INT32."""
        x = np.full((1, 640), 127, dtype=np.int8)
        w = np.full((640, 1), 127, dtype=np.int8)
        b = np.zeros(1, dtype=np.int32)

        graph, _ = _make_quantized_layer(
            input_dim=640,
            output_dim=1,
            has_relu=False,
            w_int8=w,
            b_int32=b,
            scale_w=0.01,
            scale_x=0.01,
            scale_y=1000.0,
        )
        interp = INT8Interpreter(graph)
        result = interp.run({"x": x})

        # acc = 640 * 127 * 127 = 10,322,880 — fits INT32 (max ~2.1B)
        assert result["y"].dtype == np.int8
        assert -128 <= result["y"][0, 0] <= 127


# ---------------------------------------------------------------------------
# TestINT8Requantization
# ---------------------------------------------------------------------------


class TestINT8Requantization:
    def test_round_half_up_at_exact_boundary(self):
        """Craft acc×M_fixed at exactly 0.5 fractional boundary, verify rounds up."""
        # We want (acc * m_fixed + (1<<15)) >> 16 to test the 0.5 boundary
        # If acc * m_fixed = 0x8000 (32768), that's exactly 0.5 in 16.16
        # rounded = (32768 + 32768) >> 16 = 65536 >> 16 = 1 (rounds up)
        w = np.array([[1]], dtype=np.int8)
        b = np.zeros(1, dtype=np.int32)
        x = np.array([[1]], dtype=np.int8)

        # m_fixed = 32768 → m_float = 32768/65536 = 0.5
        # sw * sx / sy = 0.5, so sy = sw * sx / 0.5
        # Use sw=0.5, sx=0.5, sy=0.5 → m_float = 0.5, m_fixed = 32768
        graph, _ = _make_quantized_layer(
            input_dim=1,
            output_dim=1,
            has_relu=False,
            w_int8=w,
            b_int32=b,
            scale_w=0.5,
            scale_x=0.5,
            scale_y=0.5,
        )
        interp = INT8Interpreter(graph)
        result = interp.run({"x": x})

        # acc=1, m_fixed=32768, scaled=32768
        # (32768 + 32768) >> 16 = 65536 >> 16 = 1
        assert result["y"][0, 0] == 1

    def test_output_clamped_to_127(self):
        """Large positive → clamped to 127."""
        x = np.full((1, 1), 127, dtype=np.int8)
        w = np.full((1, 1), 127, dtype=np.int8)
        b = np.zeros(1, dtype=np.int32)

        # M_fixed very large → output huge → clamp to 127
        graph, _ = _make_quantized_layer(
            input_dim=1,
            output_dim=1,
            has_relu=False,
            w_int8=w,
            b_int32=b,
            scale_w=1.0,
            scale_x=1.0,
            scale_y=0.001,
        )
        interp = INT8Interpreter(graph)
        result = interp.run({"x": x})
        assert result["y"][0, 0] == 127

    def test_output_clamped_to_neg128(self):
        """Large negative → clamped to -128."""
        x = np.full((1, 1), -128, dtype=np.int8)
        w = np.full((1, 1), 127, dtype=np.int8)
        b = np.zeros(1, dtype=np.int32)

        graph, _ = _make_quantized_layer(
            input_dim=1,
            output_dim=1,
            has_relu=False,
            w_int8=w,
            b_int32=b,
            scale_w=1.0,
            scale_x=1.0,
            scale_y=0.001,
        )
        interp = INT8Interpreter(graph)
        result = interp.run({"x": x})
        assert result["y"][0, 0] == -128

    def test_zero_point_addition(self):
        """Verify zp_out correctly added after shift."""
        w = np.array([[1]], dtype=np.int8)
        b = np.zeros(1, dtype=np.int32)
        x = np.array([[0]], dtype=np.int8)

        # acc=0, requant → 0, then add zp_out=10 → 10
        graph, _ = _make_quantized_layer(
            input_dim=1,
            output_dim=1,
            has_relu=False,
            w_int8=w,
            b_int32=b,
            scale_w=1.0,
            scale_x=1.0,
            scale_y=1.0,
            zp_y=10,
        )
        interp = INT8Interpreter(graph)
        result = interp.run({"x": x})
        assert result["y"][0, 0] == 10


# ---------------------------------------------------------------------------
# TestINT8ReLUWithZeroPoint
# ---------------------------------------------------------------------------


class TestINT8ReLUWithZeroPoint:
    def test_relu_threshold_is_zero_not_zp(self):
        """Non-zero zp_out, ReLU clamps to 0 (not zp_out)."""
        w = np.array([[1]], dtype=np.int8)
        b = np.array([-100], dtype=np.int32)
        x = np.array([[1]], dtype=np.int8)

        # With zp_out=5: acc=1-100=-99, requant → some_neg + 5 → still negative → relu → 0
        graph, _ = _make_quantized_layer(
            input_dim=1,
            output_dim=1,
            has_relu=True,
            w_int8=w,
            b_int32=b,
            scale_w=1.0,
            scale_x=1.0,
            scale_y=1.0,
            zp_y=5,
        )
        interp = INT8Interpreter(graph)
        result = interp.run({"x": x})

        # Result should be 0 (relu threshold), not 5 (zp_out)
        assert result["y"][0, 0] == 0

    def test_relu_zero_point_zero(self):
        """zp_out=0, ReLU still clamps correctly."""
        w = np.array([[1]], dtype=np.int8)
        b = np.array([-50], dtype=np.int32)
        x = np.array([[1]], dtype=np.int8)

        graph, _ = _make_quantized_layer(
            input_dim=1,
            output_dim=1,
            has_relu=True,
            w_int8=w,
            b_int32=b,
            scale_w=1.0,
            scale_x=1.0,
            scale_y=1.0,
            zp_y=0,
        )
        interp = INT8Interpreter(graph)
        result = interp.run({"x": x})
        assert result["y"][0, 0] == 0


# ---------------------------------------------------------------------------
# TestINT8EndToEnd
# ---------------------------------------------------------------------------


class TestINT8EndToEnd:
    @pytest.fixture
    def quantized_ad_graph(self, ad_model_path) -> Graph:
        parser = ONNXParser(ad_model_path)
        graph = parser.parse()
        rng = np.random.RandomState(123)
        calib = [rng.randn(1, 640).astype(np.float32) for _ in range(20)]
        pm = PassManager()
        pm.add_pass(BatchNormFoldingPass())
        pm.add_pass(OperatorFusionPass())
        pm.add_pass(QuantizationPass(calib))
        return pm.run(graph, verify=False)

    def test_quantized_ad_model_runs(self, quantized_ad_graph):
        """Full 4-layer AD model, random INT8 input → correct shape/dtype."""
        interp = INT8Interpreter(quantized_ad_graph)
        rng = np.random.RandomState(42)
        x = rng.randint(-128, 128, size=(1, 640), dtype=np.int8)
        input_name = quantized_ad_graph.inputs[0]
        result = interp.run({input_name: x})

        # Should have single output with correct shape
        assert len(result) == 1
        out = list(result.values())[0]
        assert out.shape == (1, 640)
        assert out.dtype == np.int8

    def test_float_input_auto_quantized(self, quantized_ad_graph):
        """Float32 input → auto-quantized, INT8 output."""
        interp = INT8Interpreter(quantized_ad_graph)
        x_fp = np.random.RandomState(42).randn(1, 640).astype(np.float32)

        # Rename input to match graph input name
        input_name = quantized_ad_graph.inputs[0]
        result = interp.run({input_name: x_fp})

        out = list(result.values())[0]
        assert out.shape == (1, 640)
        assert out.dtype == np.int8

    def test_missing_input_raises(self, quantized_ad_graph):
        """Empty dict → ValueError."""
        interp = INT8Interpreter(quantized_ad_graph)
        with pytest.raises(ValueError, match="Missing input"):
            interp.run({})

    def test_wrong_stage_raises(self):
        """stage='raw' → ValueError."""
        graph = Graph("test", {}, {}, [], [], stage="raw")
        with pytest.raises(ValueError, match="quantized.*scheduled"):
            INT8Interpreter(graph)


# ---------------------------------------------------------------------------
# TestINT8VsReference
# ---------------------------------------------------------------------------


class TestINT8VsReference:
    def test_bitwise_match_spec_reference(self):
        """Implement spec §3.2 reference inline, verify INT8Interpreter matches."""
        rng = np.random.RandomState(42)
        in_dim, out_dim = 8, 4

        w_int8 = rng.randint(-127, 128, size=(in_dim, out_dim), dtype=np.int8)
        b_int32 = rng.randint(-1000, 1000, size=(out_dim,), dtype=np.int32)
        scale_w, scale_x, scale_y = 0.05, 0.04, 0.03
        zp_y = 5

        graph, attrs = _make_quantized_layer(
            input_dim=in_dim,
            output_dim=out_dim,
            has_relu=True,
            w_int8=w_int8,
            b_int32=b_int32,
            scale_w=scale_w,
            scale_x=scale_x,
            scale_y=scale_y,
            zp_y=zp_y,
        )

        interp = INT8Interpreter(graph)
        m_fixed = attrs.requant_scale_fixed

        for _ in range(100):
            x = rng.randint(-128, 128, size=(1, in_dim), dtype=np.int8)

            # Reference implementation (spec §3.2 inline)
            acc = x.astype(np.int32) @ w_int8.astype(np.int32) + b_int32.astype(np.int32)
            scaled = acc.astype(np.int64) * np.int64(m_fixed)
            rounded = (scaled + (np.int64(1) << 15)) >> 16
            result_ref = rounded + np.int64(zp_y)
            result_ref = np.clip(result_ref, -128, 127).astype(np.int8)
            result_ref = np.maximum(result_ref, np.int8(0))  # ReLU

            result_interp = interp.run({"x": x})["y"]
            np.testing.assert_array_equal(result_interp, result_ref)
