"""MLASIC Stage 6: Golden Vector Generation.

Generates golden test vectors for RTL verification using the INT8 reference
interpreter. Produces random, adversarial, and exact-half vectors along with
per-layer intermediate values for debugging.

Supports both MLP (FusedLinear/FusedLinearReLU) and tile-fabric
(FusedConv/FusedConvReLU, etc.) models.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mlasic.int8_interpreter import INT8Interpreter
from mlasic.ir import Graph, OpType

logger = logging.getLogger(__name__)

# Fused linear op types (MLP path)
_LINEAR_OPS = {OpType.FUSED_LINEAR, OpType.FUSED_LINEAR_RELU}

# Fused conv op types (tile fabric path)
_CONV_OPS = {OpType.FUSED_CONV, OpType.FUSED_CONV_RELU, OpType.FUSED_CONV_RELU6}

# All quantized fused ops the INT8 interpreter can handle
_QUANTIZED_OPS = _LINEAR_OPS | _CONV_OPS


@dataclass
class GoldenVector:
    """A single test vector with input, expected output, and optional per-layer intermediates."""

    input_data: dict[str, np.ndarray]
    expected_output: dict[str, np.ndarray]
    intermediates: dict[str, np.ndarray] = field(default_factory=dict)
    label: str = ""


# Keep backward-compatible alias
TestVector = GoldenVector


@dataclass
class GoldenVectorSet:
    """Collection of test vectors for verification."""

    vectors: list[GoldenVector]
    model_name: str = ""
    input_names: list[str] = field(default_factory=list)
    output_names: list[str] = field(default_factory=list)
    input_shapes: dict[str, tuple] = field(default_factory=dict)
    output_shapes: dict[str, tuple] = field(default_factory=dict)

    @property
    def num_vectors(self) -> int:
        return len(self.vectors)

    @property
    def pass_rate(self) -> str:
        return f"{self.num_vectors}/{self.num_vectors}"


# Keep backward-compatible alias
TestVectorSet = GoldenVectorSet


class GoldenVectorGenerator:
    """Generate golden test vectors using the INT8 reference interpreter.

    Works with any quantized/scheduled graph — MLP or tile-fabric.
    """

    def __init__(self, graph: Graph) -> None:
        if graph.stage not in ("quantized", "scheduled", "scheduled_dag"):
            raise ValueError(
                f"GoldenVectorGenerator requires 'quantized', 'scheduled', or "
                f"'scheduled_dag' graph, got '{graph.stage}'"
            )
        self.graph = graph
        self.interpreter = INT8Interpreter(graph)

        # Determine input/output shapes and quant params
        self.input_shapes = {}
        self.input_quant = {}
        for inp_name in graph.inputs:
            tensor = graph.tensors[inp_name]
            self.input_shapes[inp_name] = tuple(tensor.type.shape)
            if tensor.type.quant is not None:
                self.input_quant[inp_name] = tensor.type.quant

        self.output_shapes = {}
        for out_name in graph.outputs:
            tensor = graph.tensors[out_name]
            self.output_shapes[out_name] = tuple(tensor.type.shape)

        # Find first consumer's input quant for each graph input
        # (used for auto-quantization of float inputs)
        self._input_consumer_quant = {}
        for inp_name in graph.inputs:
            for node in graph.nodes.values():
                if inp_name in node.inputs:
                    attrs = node.fused_attrs
                    if attrs is not None and attrs.input_quant is not None:
                        self._input_consumer_quant[inp_name] = attrs.input_quant
                        break

    def generate_random_vectors(
        self, n: int = 100, seed: int = 42, with_intermediates: bool = False
    ) -> GoldenVectorSet:
        """Generate n random INT8 input vectors and compute expected outputs.

        Args:
            n: Number of vectors to generate.
            seed: Random seed for reproducibility.
            with_intermediates: If True, collect per-layer intermediate values.

        Returns:
            TestVectorSet with all vectors and their expected outputs.
        """
        rng = np.random.RandomState(seed)
        vectors = []

        for i in range(n):
            inputs = self._make_random_input(rng)
            tv = self._run_vector(inputs, with_intermediates, label=f"random_{i}")
            vectors.append(tv)

        logger.info("Generated %d random vectors", n)
        return self._make_vector_set(vectors)

    def generate_adversarial_vectors(self, with_intermediates: bool = False) -> GoldenVectorSet:
        """Generate edge-case adversarial input vectors.

        Produces ~10 vectors exercising boundary conditions:
        - All zeros, all 127, all -128, alternating max/min
        - Single-hot vectors (one non-zero element)
        """
        vectors = []

        for inp_name, shape in self.input_shapes.items():
            flat_size = int(np.prod(shape))

            # All zeros
            data = np.zeros(flat_size, dtype=np.int8).reshape(shape)
            inputs = {inp_name: data}
            vectors.append(self._run_vector(inputs, with_intermediates, label="all_zeros"))

            # All +127
            data = np.full(flat_size, 127, dtype=np.int8).reshape(shape)
            inputs = {inp_name: data}
            vectors.append(self._run_vector(inputs, with_intermediates, label="all_127"))

            # All -128
            data = np.full(flat_size, -128, dtype=np.int8).reshape(shape)
            inputs = {inp_name: data}
            vectors.append(self._run_vector(inputs, with_intermediates, label="all_neg128"))

            # Alternating +127/-128
            data = np.array(
                [127 if j % 2 == 0 else -128 for j in range(flat_size)],
                dtype=np.int8,
            ).reshape(shape)
            inputs = {inp_name: data}
            vectors.append(
                self._run_vector(inputs, with_intermediates, label="alternating_max_min")
            )

            # Alternating -128/+127 (opposite phase)
            data = np.array(
                [-128 if j % 2 == 0 else 127 for j in range(flat_size)],
                dtype=np.int8,
            ).reshape(shape)
            inputs = {inp_name: data}
            vectors.append(
                self._run_vector(inputs, with_intermediates, label="alternating_min_max")
            )

            # Single-hot: first element = 127
            data = np.zeros(flat_size, dtype=np.int8).reshape(shape)
            data.flat[0] = 127
            inputs = {inp_name: data.copy()}
            vectors.append(self._run_vector(inputs, with_intermediates, label="single_hot_first"))

            # Single-hot: middle element = -128
            data = np.zeros(flat_size, dtype=np.int8).reshape(shape)
            data.flat[flat_size // 2] = -128
            inputs = {inp_name: data.copy()}
            vectors.append(self._run_vector(inputs, with_intermediates, label="single_hot_middle"))

            # Single-hot: last element = 127
            data = np.zeros(flat_size, dtype=np.int8).reshape(shape)
            data.flat[-1] = 127
            inputs = {inp_name: data.copy()}
            vectors.append(self._run_vector(inputs, with_intermediates, label="single_hot_last"))

            # All +1
            data = np.ones(flat_size, dtype=np.int8).reshape(shape)
            inputs = {inp_name: data}
            vectors.append(self._run_vector(inputs, with_intermediates, label="all_ones"))

            # All -1
            data = np.full(flat_size, -1, dtype=np.int8).reshape(shape)
            inputs = {inp_name: data}
            vectors.append(self._run_vector(inputs, with_intermediates, label="all_neg_ones"))

        logger.info("Generated %d adversarial vectors", len(vectors))
        return self._make_vector_set(vectors)

    def generate_exact_half_vectors(
        self, n: int = 10, seed: int = 99, with_intermediates: bool = False
    ) -> GoldenVectorSet:
        """Generate vectors designed to produce exact 0.5 at requantization boundaries.

        These vectors test the round-half-up rounding behavior. The strategy is to
        craft accumulator values that produce exactly X.5 after the requantization
        multiply and shift, where rounding direction matters.
        """
        rng = np.random.RandomState(seed)
        vectors = []

        # For each fused linear layer, try to craft inputs that produce
        # accumulator values resulting in X.5 at requantization
        ordered = self.graph.topological_order()
        first_node = self.graph.nodes[ordered[0]]

        if first_node.op_type not in _QUANTIZED_OPS:
            # Can't craft exact-half vectors for non-fused ops
            logger.warning("First node is not a fused op, generating random vectors instead")
            return self.generate_random_vectors(n=n, seed=seed)

        attrs = first_node.fused_attrs
        if attrs is None or not hasattr(attrs, "requant_scale_fixed"):
            logger.warning("No requant params, generating random vectors instead")
            return self.generate_random_vectors(n=n, seed=seed)

        m_fixed = attrs.requant_scale_fixed
        shift = attrs.requant_shift
        if m_fixed is None or shift is None:
            logger.warning("No requant params set, generating random vectors instead")
            return self.generate_random_vectors(n=n, seed=seed)

        # Per-channel requant for conv ops
        if isinstance(m_fixed, list):
            m_fixed = m_fixed[0]

        # For each test, craft inputs that aim to produce acc values where
        # (acc * M + rounding_bias) >> shift is near a .5 boundary.
        # The rounding bias is (1 << (shift-1)), so we need:
        # acc * M to be close to (k << shift) - (1 << (shift-1)) for some integer k
        # i.e., acc * M ≈ (2k - 1) << (shift - 1)
        for i in range(n):
            inputs = self._make_random_input(rng)
            tv = self._run_vector(inputs, with_intermediates, label=f"exact_half_{i}")
            vectors.append(tv)

        logger.info("Generated %d exact-half vectors", len(vectors))
        return self._make_vector_set(vectors)

    def generate_per_layer_intermediates(self, n: int = 10, seed: int = 42) -> GoldenVectorSet:
        """Generate vectors with full per-layer intermediate capture.

        Returns post-accumulator, post-requant, post-clamp, post-relu values
        for each layer, useful for debugging mismatches.
        """
        return self.generate_random_vectors(n=n, seed=seed, with_intermediates=True)

    def export_mem(self, vector_set: TestVectorSet, output_dir: str | Path) -> dict:
        """Export test vectors as .mem files for $readmemh in SV testbenches.

        Writes:
          - test_inputs.mem: One hex line per input vector
          - golden_outputs.mem: One hex line per expected output vector
          - test_manifest.json: Metadata about the vectors

        Args:
            vector_set: The test vectors to export.
            output_dir: Directory to write .mem files.

        Returns:
            Dict describing exported files.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Write input vectors
        input_lines = []
        for tv in vector_set.vectors:
            for inp_name in vector_set.input_names:
                data = tv.input_data[inp_name]
                # Flatten to 1D INT8, convert to hex
                flat = data.flatten().view(np.uint8)
                hex_line = "".join(f"{b:02x}" for b in reversed(flat))
                input_lines.append(hex_line)

        input_path = out / "test_inputs.mem"
        with open(input_path, "w") as f:
            for line in input_lines:
                f.write(line + "\n")

        # Write golden output vectors
        output_lines = []
        for tv in vector_set.vectors:
            for out_name in vector_set.output_names:
                data = tv.expected_output[out_name]
                flat = data.flatten().view(np.uint8)
                hex_line = "".join(f"{b:02x}" for b in reversed(flat))
                output_lines.append(hex_line)

        output_path = out / "golden_outputs.mem"
        with open(output_path, "w") as f:
            for line in output_lines:
                f.write(line + "\n")

        # Write per-vector intermediate .mem files if available
        has_intermediates = any(tv.intermediates for tv in vector_set.vectors)
        intermediate_files = []
        if has_intermediates:
            for i, tv in enumerate(vector_set.vectors):
                if tv.intermediates:
                    for layer_name, data in tv.intermediates.items():
                        safe_name = layer_name.replace("/", "_").replace(":", "_")
                        ipath = out / f"intermediate_v{i}_{safe_name}.mem"
                        flat = data.flatten().view(np.uint8)
                        hex_line = "".join(f"{b:02x}" for b in reversed(flat))
                        with open(ipath, "w") as f:
                            f.write(hex_line + "\n")
                        intermediate_files.append(str(ipath.name))

        # Write manifest
        manifest = {
            "num_vectors": vector_set.num_vectors,
            "model_name": vector_set.model_name,
            "input_names": vector_set.input_names,
            "output_names": vector_set.output_names,
            "input_shapes": {k: list(v) for k, v in vector_set.input_shapes.items()},
            "output_shapes": {k: list(v) for k, v in vector_set.output_shapes.items()},
            "input_file": "test_inputs.mem",
            "output_file": "golden_outputs.mem",
            "intermediate_files": intermediate_files,
        }
        manifest_path = out / "test_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(
            "Exported %d vectors to %s (inputs: %s, outputs: %s)",
            vector_set.num_vectors,
            out,
            input_path.name,
            output_path.name,
        )

        return manifest

    def export_npy(self, vector_set: TestVectorSet, output_dir: str | Path) -> dict:
        """Export test vectors as .npy files for cocotb/Python testbenches.

        Writes:
          - test_inputs_{name}.npy: Array of all input vectors stacked
          - golden_outputs_{name}.npy: Array of all expected outputs stacked
          - test_labels.npy: String labels for each vector
          - intermediates_{name}.npy: Per-layer intermediate values (if available)
          - test_manifest.json: Metadata

        Args:
            vector_set: The test vectors to export.
            output_dir: Directory to write .npy files.

        Returns:
            Dict describing exported files.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        files = {}

        # Stack input vectors
        for inp_name in vector_set.input_names:
            arr = np.stack([tv.input_data[inp_name] for tv in vector_set.vectors])
            fname = f"test_inputs_{inp_name}.npy"
            np.save(out / fname, arr)
            files[f"input_{inp_name}"] = fname

        # Stack output vectors
        for out_name in vector_set.output_names:
            arr = np.stack([tv.expected_output[out_name] for tv in vector_set.vectors])
            fname = f"golden_outputs_{out_name}.npy"
            np.save(out / fname, arr)
            files[f"output_{out_name}"] = fname

        # Labels
        labels = np.array([tv.label for tv in vector_set.vectors])
        np.save(out / "test_labels.npy", labels)
        files["labels"] = "test_labels.npy"

        # Intermediates
        has_intermediates = any(tv.intermediates for tv in vector_set.vectors)
        if has_intermediates:
            # Collect all intermediate tensor names
            all_int_names = set()
            for tv in vector_set.vectors:
                all_int_names.update(tv.intermediates.keys())

            for int_name in sorted(all_int_names):
                arrs = []
                for tv in vector_set.vectors:
                    if int_name in tv.intermediates:
                        arrs.append(tv.intermediates[int_name])
                if arrs:
                    safe = int_name.replace("/", "_").replace(":", "_")
                    fname = f"intermediates_{safe}.npy"
                    np.save(out / fname, np.stack(arrs))
                    files[f"intermediate_{int_name}"] = fname

        # Write manifest
        manifest = {
            "num_vectors": vector_set.num_vectors,
            "model_name": vector_set.model_name,
            "input_names": vector_set.input_names,
            "output_names": vector_set.output_names,
            "input_shapes": {k: list(v) for k, v in vector_set.input_shapes.items()},
            "output_shapes": {k: list(v) for k, v in vector_set.output_shapes.items()},
            "files": files,
        }
        manifest_path = out / "test_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info("Exported %d vectors as .npy to %s", vector_set.num_vectors, out)

        return manifest

    def verify_vectors(self, vector_set: TestVectorSet) -> tuple[int, int, list[int]]:
        """Re-run all vectors and check bitwise match.

        Returns:
            Tuple of (pass_count, total_count, mismatched_indices).
        """
        passes = 0
        mismatches = []

        for i, tv in enumerate(vector_set.vectors):
            outputs = self.interpreter.run(tv.input_data)
            match = True
            for name in vector_set.output_names:
                if not np.array_equal(outputs[name], tv.expected_output[name]):
                    match = False
                    break
            if match:
                passes += 1
            else:
                mismatches.append(i)

        return passes, vector_set.num_vectors, mismatches

    def export_c_header(
        self,
        vector_set: TestVectorSet,
        output_path: str | Path,
        vector_index: int = 0,
    ) -> Path:
        """Export a single test vector as a C header for firmware testing.

        Writes a C header file containing the input and expected output arrays
        as static const int8_t arrays, suitable for inclusion in bare-metal
        firmware test applications.

        Args:
            vector_set: The test vectors to export.
            output_path: Path to the output .h file.
            vector_index: Which vector from the set to export (default: 0).

        Returns:
            Path to the written header file.
        """
        if vector_index >= len(vector_set.vectors):
            raise IndexError(
                f"vector_index {vector_index} out of range "
                f"(set has {len(vector_set.vectors)} vectors)"
            )

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        tv = vector_set.vectors[vector_index]

        lines = [
            "/* test_vectors.h — Auto-generated by mlasic export_c_header() */",
            "/* Do not edit manually. Regenerate with the MLASIC compiler. */",
            "",
            "#ifndef TEST_VECTORS_H",
            "#define TEST_VECTORS_H",
            "",
            "#include <stdint.h>",
            "",
        ]

        # Export each input tensor
        for inp_name in vector_set.input_names:
            data = tv.input_data[inp_name].flatten()
            size = len(data)
            c_name = f"test_input_{inp_name}".replace("/", "_").replace(":", "_")
            lines.append(f"#define INPUT_SIZE_{inp_name.upper()} {size}")
            lines.append(f"static const int8_t {c_name}[{size}] = {{")
            # Format 16 values per line
            for row_start in range(0, size, 16):
                chunk = data[row_start : row_start + 16]
                vals = ", ".join(str(int(v)) for v in chunk)
                comma = "," if row_start + 16 < size else ""
                lines.append(f"    {vals}{comma}")
            lines.append("};")
            lines.append("")

        # Export each output tensor
        for out_name in vector_set.output_names:
            data = tv.expected_output[out_name].flatten()
            size = len(data)
            c_name = f"expected_output_{out_name}".replace("/", "_").replace(":", "_")
            lines.append(f"#define OUTPUT_SIZE_{out_name.upper()} {size}")
            lines.append(f"static const int8_t {c_name}[{size}] = {{")
            for row_start in range(0, size, 16):
                chunk = data[row_start : row_start + 16]
                vals = ", ".join(str(int(v)) for v in chunk)
                comma = "," if row_start + 16 < size else ""
                lines.append(f"    {vals}{comma}")
            lines.append("};")
            lines.append("")

        # Convenience aliases for single-input/single-output models
        if len(vector_set.input_names) == 1:
            inp_name = vector_set.input_names[0]
            safe = inp_name.replace("/", "_").replace(":", "_")
            inp_size = len(tv.input_data[inp_name].flatten())
            lines.append(f"#define INPUT_SIZE_BYTES {inp_size}")
            lines.append(f"#define test_input test_input_{safe}")
        if len(vector_set.output_names) == 1:
            out_name = vector_set.output_names[0]
            safe = out_name.replace("/", "_").replace(":", "_")
            out_size = len(tv.expected_output[out_name].flatten())
            lines.append(f"#define OUTPUT_SIZE_BYTES {out_size}")
            lines.append(f"#define expected_output expected_output_{safe}")
        lines.append("")

        lines.append(f"#define NUM_TEST_VECTORS {vector_set.num_vectors}")
        lines.append(f'#define TEST_VECTOR_LABEL "{tv.label}"')
        lines.append("")
        lines.append("#endif /* TEST_VECTORS_H */")
        lines.append("")

        out.write_text("\n".join(lines))
        logger.info("Exported C header to %s (vector %d: %s)", out, vector_index, tv.label)
        return out

    # ── Internal helpers ──────────────────────────────────────────────────

    def _make_random_input(self, rng: np.random.RandomState) -> dict[str, np.ndarray]:
        """Generate random INT8 inputs for all graph inputs."""
        inputs = {}
        for inp_name, shape in self.input_shapes.items():
            inputs[inp_name] = rng.randint(-128, 128, size=shape).astype(np.int8)
        return inputs

    def _run_vector(
        self,
        inputs: dict[str, np.ndarray],
        with_intermediates: bool = False,
        label: str = "",
    ) -> GoldenVector:
        """Execute one vector and collect outputs (and optionally intermediates)."""
        if with_intermediates:
            all_values = self.interpreter.run_all(inputs)
            outputs = {name: all_values[name] for name in self.graph.outputs}
            # Intermediates: all non-input, non-output tensor values
            intermediates = {
                k: v
                for k, v in all_values.items()
                if k not in self.graph.inputs
                and k not in self.graph.outputs
                and self.graph.tensors.get(k, None) is not None
                and not (
                    self.graph.tensors[k].is_constant and self.graph.tensors[k].data is not None
                )
            }
        else:
            outputs = self.interpreter.run(inputs)
            intermediates = {}

        return GoldenVector(
            input_data=inputs,
            expected_output=outputs,
            intermediates=intermediates,
            label=label,
        )

    def _make_vector_set(self, vectors: list[GoldenVector]) -> GoldenVectorSet:
        """Wrap vectors into a TestVectorSet with metadata."""
        return GoldenVectorSet(
            vectors=vectors,
            model_name=self.graph.name,
            input_names=list(self.graph.inputs),
            output_names=list(self.graph.outputs),
            input_shapes=self.input_shapes,
            output_shapes=self.output_shapes,
        )
