"""MLASIC compiler exceptions."""


class IRValidationError(Exception):
    """Raised when IR graph fails invariant validation."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        msg = f"{len(errors)} validation error(s):\n" + "\n".join(f"  - {e}" for e in errors)
        super().__init__(msg)


class UnsupportedOperatorError(Exception):
    """Raised when an ONNX model contains unsupported operators."""

    def __init__(self, unsupported_ops: list[tuple[str, str]], *, message: str | None = None):
        """
        Args:
            unsupported_ops: List of (op_type, node_name) tuples.
            message: Optional custom error message (overrides default).
        """
        self.unsupported_ops = unsupported_ops
        if message:
            super().__init__(message)
        else:
            details = ", ".join(f"{op} (node: {name})" for op, name in unsupported_ops)
            super().__init__(f"Unsupported operator(s): {details}")


class ShapeInferenceError(Exception):
    """Raised when shape inference fails to resolve all tensor shapes."""

    def __init__(self, unresolved: list[str]):
        self.unresolved = unresolved
        super().__init__(f"Shape inference failed for tensors: {unresolved}")


class ParseError(Exception):
    """Raised for general ONNX parsing errors."""
