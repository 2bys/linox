"""Public type aliases for linox.

Downstream code should import types from here rather than from the private
``linox._types`` module, which may change without notice.
"""

from linox._types import (
    AnyType,
    ArrayLike,
    CallableType,
    DTypeLike,
    FloatLike,
    IntLike,
    Key,
    LinearOperatorLike,
    MatrixType,
    NotImplementedType,
    ScalarLike,
    ScalarType,
    ShapeLike,
    ShapeType,
)

__all__ = [
    "AnyType",
    "ArrayLike",
    "CallableType",
    "DTypeLike",
    "FloatLike",
    "IntLike",
    "Key",
    "LinearOperatorLike",
    "MatrixType",
    "NotImplementedType",
    "ScalarLike",
    "ScalarType",
    "ShapeLike",
    "ShapeType",
]
