# The following file follows the implementation of probnum.typing
# see:
from collections.abc import Iterable
from typing import Union

import jax
import jax.extend
import jax.numpy as jnp

import linox  # noqa: TCH001

# from linox._matrix import Scalar

########################################################################################
# API Types
########################################################################################

# Array Utilities
ShapeType = tuple[int, ...]
"""Type defining a shape of an object."""

# Scalars, Arrays and Matrices
ScalarType = jnp.ndarray
"""Type defining a scalar."""

MatrixType = Union[jnp.ndarray, "linox._linear_operator.LinearOperator"]  # noqa: SLF001
"""Type defining a matrix, i.e. a linear map between \
finite-dimensional vector spaces."""

########################################################################################
# Argument Types
########################################################################################

# Python Numbers
IntLike = int | jnp.integer
"""Object that can be converted to an integer.

Arguments of type :attr:`IntLike` should always be converted
into :class:`int`\\ s before further internal processing."""

FloatLike = float | jnp.floating
"""Object that can be converted to a float.

Arguments of type :attr:`FloatLike` should always be converted
into :class:`float`\\ s before further internal processing."""

# Array Utilities
ShapeLike = IntLike | Iterable[IntLike]
"""Object that can be converted to a shape.

Arguments of type :attr:`ShapeLike` should always be converted
into :class:`ShapeType` using the function :func:`linox.utils.as_shape`
before internal processing."""

DTypeLike = jax.numpy.dtype
"""Object that can be converted to an array dtype.

Arguments of type :attr:`DTypeLike` should always be converted
into :class:`jax.numpy.dtype`\\ s before further internal processing."""

# Scalars, Arrays and Matrices
ScalarLike = (
    int | float | complex | jax.extend.core.Primitive | jnp.number  # | linox._matrix.Scalar
)
"""Object that can be converted to a scalar value.

Arguments of type :attr:`ScalarLike` should always be converted
into :class:`jax.numpy.number`\\ s using the function :func:`probnum.utils.as_scalar`
before further internal processing."""

ArrayLike = jax.Array | jnp.ndarray | Iterable
"""Object that can be converted to an array.

Arguments of type :attr:`ArrayLike` should always be converted
into :class:`jax.Array`\\ s using the function :func:`jnp.asarray`
before further internal processing."""

LinearOperatorLike = Union["linox.LinearOperator", jax.Array, jnp.ndarray]
"""Object that can be converted to a :class:`~probnum.linops.LinearOperator`.

Arguments of type :attr:`LinearOperatorLike` should always be converted
into :class:`~probnum.linops.\\
LinearOperator`\\ s using the function :func:`probnum.linops.aslinop` before further
internal processing."""

########################################################################################
# Other Types
########################################################################################

NotImplementedType = type(NotImplemented)
"""Type of the `NotImplemented` constant."""
