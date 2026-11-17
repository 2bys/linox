"""Toeplitz matrix operators with FFT-based multiplication.

Provides efficient O(n log n) matrix-vector products for Toeplitz matrices
using FFT-based circular convolution.
"""
# _toeplity.py

import jax
from jax import numpy as jnp
from jax import scipy as jsp

from linox._types import ArrayLike
from linox.linalg._toeplitz_solve import solve_toeplitz_jax
from linox.operators.arithmetic import diagonal, lsolve
from linox.operators.base import LinearOperator
from linox.operators.special import Identity

# --------------------------------------------------------------------------- #
# Toeplitz Linear Operator
# --------------------------------------------------------------------------- #


class Toeplitz(LinearOperator):
    """Symmetric Toeplitz matrix represented by its first column/row v."""

    def __init__(self, v: ArrayLike) -> None:
        self.v = jnp.asarray(v)
        n = int(self.v.shape[0])
        super().__init__((n, n), self.v.dtype)

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the Toeplitz matrix."""
        n = int(self.v.shape[0])
        return (n, n)

    def _matmul(self, vector: jax.Array) -> jax.Array:
        """Matrix-vector product for symmetric Toeplitz matrix.

        Uses FFT-based circulant convolution for O(n log n) complexity.

        Args:
            vector: Vector to multiply with, shape (n,) or (n, k)

        Returns
        -------
            Result of multiplication, shape (n,) or (n, k)
        """
        if vector.ndim == 1:
            vector = vector[:, None]
            squeeze_output = True
        else:
            squeeze_output = False

        # Input shape: (..., n, k)
        # We need to perform FFT along the dimension with size n (axis -2)
        n = self.shape[0]
        if vector.shape[-2] != n:
            msg = f"Dimension mismatch: expected size {n} at axis -2, got {vector.shape[-2]}"
            raise ValueError(msg)

        # Embed first row into circulant row: [v, 0, v[-1:0:-1]]
        # But for symmetric Toeplitz: [v, v[n-1:0:-1]] ?
        # Code was: jnp.concatenate([self.v, self.v[-1:0:-1]])
        # This creates the first column of the circulant matrix.
        embedded_col = jnp.concatenate([self.v, self.v[-1:0:-1]])
        p = embedded_col.shape[0]

        # FFT of the circulant column (1D)
        fft_col = jnp.fft.fft(embedded_col)

        # Pad vector along axis -2 to length p
        # padding shape: (..., p-n, k)
        pad_width = p - n
        padding_shape = list(vector.shape)
        padding_shape[-2] = pad_width
        zeros = jnp.zeros(padding_shape, dtype=vector.dtype)
        vector_padded = jnp.concatenate([vector, zeros], axis=-2)

        # FFT along axis -2
        fft_vector = jnp.fft.fft(vector_padded, axis=-2)

        # Broadcast fft_col: needs shape (1, ..., 1, p, 1) to match (..., p, k)
        # Actually standard broadcasting rules: (p,) broadcasts to (..., p, k) if p is last?
        # No, p is second to last.
        # We need (..., p, 1).
        # Reshape fft_col to (1, ..., 1, p, 1)
        # Simpler: expand dims at -1 and as many as needed on left
        fft_col.reshape((-1, 1))  # (p, 1)
        # This will broadcast against (..., p, k) correctly as (p, 1) * (..., p, k) ?
        # No. (p, 1) * (batch, p, k) -> (batch, p, k). Correct.
        # Wait, if batch is present. (B, p, k). (p, 1).
        # (p, 1) broadcasts to (B, p, k) ? No. Last dims must align.
        # (p, 1) aligns with (k,) ? No.
        # (B, p, k) * (p, 1) -> broadcasting (p, 1) against (k) fails.
        # We need to broadcast over k.
        # fft_vector: (..., p, k). fft_col: (p,).
        # We want fft_col to multiply along p axis.
        # Reshape fft_col to (p, 1).
        # (..., p, k) * (p, 1) -> (..., p, k).
        # Example: (2, 5, 3) * (5, 1).
        # (2, 5, 3). (5, 1).
        # 3 vs 1 -> ok.
        # 5 vs 5 -> ok.
        # 2 vs ? -> 1. ok.
        # So (p, 1) works!

        fft_result = fft_vector * fft_col.reshape((-1, 1))

        # IFFT
        result = jnp.fft.ifft(fft_result, axis=-2).real

        # Slice to original size n
        # This slicing doesn't work simply with [..., :n, :] syntax in python slices unless we build it?
        # Actually result[..., :n, :] syntax works in JAX/Numpy.
        result = result[..., :n, :]

        if squeeze_output:
            result = result.squeeze(axis=-1)  # Was axis=1 which assumes (n, 1) -> (n,)
            # If input was (n,), vector became (n, 1). Result (n, 1). Squeeze -> (n,).
            # If input was (1, n), vector became (1, n). Reshape?
            # Wait, line starts: if vector.ndim == 1: vector = vector[:, None].
            # (n,) -> (n, 1).
            # Squeeze -1 works.

        return result

    def _todense(self) -> jax.Array:
        return jsp.linalg.toeplitz(self.v)

    def from_matrix(self, matrix: jax.Array) -> "Toeplitz":
        """Create Toeplitz operator from matrix."""
        self.v = matrix[0, :]
        return Toeplitz(self.v)

    def transpose(self) -> "Toeplitz":
        """Return transpose (self for symmetric Toeplitz)."""
        return Toeplitz(self.v)


@diagonal.dispatch
def _(A: Toeplitz) -> LinearOperator:
    return A.v[0] * Identity(A.shape[0])


@lsolve.dispatch
def _(A: Toeplitz, b: jax.Array) -> jax.Array:
    """Solve a Toeplitz system."""
    # Use the algorithm module implementation (hybrid SciPy+JAX with custom VJP)
    return solve_toeplitz_jax(A.v, b, check_finite=False)


# Solver and related algorithms moved to linox/_algorithms/_toeplitz.py
