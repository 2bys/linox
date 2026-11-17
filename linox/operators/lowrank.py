# _low_rank.py

r"""Low rank representations as linear operators.

This module implements various low rank representations as linear operators, including:

- :class:`LowRank`: Represents a low rank matrix :math:`A = U \text{diag}(S) V^T`
- :class:`SymmetricLowRank`: Represents a symmetric low rank matrix
    :math:`A = U \text{diag}(S) U^T`
- :class:`IsotropicScalingPlusSymmetricLowRank`: Represents
    :math:`\sigma I + U \text{diag}(S) U^T`
- :class:`PositiveDiagonalPlusSymmetricLowRank`: Represents
    :math:`D + \alpha U \text{diag}(S) U^T`
    where :math:`D` is a positive diagonal matrix
"""

import functools

import jax
import jax.numpy as jnp

from linox.linalg.woodbury import woodbury_solve
from linox.operators.arithmetic import (
    AddLinearOperator,
    ProductLinearOperator,
    congruence_transform,
    diagonal,
    linverse,
    lsolve,
    lsqrt,
    slogdet,
)
from linox.operators.base import LinearOperator
from linox.operators.dense import Matrix
from linox.operators.diagonal import Diagonal
from linox.operators.special import Identity


class LowRank(LinearOperator):
    r"""Low rank operator.

    For matrices :math:`U`, :math:`S`, and :math:`V`, this represents the low rank
    matrix :math:`A = U \text{diag}(S) V^T`. The action on a vector :math:`x` is given
    by :math:`Ax = U(S \odot (V^T x))` where :math:`\odot` denotes element-wise
    multiplication.

    Args:
        U: Left factor matrix
        S: Vector of singular values (optional, defaults to ones)
        V: Right factor matrix (optional, defaults to U)
    """

    def __init__(self, U: jax.Array, S: jax.Array | None = None, V: jax.Array | None = None) -> None:
        # Check shapes
        if S is not None:
            assert U.shape[-1] == S.shape[-1]
        if V is not None:
            assert U.shape[-1] == V.shape[-1]

        self._U = U
        # A defaulted S carries no data of its own, so it follows U's dtype
        # instead of JAX's default (which would upcast float32 under x64).
        self._S = S if S is not None else jnp.ones(U.shape[-1], dtype=U.dtype)
        self._V = V

        dtype = jnp.result_type(*(x for x in (U, self._S, V) if x is not None))
        # Move to an abstract base class instead
        super().__init__(shape=(*U.shape[:-2], U.shape[-2], U.shape[-2]), dtype=dtype)

    @property
    def U(self) -> jax.Array:
        """Left factor matrix."""
        return self._U

    @property
    def S(self) -> jax.Array:
        """Singular values vector."""
        return self._S

    @property
    def V(self) -> jax.Array:
        """Right factor matrix (defaults to U if not provided)."""
        return self._V if self._V is not None else self._U

    def _matmul(self, arr: jnp.array) -> jnp.array:
        return self.U @ (self.S[:, None] * (self.V.T @ arr))

    def _todense(self) -> jnp.array:
        return self.U @ jnp.diag(self.S) @ self.V.T

    def transpose(self) -> "LowRank":
        """Return transposed low rank operator."""
        return LowRank(self.V, self.S, self.U)

    def tree_flatten(self) -> tuple[tuple[any, ...], dict[str, any]]:
        """Flatten for JAX pytree registration."""
        children = (self._U, self._S, self._V)
        aux_data = {}
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: dict[str, any],
        children: tuple[any, ...],
    ) -> "LowRank":
        """Unflatten for JAX pytree registration."""
        del aux_data
        U, S, V = children
        return cls(U=U, S=S, V=V)


class SymmetricLowRank(LowRank):
    r"""Symmetric low rank operator.

    For matrices :math:`U` and :math:`S`, this represents the symmetric low rank matrix
    :math:`A = U \text{diag}(S) U^T`. The action on a vector :math:`x` is given by
    :math:`Ax = U(S \odot (U^T x))` where :math:`\odot` denotes element-wise
    multiplication.

    Args:
        U: Factor matrix
        S: Vector of singular values (optional, defaults to ones)
    """

    def __init__(self, U: jax.Array, S: jax.Array | None = None) -> None:
        super().__init__(U, S)

    def transpose(self) -> "SymmetricLowRank":
        """Return transpose (self for symmetric operators)."""
        return self

    def tree_flatten(self) -> tuple[tuple[any, ...], dict[str, any]]:
        """Flatten for JAX pytree registration."""
        children = (self._U, self._S)
        aux_data = {}
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: dict[str, any],
        children: tuple[any, ...],
    ) -> "SymmetricLowRank":
        """Unflatten for JAX pytree registration."""
        del aux_data
        U, S = children
        return cls(U=U, S=S)


class IsotropicScalingPlusSymmetricLowRank(AddLinearOperator):
    r"""Isotropic scaling plus symmetric low rank operator.

    For scalar :math:`\sigma`, matrix :math:`U`, and vector :math:`S`, this represents
    :math:`A = \sigma I + U \text{diag}(S) U^T`. The action on a vector :math:`x` is
    given by :math:`Ax = \sigma x + U(S \odot (U^T x))` where :math:`\odot` denotes
    element-wise multiplication.

    Args:
        scalar: Isotropic scaling factor :math:`\sigma`
        U: Factor matrix
        S: Vector of singular values
    """

    def __init__(self, scalar: jax.Array, U: jax.Array, S: jax.Array) -> None:
        self._scalar = scalar

        self._U = U
        self._S = S

        # Move to an abstract base class instead
        super().__init__(
            self._scalar * Identity(self._U.shape[-2], dtype=self._U.dtype),
            SymmetricLowRank(self._U, self._S),
        )

        # Add general tagging
        # self.is_symmetric = True
        self._lr_eigh = (self._U, self._S)

    @property
    def scalar(self) -> float:
        """Isotropic scaling factor."""
        return self._scalar

    @property
    def U(self) -> jax.Array:
        """Factor matrix."""
        return self._U

    @property
    def S(self) -> jax.Array:
        """Singular values vector."""
        return self._S

    @property
    def lr_eigh(self) -> jax.Array:
        """Low rank eigendecomposition (U, S)."""
        return self._lr_eigh

    def transpose(self) -> "IsotropicScalingPlusSymmetricLowRank":
        """Return transpose (self for symmetric operators)."""
        return self

    def tree_flatten(self) -> tuple[tuple[any, ...], dict[str, any]]:
        """Flatten for JAX pytree registration."""
        # We need to override the AddLinearOperator's tree_flatten
        children = (self._scalar, self._U, self._S)
        aux_data = {}
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: dict[str, any],
        children: tuple[any, ...],
    ) -> "IsotropicScalingPlusSymmetricLowRank":
        """Unflatten for JAX pytree registration."""
        del aux_data
        scalar, U, S = children
        return cls(scalar=scalar, U=U, S=S)


@linverse.dispatch
def _(a: IsotropicScalingPlusSymmetricLowRank) -> IsotropicScalingPlusSymmetricLowRank:
    r"""Inverse of an isotropic scaling plus symmetric low rank operator.

    For :math:`A = \sigma I + U \text{diag}(S) U^T`, this represents
    :math:`A^{-1} = \frac{1}{\sigma}I - \frac{1}{\sigma}U \text{diag}
    (\frac{S}{\sigma(S + \sigma)}) U^T`
    """
    scalar_inv = 1 / a.scalar
    Q, lr_eigvals = a.lr_eigh
    return IsotropicScalingPlusSymmetricLowRank(
        scalar_inv,
        Q,
        -lr_eigvals / (a.scalar * (lr_eigvals + a.scalar)),
        # This is numerically more stable than:
        # (1 / (lr_eigvals + a.scalar)) - scalar_inv
    )


@lsqrt.dispatch
def _(a: IsotropicScalingPlusSymmetricLowRank) -> IsotropicScalingPlusSymmetricLowRank:
    r"""Square root of an isotropic scaling plus symmetric low rank operator.

    For :math:`A = \sigma I + U \text{diag}(S) U^T`, this represents
    :math:`A^{1/2} = \sqrt{\sigma}I + U \text{diag}
    (\sqrt{\sigma}(\sqrt{\frac{S}{\sigma} + 1} - 1)) U^T`
    """
    scalar_sqrt = jnp.sqrt(a.scalar)
    Q, lr_eigvals = a.lr_eigh
    return IsotropicScalingPlusSymmetricLowRank(
        scalar_sqrt,
        Q,
        scalar_sqrt * (jnp.sqrt((lr_eigvals / a.scalar) + 1) - 1),
    )


class PositiveDiagonalPlusSymmetricLowRank(AddLinearOperator):
    r"""Positive diagonal plus symmetric low rank operator.

    For positive diagonal matrix :math:`D`, matrix :math:`U`, vector :math:`S`, and
    scalar :math:`\alpha`, this represents :math:`A = D + \alpha U \text{diag}(S) U^T`.
    The action on a vector :math:`x` is given by
    :math:`Ax = Dx + \alpha U(S \odot (U^T x))` where :math:`\odot` denotes
    element-wise multiplication.

    Args:
        diagonal: Positive diagonal matrix :math:`D`
        low_rank: Symmetric low rank component :math:`U \text{diag}(S) U^T`
        low_rank_scale: Scaling factor :math:`\alpha` (default: 1.0)
    """

    def __init__(
        self,
        diagonal: Diagonal,  # D
        low_rank: SymmetricLowRank,  # U S U^T
        low_rank_scale: float = 1.0,  # a
    ) -> None:
        self._diagonal = diagonal
        self._low_rank = low_rank
        self._low_rank_scale = low_rank_scale

        super().__init__(self._diagonal, self._low_rank_scale * self._low_rank)

    @property
    def diagonal(self) -> jax.Array:
        """Diagonal component."""
        return self._diagonal

    @property
    def low_rank(self) -> SymmetricLowRank:
        """Low rank component."""
        return self._low_rank

    @property
    def low_rank_scale(self) -> float:
        """Scaling factor for low rank component."""
        return self._low_rank_scale

    @functools.cached_property
    def _id_plus_low_rank(self) -> IsotropicScalingPlusSymmetricLowRank:
        """1 + a (D^{-1/2} U) S (D^{-1/2} U)^T = D^{-1/2} (D + a U S U^T) D^{-1/2}."""
        U, sqrt_S, _ = jnp.linalg.svd(
            ((self.low_rank.U * jnp.sqrt(self.low_rank.S)) / jnp.sqrt(self._diagonal.diag[:, None])),
            full_matrices=False,
            compute_uv=True,
        )

        return IsotropicScalingPlusSymmetricLowRank(
            1.0,
            U,
            self._low_rank_scale * sqrt_S**2,
        )

    def transpose(self) -> "PositiveDiagonalPlusSymmetricLowRank":
        """Return transpose (self for symmetric operators)."""
        return self

    def tree_flatten(self) -> tuple[tuple[any, ...], dict[str, any]]:
        """Flatten for JAX pytree registration."""
        # We need to override the AddLinearOperator's tree_flatten
        children = (self._diagonal, self._low_rank, self._low_rank_scale)
        aux_data = {}
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: dict[str, any],
        children: tuple[any, ...],
    ) -> "PositiveDiagonalPlusSymmetricLowRank":
        """Unflatten for JAX pytree registration."""
        del aux_data
        diagonal, low_rank, low_rank_scale = children
        return cls(diagonal=diagonal, low_rank=low_rank, low_rank_scale=low_rank_scale)


@linverse.dispatch
def _(A: PositiveDiagonalPlusSymmetricLowRank) -> PositiveDiagonalPlusSymmetricLowRank:
    r"""Inverse of a positive diagonal plus symmetric low rank operator.

    For :math:`A = D + \alpha U \text{diag}(S) U^T`, this represents
    :math:`A^{-1} = D^{-1} - \alpha D^{-1}U(S^{-1} + \alpha U^TD^{-1}U)^{-1}U^TD^{-1}`
    """
    # (D + a U S U^T)^{-1} = D^{-1} - a D^{-1} U (S^{-1} + a U^T D^{-1} U)^-1 U^T D^{-1}

    D_inv = linverse(A.diagonal)

    schur = A.low_rank_scale * A.low_rank.U.T @ D_inv @ A.low_rank.U + jnp.diag(1 / A.low_rank.S)
    schur_eigvals, schur_eigvecs = jnp.linalg.eigh(schur)

    # Compute eigendecomposition of (D^{-1} U schur^{-1/2}) (D^{-1} U schur^{-1/2})^T
    U, sqrt_S, _ = jnp.linalg.svd(
        D_inv @ A.low_rank.U @ schur_eigvecs / jnp.sqrt(schur_eigvals),
        full_matrices=False,
        compute_uv=True,
    )

    return PositiveDiagonalPlusSymmetricLowRank(
        linverse(A.diagonal),
        SymmetricLowRank(U, sqrt_S**2),
        low_rank_scale=-A.low_rank_scale,
    )


@lsolve.dispatch
def _(A: PositiveDiagonalPlusSymmetricLowRank, b: jax.Array) -> jax.Array:
    r"""Solve a linear system with a positive diagonal plus symmetric low rank operator.

    Using the Woodbury matrix identity:
    (D + a U S U^T)^{-1} v = D^{-1} v - D^{-1} U (S^{-1} + a U^T D^{-1} U)^{-1} U^T D^{-1} v.
    """
    scale = A.low_rank_scale
    U = A.low_rank.U
    S = A.low_rank.S

    # woodbury_solve(U, s, d, v) solves (U diag(s) U^T + D) x = v
    # Our form: a U S U^T + D = U diag(a S) U^T + D
    return woodbury_solve(U, scale * S, A.diagonal.diag, b)


@lsqrt.dispatch
def _(
    A: PositiveDiagonalPlusSymmetricLowRank,
) -> ProductLinearOperator:
    r"""Square root of a positive diagonal plus symmetric low rank operator.

    For :math:`A = D + \alpha U \text{diag}(S) U^T`, this represents
    :math:`A^{1/2} = D^{1/2}(I + \alpha D^{-1/2}U \text{diag}(S) U^TD^{-1/2})^{1/2}`
    """
    return lsqrt(A.diagonal) @ lsqrt(A._id_plus_low_rank)


@slogdet.dispatch
def _(A: PositiveDiagonalPlusSymmetricLowRank) -> tuple[jax.Array, jax.Array]:
    r"""Compute log-determinant using the Matrix Determinant Lemma.

    |D + U S U^T| = |D| * |I + S U^T D^{-1} U|
    or |I + U S U^T D^{-1}| = |I + S U^T D^{-1} U| (Sylvester's determinant theorem)

    Actually: |D + U S U^T| = |D| |I + S^{-1} + U^T D^{-1} U| wait no.
    Matrix Determinant Lemma: |A + U W V^T| = |D| |W| |W^{-1} + V^T A^{-1} U|
    Here A=D, W=S (diag), V=U, U=U.
    |D + U S U^T| = |D| |S| |S^{-1} + U^T D^{-1} U|

    But S might be singular or non-invertible if used loosely.
    Alternative: |D + U S U^T| = |D| |I + S U^T D^{-1} U|
    Currently we assume S is diagonal vector.

    C = S * (U^T D^{-1} U)
    inner = I + C
    """
    from linox.operators.arithmetic import slogdet

    sign_D, logdet_D = slogdet(A.diagonal)

    # Inner matrix: I + alpha * S * (U^T D^{-1} U)
    # Shape k x k
    D_inv = linverse(A.diagonal)

    # U^T D^{-1} U
    # If D is diagonal, D_inv is diagonal.
    # We can compute efficiently: (U / D_diag)^T @ U
    Ut_Dinv_U = (A.low_rank.U.T @ D_inv) @ A.low_rank.U

    # Scale by S * alpha
    # S is vector, acting as diagonal matrix
    scaled = A.low_rank.S[:, None] * Ut_Dinv_U  # S @ (Ut Dinv U)

    # Add Identity
    k = A.low_rank.S.shape[0]
    inner = jnp.eye(k, dtype=A.dtype) + A.low_rank_scale * scaled

    sign_inner, logdet_inner = jnp.linalg.slogdet(inner)

    return sign_D * sign_inner, logdet_D + logdet_inner


# --------------------------------------------------------------------------- #
# Structured diagonal / congruence rules
# --------------------------------------------------------------------------- #


@diagonal.dispatch
def _(a: LowRank) -> jax.Array:
    r""":math:`\operatorname{diag}(U \operatorname{diag}(S) V^\top)_i = \sum_j U_{ij} S_j V_{ij}`, in :math:`O(nr)`."""
    return jnp.sum(a.U * a.S[..., None, :] * a.V, axis=-1)


@congruence_transform.dispatch
def _(A: LinearOperator | jax.Array, B: SymmetricLowRank) -> SymmetricLowRank:
    r""":math:`A (U S U^\top) A^\top = (AU) S (AU)^\top`."""
    return SymmetricLowRank(A @ B.U, B.S)


@congruence_transform.dispatch
def _(A: Matrix, B: Identity) -> SymmetricLowRank:
    r"""For dense ``A`` of shape ``(n, k)``, :math:`A A^\top` is ``SymmetricLowRank(A, 1)``.

    This keeps e.g. :func:`diagonal` at :math:`O(nk)` instead of materialising
    the ``(n, n)`` product.
    """
    # `Identity` carries no data, so the result takes A's dtype.
    return SymmetricLowRank(A.A, jnp.ones(B.shape[-1], dtype=A.dtype))


@congruence_transform.dispatch
def _(A: Matrix, B: Diagonal) -> SymmetricLowRank:
    r""":math:`A \operatorname{diag}(d) A^\top` for dense ``A`` is ``SymmetricLowRank(A, d)``."""
    return SymmetricLowRank(A.A, B.diag)


# Register all low rank operators as PyTrees
jax.tree_util.register_pytree_node_class(LowRank)
jax.tree_util.register_pytree_node_class(SymmetricLowRank)
jax.tree_util.register_pytree_node_class(IsotropicScalingPlusSymmetricLowRank)
jax.tree_util.register_pytree_node_class(PositiveDiagonalPlusSymmetricLowRank)
