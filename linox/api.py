"""Linox API - Public Functional Interface.

This module provides the main functional entry points for the Linox library.
"""

from typing import Any

import jax
import jax.numpy as jnp

from . import config
from .config import is_debug, set_debug
from .linalg import functions as _functions_module
from .linalg import spectral as _spectral_module
from .linalg import trace as _trace_module
from .linalg.approx.arnoldi import arnoldi_iteration, arnoldi_matrix_function
from .linalg.approx.hutchinson import (
    hutchinson_diagonal,
    hutchinson_trace,
    hutchinson_trace_and_diagonal,
)
from .linalg.approx.lanczos import (
    lanczos_eigh,
    lanczos_matrix_function,
    lanczos_tridiag,
)
from .linalg.approx.lsmr import lsmr_solve
from .linalg.approx.slq import slq, slq_logdet
from .linalg.functions import stochastic_lanczos_quadrature
from .linalg.solution import RESULTS, LinearSolveError, Solution
from .linalg.solution import RESULTS as _RESULTS
from .linalg.solution import check_result as _check_result
from .linalg.solution import residual_result as _residual_result
from .linalg.spectral import lanczos_bidiag, svd_partial

# Arithmetic operators (for export compatibility)
from .operators.arithmetic import (
    AddLinearOperator,
    InverseLinearOperator,  # Compatibility aliases
    ProductLinearOperator,
    PseudoInverseLinearOperator,
    ScaledLinearOperator,
    TransposedLinearOperator,
    cholesky,
    congruence_transform,
    diagonal,
    is_hermitian,
    is_square,
    is_symmetric,
    lcholesky,
    ldet,
    lexp,
    linverse,
    llog,
    lpinverse,
    lpow,
    lpsolve,
    lqr,
    lu_factor,
    lu_solve,
    psolve,
    qr,
    symmetrize,
)
from .operators.arithmetic import leigh as _leigh_impl
from .operators.arithmetic import lsolve as _lsolve_impl
from .operators.arithmetic import lsqrt as _lsqrt_impl
from .operators.arithmetic import (
    svd as _svd_impl,
)

# Internal imports
from .operators.base import LinearOperator
from .operators.block import BlockDiagonal, BlockMatrix, BlockMatrix2x2

# Matrix classes
from .operators.dense import Matrix
from .operators.diagonal import Diagonal
from .operators.eigen import EigenD
from .operators.isotropic import IsotropicAdditiveLinearOperator
from .operators.kron import Kronecker, KroneckerSelectedEigenvectors, topk_eigh
from .operators.lowrank import (
    IsotropicScalingPlusSymmetricLowRank,
    LowRank,
    PositiveDiagonalPlusSymmetricLowRank,
    SymmetricLowRank,
)
from .operators.permutation import Permutation
from .operators.special import Identity, Ones, Scalar, Zero
from .operators.toeplitz import Toeplitz
from .operators.wrappers import (
    PSD,
    SPD,
    Sym,
    assume_psd,
    assume_spd,
    assume_symmetric,
)
from .utils import array as _array_module

# Not part of the public API; re-exported only because `linox._broadcast_shapes`
# was importable in earlier releases.
from .utils.array import _broadcast_shapes as _broadcast_shapes
from .utils.array import allclose
from .utils.debug import inspect_run, linop_graph
from .utils.validation import ValidationError, validate

# Type aliases
ArrayLike = jax.Array | Any
LinearlyOperatorLike = LinearOperator | ArrayLike
Int = int


__all__ = [
    "PSD",
    "RESULTS",
    "SPD",
    "AddLinearOperator",
    "BlockDiagonal",
    "BlockMatrix",
    "BlockMatrix2x2",
    "Diagonal",
    "EigenD",
    "Identity",
    "InverseLinearOperator",
    "IsotropicAdditiveLinearOperator",
    "IsotropicScalingPlusSymmetricLowRank",
    "Kronecker",
    "KroneckerSelectedEigenvectors",
    "LinearOperator",
    "LinearSolveError",
    "LowRank",
    "Matrix",
    "Ones",
    "Permutation",
    "PositiveDiagonalPlusSymmetricLowRank",
    "ProductLinearOperator",
    "PseudoInverseLinearOperator",
    "Scalar",
    "ScaledLinearOperator",
    "Solution",
    "Sym",
    "SymmetricLowRank",
    "Toeplitz",
    "TransposedLinearOperator",
    "ValidationError",
    "Zero",
    "allclose",
    "arnoldi_iteration",
    "arnoldi_matrix_function",
    "as_linop",
    "assume_psd",
    "assume_spd",
    "assume_symmetric",
    "block_diag",
    "bmat",
    "cholesky",
    "congruence_transform",
    "det",
    "diag",
    "diagonal",
    "eigh",
    "exp",
    "eye",
    "hutchinson_diagonal",
    "hutchinson_trace",
    "hutchinson_trace_and_diagonal",
    "inspect_run",
    "inv",
    "inverse",
    "is_debug",
    "is_hermitian",
    "is_square",
    "is_symmetric",
    "kron",
    "lanczos_bidiag",
    "lanczos_eigh",
    "lanczos_matrix_function",
    "lanczos_tridiag",
    "lcholesky",
    "ldet",
    "leigh",
    "lexp",
    "linop_graph",
    "linverse",
    "llog",
    "log",
    "logdet",
    "lpinverse",
    "lpow",
    "lpsolve",
    "lqr",
    "lsmr_solve",
    "lsolve",
    "lsqrt",
    "lu_factor",
    "lu_solve",
    "ones",
    "pinv",
    "pinverse",
    "pow",
    "psolve",
    "qr",
    "set_debug",
    "slogdet",
    "slq",
    "slq_logdet",
    "solve",
    "sqrt",
    "stochastic_lanczos_quadrature",
    "svd",
    "svd_partial",
    "symmetrize",
    "todense",
    "toeplitz",
    "topk_eigh",
    "trace",
    "transpose",
    "validate",
    "zeros",
]


# --- Utilities ---


def ensure_linop(a: Any) -> LinearOperator:
    """Helper to ensure input is a LinearOperator."""
    return _array_module.as_linop(a)


def as_linop(a: Any) -> LinearOperator:
    """Convert object to LinearOperator."""
    return _array_module.as_linop(a)


def todense(a: LinearlyOperatorLike) -> jax.Array:
    """Convert operator to dense matrix."""
    return _array_module.todense(a)


# --- Creation functions ---


def eye(N: int, M: int | None = None, k: int = 0, dtype: Any = None) -> LinearOperator:
    """Return a 2-D array with ones on the diagonal and zeros elsewhere.

    Args:
        N: Number of rows in the output.
        M: Number of columns in the output. If None, defaults to N.
        k: Index of the diagonal: 0 (the default) refers to the main diagonal,
            a positive value refers to an upper diagonal, and a negative value
            to a lower diagonal.
        dtype: Data-type of the returned array.
    """
    if M is None:
        M = N

    if k == 0 and N == M:
        return Identity(N, dtype=dtype)

    # For off-diagonals or non-square, just use Matrix wrapping dense eye for now
    return Matrix(jnp.eye(N, M, k=k, dtype=dtype))


def zeros(dim: Int, shape: tuple[Int, Int] | None = None) -> LinearOperator:
    """Create a zero operator.

    Args:
        dim: Dimension or number of rows.
        shape: Optional shape tuple (rows, cols).
    """
    if shape is None:
        return Zero((dim, dim))
    return Zero(shape)


def ones(dim: Int, shape: tuple[Int, Int] | None = None) -> LinearOperator:
    """Create a ones operator."""
    if shape is None:
        return Ones((dim, dim))
    return Ones(shape)


def diag(v: jax.Array) -> LinearOperator:
    """Create a diagonal operator from a vector."""
    return Diagonal(v)


# --- Structure / Arithmetic wrappers ---


def transpose(a: LinearlyOperatorLike) -> LinearOperator:
    """Return the transpose of a linear operator."""
    return ensure_linop(a).T


def inv(a: LinearlyOperatorLike) -> LinearOperator:
    """Compute the inverse of a linear operator.

    Uses the exact structured inverse when one is registered (e.g. the
    Woodbury form for :class:`IsotropicScalingPlusSymmetricLowRank`) and a
    lazy :class:`InverseLinearOperator` otherwise. Same as
    ``inverse(a, method="exact")``.
    """
    return linverse(ensure_linop(a))


def pinv(a: LinearlyOperatorLike) -> LinearOperator:
    """Compute the pseudo-inverse of a linear operator.

    Uses a structured pseudo-inverse when one is registered, and a lazy
    :class:`PseudoInverseLinearOperator` otherwise.
    """
    return lpinverse(ensure_linop(a))


def kron(a: LinearlyOperatorLike, b: LinearlyOperatorLike) -> LinearOperator:
    """Compute the Kronecker product of two linear operators."""
    return Kronecker(ensure_linop(a), ensure_linop(b))


def block_diag(*opers: LinearlyOperatorLike) -> LinearOperator:
    """Construct a block diagonal operator from input operators."""
    return BlockDiagonal(*(ensure_linop(op) for op in opers))


def bmat(blocks: list[list[LinearlyOperatorLike]]) -> LinearOperator:
    """Construct a block matrix from a list of lists of operators."""
    linop_blocks = [[ensure_linop(op) for op in row] for row in blocks]
    return BlockMatrix(linop_blocks)


def toeplitz(c: jax.Array, r: jax.Array | None = None) -> LinearOperator:
    """Construct a Toeplitz operator from column c and optional row r.

    If r is None, assumes symmetric Toeplitz (r = c).
    """
    if r is not None:
        msg = "Asymmetric Toeplitz not yet supported via simple wrapper."
        raise NotImplementedError(msg)
    return Toeplitz(c)


# --- Linear Algebra Functions (Canonical Wrappers) ---


def trace(a: LinearlyOperatorLike, method: str = "auto", **kwargs) -> jax.Array:
    """Compute the trace of a linear operator.

    Args:
        a: Linear operator.
        method: Computation method ("auto", "exact", "hutchinson").
    """
    op = ensure_linop(a)
    m = config.resolve_method("trace", op, method)

    # `auto` picks Hutchinson for large operators, but that needs a PRNG key.
    # Without one, fall back to the exact path rather than failing a plain
    # `trace(a)` call purely because the operator is big.
    if m == "hutchinson" and method == "auto" and kwargs.get("key") is None:
        m = "exact"

    if m == "hutchinson":
        return _trace_module.trace(op, method="hutchinson", **kwargs)
    return _trace_module.trace(op, **kwargs)


def det(a: LinearlyOperatorLike) -> jax.Array:
    """Compute determinant."""
    from linox.linalg.determinants import det as _det

    return _det(ensure_linop(a))


def slogdet(a: LinearlyOperatorLike, method: str = "auto", **kwargs) -> tuple[jax.Array, jax.Array]:
    """Compute sign and log of determinant.

    Args:
        a: Linear operator.
        method: Computation method ("auto", "exact", "slq").
    """
    from linox.linalg.determinants import slogdet as _slogdet

    op = ensure_linop(a)
    m = config.resolve_method("slogdet", op, method)

    # As in `trace`: SLQ needs a PRNG key, so an `auto` resolution that lands
    # there without one falls back to the exact path.
    if m == "slq" and method == "auto" and kwargs.get("key") is None:
        m = "exact"

    return _slogdet(op, method=m, **kwargs)


def logdet(a: LinearlyOperatorLike) -> jax.Array:
    """Compute log of determinant."""
    from linox.linalg.determinants import logdet as _logdet

    return _logdet(ensure_linop(a))


def _has_structured_solver(op: LinearOperator) -> bool:
    """Check if the operator type has a specialized efficient lsolve dispatch.

    Structured operators (Kronecker, ScaledLinearOperator wrapping structured,
    etc.) have plum-dispatched solvers that are more efficient than iterative
    methods like LSMR and don't require densification.
    """
    from linox.operators.factor import CholeskyFactor, PSDFromFactor, Triangular
    from linox.operators.isotropic import IsotropicAdditiveLinearOperator
    from linox.operators.lowrank import PositiveDiagonalPlusSymmetricLowRank
    from linox.operators.toeplitz import Toeplitz

    _STRUCTURED_TYPES = (
        Kronecker,
        Toeplitz,
        PositiveDiagonalPlusSymmetricLowRank,
        IsotropicAdditiveLinearOperator,
        Triangular,
        CholeskyFactor,
        PSDFromFactor,
    )

    if isinstance(op, _STRUCTURED_TYPES):
        return True
    if isinstance(op, ScaledLinearOperator):
        return _has_structured_solver(op.operator)
    if isinstance(op, InverseLinearOperator):
        return _has_structured_solver(op.operator)
    return False


def _is_dense_storage(op: LinearOperator) -> bool:
    """Whether ``op`` is a dense matrix, possibly scaled or inverted."""
    if isinstance(op, Matrix):
        return True
    if isinstance(op, ScaledLinearOperator | InverseLinearOperator):
        return _is_dense_storage(op.operator)
    return False


# LSMR termination codes that mean "converged"; see `linox.linalg.approx.lsmr`.
# 0 means the loop exited on the iteration cap, 3 means the conditioning limit
# was hit -- neither is a usable answer.
_LSMR_RESULT_FROM_ISTOP = {
    0: _RESULTS.max_steps_reached,
    3: _RESULTS.conlim,
}

#: Relative-residual threshold for iterative solvers that report no status of
#: their own. Deliberately loose -- it is a "this is obviously not a solution"
#: guard, not a convergence criterion. Currently unused: every iterative path
#: now reports its own outcome. Kept for solvers added later that do not.
_ITERATIVE_RESIDUAL_RTOL = 1e-2


def solve(
    a: LinearlyOperatorLike,
    b: jax.Array,
    method: str = "auto",
    *,
    throw: bool = True,
    return_info: bool = False,
    residual_rtol: float = 1e-5,
    **kwargs,
) -> jax.Array | tuple[jax.Array, Solution]:
    """Solve linear system Ax = b.

    Args:
        a: Linear operator.
        b: Right-hand side vector/matrix.
        method: Solver method ("exact", "lsmr", "cg", "auto"). ``"cg"`` uses
            linox's own preconditioned conjugate gradients, which requires a
            symmetric positive-definite operator and accepts a
            ``preconditioner=`` operator.
        throw: Raise :class:`~linox.linalg.solution.LinearSolveError` when the
            solve fails (the default). Pass ``False`` to accept whatever the
            solver produced. Under ``jax.jit`` the outcome is a tracer and
            cannot be raised at trace time, so the failure is reported by a
            runtime callback instead -- branch on ``info.result`` if you need
            to handle it inside the computation.
        return_info: Also return a :class:`~linox.linalg.solution.Solution`
            carrying the outcome code and solver diagnostics.
        residual_rtol: Relative residual above which a square solve is judged
            to have failed. A singular direct solve typically returns finite,
            enormous values rather than NaN, so the residual is the only
            reliable detector.

    Returns
    -------
        ``x``, or ``(x, info)`` when ``return_info=True``.

    Raises
    ------
    LinearSolveError
        If the solve failed and ``throw=True``.
    """
    op = ensure_linop(a)
    b = jnp.asarray(b)

    m = config.resolve_method("solve", op, method)

    stats: dict[str, jax.Array] = {}
    result: _RESULTS | jax.Array | None = None
    # Tolerance for the residual sanity check, or None to skip it because the
    # solver reported its own outcome.
    check_rtol: float | None = residual_rtol

    if m == "exact" or _has_structured_solver(op):
        x = _lsolve_impl(op, b, **kwargs)
    elif m == "lsmr":
        from linox.linalg.approx.lsmr import lsmr_solve

        x, info = lsmr_solve(op, b, **kwargs)
        stats = dict(info)
        # LSMR reports its own termination code, and stops at *its* tolerance
        # rather than machine precision. Second-guessing that with a tighter
        # residual threshold would flag perfectly good converged solves.
        result = _lsmr_result(info["istop"])
        check_rtol = None
    elif m in {"cg", "conjugate_gradient"}:
        from linox.linalg.approx.cg import CG_CONVERGED, cg_solve

        x, info = cg_solve(op, b, **kwargs)
        stats = dict(info)
        # Unlike `jax.scipy.sparse.linalg.cg`, this reports whether it
        # converged, so the loose residual guard is no longer needed.
        result = jnp.where(
            jnp.asarray(info["istop"]) == CG_CONVERGED,
            jnp.int32(_RESULTS.successful),
            jnp.int32(_RESULTS.max_steps_reached),
        )
        check_rtol = None
    else:
        x = _lsolve_impl(op, b, **kwargs)

    # Residual check for square systems. Rectangular ones legitimately have a
    # nonzero residual (that is least squares, not failure), so skip them.
    if check_rtol is not None and is_square(op):
        result, residual = _residual_result(op, x, b, rtol=check_rtol)
        stats = {**stats, "residual": residual}

    if result is None:
        result = _RESULTS.successful

    _check_result(result, throw=throw, detail=f"operator: {op}")

    if return_info:
        return x, Solution(value=x, result=result, stats=stats)
    return x


def _lsmr_result(istop: jax.Array) -> jax.Array:
    """Map an LSMR termination code onto a :class:`RESULTS` value."""
    result = jnp.int32(_RESULTS.successful)
    for code, outcome in _LSMR_RESULT_FROM_ISTOP.items():
        result = jnp.where(jnp.asarray(istop) == code, jnp.int32(outcome), result)
    return result


def eigh(
    a: LinearlyOperatorLike,
    k: Int | None = None,
    subset_by_index: tuple[Int, Int] | None = None,
    method: str = "auto",
    **kwargs,
) -> tuple[jax.Array, LinearOperator] | jax.Array:
    """Compute eigenvalues and eigenvectors.

    Args:
        a: Linear operator.
        k: Number of eigenvalues (for approx/partial).
        subset_by_index: Range of indices (start, end) for eigenvalues.
        method: "exact" or "lanczos".
    """
    m = config.resolve_method("eigh", ensure_linop(a), method)
    return _spectral_module.eigh(ensure_linop(a), k=k, subset_by_index=subset_by_index, method=m, **kwargs)


def svd(a: LinearlyOperatorLike, **kwargs) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Singular Value Decomposition."""
    return _svd_impl(ensure_linop(a), **kwargs)


# --- Element-wise / Function Application ---


def sqrt(a: LinearlyOperatorLike, method: str = "auto", **kwargs) -> LinearOperator:
    """Matrix square root factor.

    Returns an operator ``S`` satisfying ``S @ S.T == a``. Note this is a
    *factor*, not necessarily the symmetric principal square root: the exact
    path returns whatever structured factorisation is available for the
    operator (a Cholesky factor for a dense :class:`Matrix`, the elementwise
    root for a :class:`Diagonal`, and so on). Request ``method="lanczos"``
    to get the principal square root ``a**(1/2)`` via a Krylov method.

    With ``method="auto"``, operators larger than
    :func:`linox.config.get_max_dense_n` use a structured exact factor when
    one exists, and the Lanczos square root otherwise. A dense
    :class:`Matrix` always takes the Lanczos path at that size.

    Args:
        a: Linear operator.
        method: One of ``"auto"``, ``"exact"``, ``"approx"``, ``"lanczos"``.
    """
    op = ensure_linop(a)
    m = config.resolve_method("sqrt", op, method)

    if m in {"lanczos", "approx"}:
        # An explicit approximate request is honoured as given. Only when
        # `auto` resolution picked the Krylov path do we prefer a structured
        # exact factorisation if one happens to exist -- but not for a dense
        # `Matrix`, whose "exact" rule is a full O(n^3) Cholesky, which is
        # what the large-n heuristic is avoiding.
        if method == "auto" and not _is_dense_storage(op):
            try:
                return _lsqrt_impl(op)
            except NotImplementedError:
                pass
        return _functions_module.sqrt(op, method="lanczos", **kwargs)

    return _lsqrt_impl(op)


def log(a: LinearlyOperatorLike, **kwargs) -> LinearOperator:
    """Matrix logarithm."""
    return _functions_module.log(ensure_linop(a), **kwargs)


def exp(a: LinearlyOperatorLike, **kwargs) -> LinearOperator:
    """Matrix exponential."""
    return _functions_module.exp(ensure_linop(a), **kwargs)


def pow(a: LinearlyOperatorLike, p: float, **kwargs) -> LinearOperator:
    """Matrix power."""
    return _functions_module.pow(ensure_linop(a), p, **kwargs)


# Backward compatibility aliases
leigh = _leigh_impl
lsolve = _lsolve_impl
lsqrt = _lsqrt_impl
linverse = linverse


# --- Implementation details for wrappers ---


def inverse(a: LinearlyOperatorLike, method: str = "auto", **kwargs) -> LinearOperator:
    """Compute the inverse of a linear operator.

    Args:
        a: Linear operator.
        method: "exact" (lazy inverse) or "auto".
    """
    op = ensure_linop(a)
    m = config.resolve_method("inverse", op, method)

    if m == "exact":
        return linverse(op)

    # As in `sqrt`: when `auto` resolution picked the approximate path only
    # because the operator is large, still prefer an exact structured inverse
    # (or a lazy inverse backed by a structured solver) if one exists. An
    # explicit approximate request is honoured as given.
    # A dense `Matrix` also has an "exact" rule, but that is a full O(n^3)
    # `jnp.linalg.inv` -- exactly what the large-n heuristic is avoiding.
    if method == "auto" and not _is_dense_storage(op):
        exact = linverse(op)
        if type(exact) is not InverseLinearOperator or _has_structured_solver(op):
            return exact

    if m == "approx":
        # Default to LSMR for approximate inverse
        m = "lsmr"

    # Return lazy inverse with specified solver method
    return InverseLinearOperator(op, method=m, solver_options=kwargs)


def pinverse(a: LinearlyOperatorLike, method: str = "auto", **kwargs) -> LinearOperator:
    """Compute the pseudo-inverse of a linear operator."""
    op = ensure_linop(a)
    # Similar method resolution could apply
    return lpinverse(op)
