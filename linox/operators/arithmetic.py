# _arithmetic.py

r"""Arithmetic operations for linear operators.

This module implements various arithmetic operations for linear operators, including:

- :class:`ScaledLinearOperator`: Represents :math:`\alpha A` for scalar :math:`\alpha`
    and operator :math:`A`
- :class:`AddLinearOperator`: Represents :math:`A_1 + A_2 + \ldots + A_n` for
    operators :math:`A_i`
- :class:`ProductLinearOperator`: Represents :math:`A_1A_2\ldots A_n` for operators
    :math:`A_i`
- :class:`CongruenceTransform`: Represents :math:`ABA^T` for operators
    :math:`A` and :math:`B`
- :class:`TransposedLinearOperator`: Represents :math:`A^T` for operator :math:`A`
- :class:`InverseLinearOperator`: Represents :math:`A^{-1}` for operator :math:`A`

These operators can be combined to form complex linear transformations while maintaining
efficient computation through lazy evaluation.
"""

import operator
import warnings
from collections.abc import Iterable
from functools import reduce, wraps

import jax
import jax.numpy as jnp
import plum  # type: ignore

from linox import config, utils
from linox._types import ScalarLike, ShapeLike
from linox.config import warn as _warn
from linox.operators.base import LinearOperator
from linox.utils.array import default_floating_dtype

ArithmeticType = LinearOperator | jax.Array


# TODO: Move to utils
def _deprecated_l_prefix(func_name: str):
    """Create deprecation warning for functions with 'l' prefix."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"'{func_name}' is deprecated and will be removed in linox 0.0.3. Use '{func_name[1:]}' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return wrapper

    return decorator


# TODO: Move to utils
def _rhs_rows(b: jax.Array) -> int:
    if b.ndim == 1:
        return b.shape[0]
    if b.ndim >= 2:
        return b.shape[-2]
    msg = f"Unsupported rhs ndim: {b.ndim}"
    raise ValueError(msg)


# TODO: Move to utils
def _as_2d_rhs(b: jax.Array) -> tuple[jax.Array, bool]:
    """Return (b2d, squeeze) where b2d has shape (..., m, k)."""
    if b.ndim == 1:
        return b[:, None], True
    return b, False


# TODO: Move to utils
def x_in_y(x: LinearOperator, Y: tuple[type, ...]) -> bool:
    """Check if x is instance of Y or Scaled(Y)."""
    if isinstance(x, ScaledLinearOperator):
        return isinstance(x.operator, Y)
    return isinstance(x, Y)


# TODO: Move to utils
def get_scale_and_op(x: LinearOperator) -> tuple[jax.Array, LinearOperator]:
    """Return (scale, op) for x."""
    if isinstance(x, ScaledLinearOperator):
        return x.scalar, x.operator
    return jnp.array(1.0, dtype=x.dtype), x


# TODO: Move to utils
def smart_add(*operators: LinearOperator) -> LinearOperator:
    """Smart addition with simplification rules."""
    # Lazy imports to avoid circular dependencies
    from linox.operators.isotropic import (
        IsotropicAdditiveLinearOperator,
    )
    from linox.operators.lowrank import (
        IsotropicScalingPlusSymmetricLowRank,
        PositiveDiagonalPlusSymmetricLowRank,
        SymmetricLowRank,
    )
    from linox.operators.special import Identity, Scalar, Zero

    # 1. Flatten AddLinearOperators
    flat_ops = []
    for op in operators:
        op = utils.as_linop(op)
        if isinstance(op, AddLinearOperator):
            flat_ops.extend(op.operator_list)
        else:
            flat_ops.append(op)

    # 2. Filter Zeros
    non_zero_ops = [op for op in flat_ops if not isinstance(op, Zero)]

    if not non_zero_ops:
        # All were zero or empty, return Zero of correct shape
        # In practice shape must match, so use first op shape
        shape = operators[0].shape if operators else (0, 0)
        dtype = operators[0].dtype if operators else default_floating_dtype()
        return Zero(shape, dtype=dtype)

    if len(non_zero_ops) == 1:
        return non_zero_ops[0]

    # 3. Check for specific patterns (pair-wise for now or more complex)
    # Currently focusing on the 2-operand rewrites primarily

    if len(non_zero_ops) == 2:
        op1, op2 = non_zero_ops

        # Rewrites for Isotropic Add: s*I + A
        # Check if one is Scaled(Identity) and other is ANY symmetric operator

        # Identify which is Identity-like
        id_op = None
        other_op = None

        if x_in_y(op1, (Identity, Scalar)):
            id_op, other_op = op1, op2
        elif x_in_y(op2, (Identity, Scalar)):
            id_op, other_op = op2, op1

        if id_op is not None:
            s, _ = get_scale_and_op(id_op)
            if isinstance(id_op, Scalar):
                s = id_op.scalar

            if isinstance(other_op, SymmetricLowRank):
                # IsotropicScalingPlusSymmetricLowRank(scalar, U, S)
                return IsotropicScalingPlusSymmetricLowRank(s, other_op.U, other_op.S)

            return IsotropicAdditiveLinearOperator(s, other_op)

        # Rewrites for Diagonal + SymmetricLowRank -> PositiveDiagonalPlusSymmetricLowRank
        # Requires Diagonal to be positive, but we might just assume valid or check cheap?
        # If we have Diagonal + SymmetricLowRank, we can use the specialized class
        # which has efficient solves/determinants via Woodbury.
        # NOTE: Order matters. Check both op1+op2 and op2+op1.
        from linox.operators.diagonal import Diagonal

        diag_op = None
        lr_op = None

        if isinstance(op1, Diagonal) and isinstance(op2, (SymmetricLowRank, ScaledLinearOperator)):
            if isinstance(op2, SymmetricLowRank):
                diag_op, lr_op = op1, op2
                alpha = 1.0
            elif isinstance(op2, ScaledLinearOperator) and isinstance(op2.operator, SymmetricLowRank):
                diag_op, lr_op = op1, op2.operator
                alpha = op2.scalar

        elif isinstance(op2, Diagonal) and isinstance(op1, (SymmetricLowRank, ScaledLinearOperator)):
            if isinstance(op1, SymmetricLowRank):
                diag_op, lr_op = op2, op1
                alpha = 1.0
            elif isinstance(op1, ScaledLinearOperator) and isinstance(op1.operator, SymmetricLowRank):
                diag_op, lr_op = op2, op1.operator
                alpha = op1.scalar

        if diag_op is not None and lr_op is not None:
            # Check positivity of diagonal?
            # The class name is PositiveDiagonal..., implying it relies on D^-1 existing and D > 0.
            # We can check min(diag) > 0 ideally, or just optimistically use it
            # and let it fail/warn if negative?
            # For now, simplistic check or valid assumption.
            return PositiveDiagonalPlusSymmetricLowRank(diag_op, lr_op, low_rank_scale=alpha)

    # Default fallback
    return AddLinearOperator(*non_zero_ops)


# all arithmetic functions
@plum.dispatch
def ladd(a: LinearOperator, b: LinearOperator) -> LinearOperator:
    """Add two linear operators, applying structure-preserving rewrites."""
    return smart_add(a, b)


@ladd.dispatch
def _(a: LinearOperator, b: jax.Array) -> LinearOperator:
    return smart_add(a, utils.as_linop(b))


@plum.dispatch
def lsub(a: LinearOperator, b: LinearOperator) -> LinearOperator:
    """Subtract ``b`` from ``a``."""
    return AddLinearOperator(a, -b)


@lsub.dispatch
def _(a: LinearOperator, b: jax.Array) -> LinearOperator:
    return AddLinearOperator(a, -utils.as_linop(b))


@plum.dispatch
def lmul(a: ScalarLike | jax.Array, b: LinearOperator) -> LinearOperator:
    """Scale a linear operator by a scalar."""
    return ScaledLinearOperator(scalar=a, operator=b)


@plum.dispatch
def ldiv(a: LinearOperator, b: LinearOperator) -> LinearOperator:
    """Divide ``a`` by ``b`` elementwise (defined for diagonal-like operators)."""
    if len(a.shape) < 2 and len(b.shape) < 2:
        return a._todense() / b._todense()
    msg = f"Division only supported for Diagonal operators, got {type(a)} and {type(b)}"
    raise TypeError(msg)


@plum.dispatch
def lmatmul(a: LinearOperator, b: LinearOperator) -> ArithmeticType:
    """Compose two linear operators, applying structure-preserving rewrites."""
    return smart_matmul(a, b)


def smart_matmul(a: LinearOperator, b: LinearOperator) -> LinearOperator:
    """Smart matrix multiplication with simplification rules."""
    from linox.operators.arithmetic import TransposedLinearOperator
    from linox.operators.factor import PSDFromFactor

    # Check for PSDFromFactor: A @ A.T or A.T @ A
    # We check if b is the transpose of a (by structure or property)
    # A.T matches b?
    # Note: explicit .T on Matrix creates a NEW object, so 'is' check fails.
    # But TransposedLinearOperator(A) matches.
    # For Matrix, we might need to check underlying array (expensive?).
    # For now, handle TransposedLinearOperator structure.
    # Case 1: A @ A.T
    # If b is TransposedLinearOperator(A)
    if isinstance(b, TransposedLinearOperator) and b.operator is a:
        # If A @ A.T, L=A. PSDFromFactor(L) represents L L^T.
        # So return PSDFromFactor(a).
        return PSDFromFactor(a)

    # Case 2: A.T @ A (Gram matrix)
    # If a is TransposedLinearOperator(b)
    if isinstance(a, TransposedLinearOperator) and a.operator is b:
        # A.T @ A. L = A.T.
        # Return PSDFromFactor(a)
        return PSDFromFactor(a)

    # Case 3: Matrix/Dense checks?
    # Maybe check if a and b.T share buffer?
    # Skipping for now to avoid overhead.

    return ProductLinearOperator(a, b)


# @plum.dispatch
def lneg(a: LinearOperator) -> LinearOperator:
    """Negate a linear operator."""
    return ScaledLinearOperator(operator=a, scalar=-1)


@plum.dispatch
def lsqrt(a: LinearOperator) -> LinearOperator:
    """Return a factor ``S`` of ``a`` satisfying ``S @ S.T == a``."""
    msg = f"Square root of {type(a)} not implemented."
    raise NotImplementedError(msg)


# --------------------------------------------------------------------------- #
# Operations
# --------------------------------------------------------------------------- #


@plum.dispatch
def diagonal(a: LinearOperator) -> jax.Array:
    """Extract the diagonal of an operator as a :class:`jax.Array`."""
    _warn(f"Linear operator {a} is densed for diagonal computation.")
    dense_matrix = a._todense()
    if len(a.shape) <= 2:
        return jnp.diag(dense_matrix)
    n = dense_matrix.shape[-1]
    diag_indices = jnp.arange(n)
    return dense_matrix[..., diag_indices, diag_indices]


def transpose(a: LinearOperator) -> ArithmeticType:
    """Return the transpose of this operator."""
    return TransposedLinearOperator(a)


@plum.dispatch
def linverse(a: LinearOperator) -> ArithmeticType:
    """Return the inverse of an operator, lazily where possible."""
    return InverseLinearOperator(a)


@plum.dispatch
def lpinverse(a: LinearOperator) -> ArithmeticType:
    """Return the Moore-Penrose pseudo-inverse of an operator."""
    return PseudoInverseLinearOperator(a)


@plum.dispatch
def iso(scalar: ScalarLike, a: LinearOperator) -> LinearOperator:
    """Build the isotropic additive operator ``scalar * I + a``."""
    from linox.operators.isotropic import (
        IsotropicAdditiveLinearOperator,
    )

    return IsotropicAdditiveLinearOperator(scalar, a)


@plum.dispatch
def leigh(a: LinearOperator) -> tuple[jax.Array, LinearOperator]:
    """Eigendecomposition of a Hermitian operator, as ``(eigenvalues, Q)``."""
    with config.profile(
        kind="eigh",
        msg=f"eigh: {type(a).__name__}",
        op_type=type(a).__name__,
        op_id=id(a),
        shape=a.shape,
        dtype=a.dtype,
    ):
        _warn(f"Linear operator {a} is densed for leigh computation.")
        ev, evec = jnp.linalg.eigh(a._todense())
        return ev, utils.as_linop(evec)


@plum.dispatch
def svd(
    a: LinearOperator,
    *,
    full_matrices: bool = True,
    compute_uv: bool = True,
    k: int | None = None,
    num_iters: int | None = None,
    u0: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Singular Value Decomposition of a linear operator.

    When k is None, computes full SVD (may densify the operator).
    When k is provided, uses matrix-free Lanczos bidiagonalization to compute
    the k largest singular values/vectors efficiently.

    Args:
        a: Linear operator of shape (m, n)
        full_matrices: If True, return full-sized U and Vh matrices (only for full SVD)
        compute_uv: If True, compute U and Vh in addition to S (only for full SVD)
        k: If provided, compute only the k largest singular values/vectors using
            matrix-free methods. This is efficient for large matrices. If None,
            compute full SVD (default: None)
        num_iters: Number of Lanczos iterations for partial SVD. Should be larger
            than k. If None, uses min(2*k, min(m, n)). Only used when k is provided.
        u0: Initial vector for Lanczos bidiagonalization. Only used when k is provided.

    Returns
    -------
        U: Left singular vectors - shape (m, m) for full SVD or (m, k) for partial
        S: Singular values in descending order - shape (min(m,n),) or (k,)
        Vt: Right singular vectors (transposed) - shape (n, n) for full or (k, n)

    Examples
    --------
        Full SVD (may densify):
        >>> U, S, Vt = svd(operator)

        Partial SVD (matrix-free, recommended for large matrices):
        >>> U, S, Vt = svd(operator, k=10)
        >>> # Only top 10 singular values/vectors computed efficiently

    Notes
    -----
        - For large matrices, use k parameter to avoid densification
        - Structure-exploiting dispatches exist for Kronecker and other special operators
        - Partial SVD uses Lanczos bidiagonalization (Golub-Kahan process)
    """
    with config.profile(
        kind="svd",
        msg=f"svd: {type(a).__name__}",
        op_type=type(a).__name__,
        op_id=id(a),
        shape=a.shape,
        dtype=a.dtype,
        meta={"k": k},
    ):
        if k is not None:
            # Use matrix-free partial SVD
            from linox.linalg.spectral import svd_partial

            return svd_partial(a, k, num_iters, u0)

        # Full SVD - may densify
        _warn(f"Linear operator {a} is densed for svd computation.")
        return jax.scipy.linalg.svd(
            a._todense(),
            full_matrices=full_matrices,
            compute_uv=compute_uv,
        )


@plum.dispatch
def lsvd(
    a: LinearOperator,
    k: int,
    num_iters: int | None = None,
    u0: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute partial SVD using matrix-free Lanczos bidiagonalization.

    .. deprecated::
        Use :func:`svd` with ``k`` parameter instead: ``svd(a, k=k)``.
        This function will be removed in a future version.

    Computes the k largest singular values and corresponding singular vectors
    without forming the full matrix explicitly. This is efficient for large
    sparse or structured matrices where only a few singular values are needed.

    Args:
        a: Linear operator of shape (m, n)
        k: Number of singular values/vectors to compute
        num_iters: Number of Lanczos iterations. Should be larger than k.
            If None, uses min(2*k, min(m, n)). Default is None.
        u0: Initial vector of shape (m,) for bidiagonalization.
            If None, uses vector of ones. Default is None.

    Returns
    -------
        U: Left singular vectors of shape (m, k)
        S: Singular values of shape (k,) in descending order
        Vt: Right singular vectors of shape (k, n)

    Examples
    --------
        >>> import jax.numpy as jnp
        >>> from linox import Matrix
        >>> A = Matrix(jnp.random.randn(1000, 500))
        >>> # Deprecated:
        >>> U, S, Vt = linox.lsvd(A, k=10)
        >>> # Use instead:
        >>> U, S, Vt = linox.svd(A, k=10)

    Notes
    -----
        **Deprecated:** Use ``svd(a, k=k)`` instead.

        This is the matrix-free alternative to jnp.linalg.svd for computing
        a few singular values.

        Inspired by matfree library (https://github.com/pnkraemer/matfree).
    """
    warnings.warn(
        "lsvd is deprecated and will be removed in a future version. Use svd(a, k=k) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return svd(a, k=k, num_iters=num_iters, u0=u0)


@plum.dispatch
def lqr(a: LinearOperator) -> tuple[jax.Array, jax.Array]:
    """QR decomposition of a linear operator.

    Returns
    -------
        Q: Orthogonal matrix
        R: Upper triangular matrix.
    """
    warnings.warn(
        "lqr is deprecated and will be removed in a future version. Use qr(a) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    _warn(f"Linear operator {a} is densed for lqr computation.")
    return jnp.linalg.qr(a._todense())


@plum.dispatch
def lsolve(a: LinearOperator, b: jax.Array, method: str = "exact", **kwargs) -> jax.Array:
    """Solve linear system Ax = b.

    Args:
        a: Linear operator
        b: Right hand side
        method: "exact" (dense), "lsmr", "cg".
    """
    with config.profile(
        kind="solve",
        msg=f"solve: {type(a).__name__} (method={method})",
        op_type=type(a).__name__,
        op_id=id(a),
        shape=a.shape,
        dtype=a.dtype,
        meta={"rhs_shape": b.shape, "method": method},
    ):
        if method == "lsmr":
            from linox.linalg.approx.lsmr import lsmr_solve

            x, _ = lsmr_solve(a, b, **kwargs)
            return x

        if method in {"cg", "conjugate_gradient"}:
            # Use JAX CG (requires a linear operator or function)
            # jax.scipy.sparse.linalg.cg expects A to show typical behavior.
            # LinearOperator implements @ / __matmul__.
            x, _ = jax.scipy.sparse.linalg.cg(a, b, **kwargs)
            return x

        if a.shape[-1] != _rhs_rows(b):
            msg = f"Shape mismatch: {a.shape} and {b.shape}"
            raise ValueError(msg)

        _warn(f"Linear operator {a} is densed for lsolve computation.")
        return jax.scipy.linalg.solve(a._todense(), b, assume_a="sym")


@lsolve.dispatch
def _(a: LinearOperator, b: LinearOperator) -> jax.Array:
    """Solve Ax = B where B is a LinearOperator by converting B to dense."""
    _warn(f"Linear operator {b} is densed for lsolve computation.")
    return lsolve(a, b._todense())
    """Solve the linear system Ax = b."""
    if a.shape[-1] != b.shape[0]:
        msg = f"Shape mismatch: {a.shape} and {b.shape}"
        raise ValueError(msg)
    # print(f"Warning: Linear operator {a} is densed for lsolve computation.")
    # return jax.scipy.linalg.solve(a.todense(), b, assume_a="sym")
    return linverse(a) @ b


@plum.dispatch
def lu_factor(
    a: LinearOperator,
    overwrite_a: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """LU factorization of a linear operator."""
    _warn(f"Linear operator {a} is densed for lu_factor computation.")
    return jax.scipy.linalg.lu_factor(a._todense(), overwrite_a=overwrite_a)


@plum.dispatch
def lu_solve(a: LinearOperator, b: jax.Array) -> jax.Array:
    """Solve the linear system Ax = b given the LU factorization of A."""
    if a.shape[-1] != _rhs_rows(b):
        msg = f"Shape mismatch: {a.shape} and {b.shape}"
        raise ValueError(msg)
    lu, piv = lu_factor(a)
    _warn(f"Linear operator {a} is densed for lu_solve computation.")
    return jax.scipy.linalg.lu_solve((lu, piv), b, overwrite_b=False)


@plum.dispatch
def lpsolve(a: LinearOperator, b: jax.Array, rtol=1e-8) -> jax.Array:
    """Solve the linear system Ax = b."""
    # if a.shape[-1] != _rhs_rows(b):
    #     msg = f"Shape mismatch: {a.shape} and {b.shape}"
    #     raise ValueError(msg)

    # return jnp.linalg.pinv(a._todense(), rtol) @ b
    b2d, squeeze = _as_2d_rhs(jnp.asarray(b))

    # IMPORTANT: b lives in the *row space* of A, so rows(b) must match m = shape[-2]
    if _rhs_rows(b2d) != a.shape[-2]:
        msg = f"Shape mismatch: {a.shape} and {b2d.shape}"
        raise ValueError(msg)

    x = jnp.linalg.pinv(a._todense(), rtol=rtol) @ b2d
    return x[..., 0] if squeeze else x


@plum.dispatch
def lcholesky(a: LinearOperator) -> jax.Array:
    """Cholesky decomposition of a linear operator."""
    _warn(f"Linear operator {a} is densed for lcholesky computation.")
    return jnp.linalg.cholesky(a._todense())


@plum.dispatch
def ldet(a: LinearOperator) -> jax.Array:
    """Compute the determinant of a linear operator."""
    if not is_square(a):
        msg = f"Operator {a} is not square."
        raise ValueError(msg)
    _warn(f"Linear operator {a} is densed for ldet computation.")
    return jnp.linalg.det(a._todense())


@plum.dispatch
def slogdet(a: LinearOperator) -> tuple[jax.Array, jax.Array]:
    """Compute the sign and log determinant of a linear operator.

    Returns
    -------
        sign: Sign of the determinant
        logdet: Logarithm of the determinant
    """
    if not is_square(a):
        msg = f"Operator {a} is not square."
        raise ValueError(msg)
    _warn(f"Linear operator {a} is densed for slogdet computation.")
    return jnp.linalg.slogdet(a._todense())


@plum.dispatch
def kron(a: LinearOperator, b: LinearOperator) -> LinearOperator:
    """Return the Kronecker product ``a (x) b``."""
    from linox.operators.kron import Kronecker

    return Kronecker(a, b)


@plum.dispatch
def ltrace(
    a: LinearOperator,
    key: jax.Array | None = None,
    num_samples: int = 100,
    distribution: str = "rademacher",
) -> tuple[jax.Array, jax.Array]:
    """Estimate the trace of a linear operator using Hutchinson's method.

    For large operators, this provides a stochastic estimate without densification.

    Args:
        a: Linear operator to compute trace of
        key: JAX random key. If None, uses a default key
        num_samples: Number of random samples for estimation
        distribution: Either 'rademacher' or 'normal'

    Returns
    -------
        trace_estimate: Monte Carlo estimate of trace(a)
        trace_std: Standard error of the estimate

    Notes
    -----
        For exact trace computation on small/dense operators, use jnp.trace(a.todense()).
        This method is inspired by matfree library (https://github.com/pnkraemer/matfree).
    """
    from linox.linalg.approx.hutchinson import hutchinson_trace

    if key is None:
        key = jax.random.PRNGKey(0)

    return hutchinson_trace(a, key, num_samples, distribution)


@plum.dispatch
def lexp(
    a: LinearOperator,
    v: jax.Array | None = None,
    num_iters: int = 20,
    method: str = "lanczos",
) -> jax.Array | LinearOperator:
    """Compute matrix exponential exp(A) or exp(A)v.

    For large operators, computes exp(A)v using Krylov subspace methods
    without forming exp(A) explicitly.

    Args:
        a: Linear operator (should be symmetric for Lanczos)
        v: Vector to multiply by exp(A). If None, returns lazy operator
        num_iters: Number of Krylov iterations
        method: 'lanczos' for symmetric or 'arnoldi' for general

    Returns
    -------
        If v is provided: exp(A) @ v
        If v is None: Lazy linear operator representing exp(A)

    Notes
    -----
        Inspired by matfree library (https://github.com/pnkraemer/matfree).
    """
    from linox.linalg.approx.arnoldi import arnoldi_matrix_function
    from linox.linalg.approx.lanczos import lanczos_matrix_function

    if v is None:
        # Return lazy operator
        _warn("lexp without vector returns lazy operator - evaluation may be expensive")
        # For now, densify (future: could implement lazy MatrixFunctionOperator)
        return jax.scipy.linalg.expm(a.todense())

    # Compute exp(A)v using Krylov methods
    if method == "lanczos":
        return lanczos_matrix_function(a, v, jnp.exp, num_iters, reortho=True)
    if method == "arnoldi":
        return arnoldi_matrix_function(a, v, jnp.exp, num_iters)
    msg = f"Unknown method: {method}. Use 'lanczos' or 'arnoldi'."
    raise ValueError(msg)


@plum.dispatch
def llog(
    a: LinearOperator,
    v: jax.Array | None = None,
    num_iters: int = 20,
    method: str = "lanczos",
) -> jax.Array | LinearOperator:
    """Compute matrix logarithm log(A) or log(A)v.

    For large operators, computes log(A)v using Krylov subspace methods.

    Args:
        a: Linear operator (should be positive definite and symmetric for Lanczos)
        v: Vector to multiply by log(A). If None, returns lazy operator
        num_iters: Number of Krylov iterations
        method: 'lanczos' for symmetric or 'arnoldi' for general

    Returns
    -------
        If v is provided: log(A) @ v
        If v is None: Lazy linear operator representing log(A)

    Notes
    -----
        Inspired by matfree library (https://github.com/pnkraemer/matfree).
    """
    from linox.linalg.approx.arnoldi import arnoldi_matrix_function
    from linox.linalg.approx.lanczos import lanczos_matrix_function

    if v is None:
        _warn("llog without vector returns lazy operator - evaluation may be expensive")
        # For now, densify
        eigvals, eigvecs = jnp.linalg.eigh(a.todense())
        return eigvecs @ jnp.diag(jnp.log(eigvals)) @ eigvecs.T

    # Compute log(A)v using Krylov methods
    if method == "lanczos":
        return lanczos_matrix_function(a, v, jnp.log, num_iters, reortho=True)
    if method == "arnoldi":
        return arnoldi_matrix_function(a, v, jnp.log, num_iters)
    msg = f"Unknown method: {method}. Use 'lanczos' or 'arnoldi'."
    raise ValueError(msg)


@plum.dispatch
def lpow(
    a: LinearOperator,
    *,
    power: float,
    v: jax.Array | None = None,
    num_iters: int = 20,
    method: str = "lanczos",
) -> jax.Array | LinearOperator:
    """Compute matrix power A^p or (A^p)v.

    For large operators, computes A^p @ v using Krylov subspace methods.

    Args:
        a: Linear operator (should be symmetric for Lanczos)
        power: Exponent p
        v: Vector to multiply by A^p. If None, returns lazy operator
        num_iters: Number of Krylov iterations
        method: 'lanczos' for symmetric or 'arnoldi' for general

    Returns
    -------
        If v is provided: A^p @ v
        If v is None: Lazy linear operator representing A^p

    Notes
    -----
        Inspired by matfree library (https://github.com/pnkraemer/matfree).
    """
    from linox.linalg.approx.arnoldi import arnoldi_matrix_function
    from linox.linalg.approx.lanczos import lanczos_matrix_function

    # Define power function (element-wise on eigenvalues)
    def power_func(eigvals):
        return eigvals**power

    if v is None:
        _warn("lpow without vector returns lazy operator - evaluation may be expensive")
        # For now, densify
        eigvals, eigvecs = jnp.linalg.eigh(a.todense())
        return eigvecs @ jnp.diag(power_func(eigvals)) @ eigvecs.T

    # Compute A^p @ v using Krylov methods
    if method == "lanczos":
        return lanczos_matrix_function(a, v, power_func, num_iters, reortho=True)
    if method == "arnoldi":
        return arnoldi_matrix_function(a, v, power_func, num_iters)
    msg = f"Unknown method: {method}. Use 'lanczos' or 'arnoldi'."
    raise ValueError(msg)


# --------------------------------------------------------------------------- #
# Linear Operator checks
# --------------------------------------------------------------------------- #


def is_square(a: LinearOperator) -> bool:
    """Whether the operator has equal row and column counts."""
    return a.shape[-1] == a.shape[-2]


def is_symmetric(
    a: LinearOperator,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    key: jax.Array | None = None,
    num_probes: int = 1,
) -> bool:
    """Check if a linear operator is symmetric without densifying.

    Uses randomized probing: generates random vectors x and checks if Ax ≈ A^T x.
    This avoids densifying the full matrix.

    Args:
        a: Linear operator to check
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison
        key: Random key for generating test vectors (default: uses key 0)
        num_probes: Number of random vectors to test (default: 1)

    Returns
    -------
        True if the operator appears symmetric within tolerance
    """
    if not is_square(a):
        return False

    if key is None:
        key = jax.random.PRNGKey(0)

    n = a.shape[-1]

    for i in range(num_probes):
        # Generate random normalized vector
        probe_key = jax.random.fold_in(key, i)
        x = jax.random.normal(probe_key, (n,), dtype=a.dtype)
        x /= jnp.linalg.norm(x)

        # Compute Ax and A^T x
        v1 = a @ x
        v2 = a.T @ x

        # Check if v1 ≈ v2
        if not jnp.allclose(v1, v2, rtol=rtol, atol=atol):
            return False

    return True


def is_hermitian(
    a: LinearOperator,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    key: jax.Array | None = None,
    num_probes: int = 1,
) -> bool:
    """Check if a linear operator is Hermitian without densifying.

    Uses randomized probing: generates random vectors x and checks if Ax ≈ (A^H x)
    where A^H is the conjugate transpose. This avoids densifying the full matrix.

    For real matrices, this is equivalent to checking symmetry.

    Args:
        a: Linear operator to check
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison
        key: Random key for generating test vectors (default: uses key 0)
        num_probes: Number of random vectors to test (default: 1)

    Returns
    -------
        True if the operator appears Hermitian within tolerance
    """
    if not is_square(a):
        return False

    if key is None:
        key = jax.random.PRNGKey(0)

    n = a.shape[-1]

    for i in range(num_probes):
        # Generate random normalized vector
        probe_key = jax.random.fold_in(key, i)
        x = jax.random.normal(probe_key, (n,), dtype=a.dtype)
        if jnp.issubdtype(a.dtype, jnp.complexfloating):
            # For complex operators, add imaginary part
            x_imag = jax.random.normal(probe_key, (n,), dtype=a.dtype)
            x += 1j * x_imag
        x /= jnp.linalg.norm(x)

        # Compute Ax and A^H x (conjugate transpose)
        v1 = a @ x
        a.T @ jnp.conj(x)

        # For Hermitian: <Ax, x> = <x, Ax> = conj(<Ax, x>)
        # Equivalently: Ax should equal conj(A^T conj(x))
        # Or more directly: check if <v1, x> ≈ conj(<v2, x>)
        # But simpler: check if v1 ≈ conj(v2) when x is real
        # Actually, let's use the proper test: A x = conj(A^T conj(x))
        v2_hermitian = jnp.conj(a.T @ jnp.conj(x))

        if not jnp.allclose(v1, v2_hermitian, rtol=rtol, atol=atol):
            return False

    return True


# --------------------------------------------------------------------------- #
# Linear Operator - Enforce tags
# --------------------------------------------------------------------------- #


def symmetrize(a: LinearOperator) -> ArithmeticType:
    """Return the symmetric part ``(a + a.T) / 2`` of an operator."""
    return 0.5 * (a + a.transpose())


# --------------------------------------------------------------------------- #
# Dispatch - lmatmul
# --------------------------------------------------------------------------- #


@lmatmul.dispatch
def _(a: LinearOperator, b: jax.Array) -> jax.Array:
    with config.profile(
        kind="matmul",
        msg=f"matmul: {type(a).__name__} @ array",
        op_type=type(a).__name__,
        op_id=id(a),
        shape=getattr(a, "shape", None),
        dtype=getattr(a, "dtype", None),
        meta={"rhs_shape": b.shape, "rhs_dtype": str(b.dtype)},
    ):
        return a._matmul(b)


@lmatmul.dispatch
def _(a: jax.Array, b: LinearOperator) -> LinearOperator:
    from linox.operators.dense import Matrix

    return Matrix(a) @ b


# --------------------------------------------------------------------------- #
# Linear Operators
# --------------------------------------------------------------------------- #


class ScaledLinearOperator(LinearOperator):
    r"""Linear operator scaled with a scalar.

    For a linear operator :math:`A` and scalar :math:`\alpha`, this represents
    :math:`\alpha A` where :math:`(\alpha A)x = \alpha(Ax)` for any vector :math:`x`

    Args:
        operator: A linear operator to be scaled
        scalar: A scalar value to multiply the operator with
    """

    def __init__(self, operator: LinearOperator, scalar: ScalarLike) -> None:
        self.operator = utils.as_linop(operator)
        scalar = jnp.asarray(scalar)
        # Promote against the array, not `scalar.dtype`: the dtype alone drops
        # JAX's weak typing, so a Python `2.0` would upcast a float32 operator
        # to float64 under x64. The operator's dtype is the user's choice.
        dtype = jnp.result_type(self.operator.dtype, scalar)
        self.scalar = utils.as_scalar(scalar, dtype)
        super().__init__(shape=operator.shape, dtype=dtype)

    @property
    def is_symmetric(self) -> bool:
        """Whether the scaled operator is symmetric."""
        return self.operator.is_symmetric

    @property
    def is_psd(self) -> bool:
        # Check if scalar >= 0 and operator is PSD
        # Note: robust checking of scalar value requires jax.Array value access?
        # If scalar is tracer, this might fail or return tracer.
        # Introspection is usually for planning (outside JIT), so concrete values expected.
        # But here scalar is stored as array/scalar.
        # For safety, we can try converting to float if it's not traced.
        # A traced scalar has no concrete sign, so make no claim rather than
        # guessing. (A bare `except` here previously swallowed every error.)
        """Whether the scaled operator is positive semi-definite."""
        try:
            s = float(self.scalar)
        except (jax.errors.ConcretizationTypeError, TypeError):
            return False
        return s >= 0 and self.operator.is_psd

    def _matmul(self, arr: jax.Array) -> jax.Array:
        return (self.operator @ arr) * self.scalar

    def _todense(self) -> jax.Array:
        return self.operator._todense() * self.scalar

    def transpose(self) -> LinearOperator:
        """Return the transpose of this operator."""
        return self.scalar * self.operator.transpose()

    def tree_flatten(self) -> tuple[tuple[any, ...], dict[str, any]]:
        """Flatten this operator into JAX pytree children and static data."""
        children = (self.operator, self.scalar)
        aux_data = {}
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: dict[str, any],
        children: tuple[any, ...],
    ) -> "ScaledLinearOperator":
        """Reconstruct this operator from JAX pytree children and static data."""
        del aux_data
        operator, scalar = children
        return cls(operator=operator, scalar=scalar)


@lsqrt.dispatch
def _(a: ScaledLinearOperator) -> LinearOperator:
    return ScaledLinearOperator(lsqrt(a.operator), jnp.sqrt(a.scalar))


@diagonal.dispatch(precedence=1)
def _(a: ScaledLinearOperator) -> jax.Array:
    return jnp.asarray(a.scalar) * diagonal(a.operator)


@linverse.dispatch
def _(a: ScaledLinearOperator) -> LinearOperator:
    return ScaledLinearOperator(linverse(a.operator), 1 / a.scalar)


@lpinverse.dispatch
def _(a: ScaledLinearOperator) -> LinearOperator:
    return ScaledLinearOperator(lpinverse(a.operator), 1 / a.scalar)


@lsolve.dispatch
def _(a: ScaledLinearOperator, b: jax.Array) -> jax.Array:
    return lsolve(a.operator, b) / a.scalar


@lpsolve.dispatch
def _(a: ScaledLinearOperator, b: jax.Array, rtol=1e-8) -> jax.Array:
    return lpsolve(a.operator, b) / a.scalar


@lcholesky.dispatch
def _(a: ScaledLinearOperator) -> ScaledLinearOperator:
    return ScaledLinearOperator(lcholesky(a.operator), jnp.sqrt(a.scalar))


@ldet.dispatch
def _(a: ScaledLinearOperator) -> jax.Array:
    """Compute the determinant of a scaled linear operator."""
    if not is_square(a):
        msg = f"Operator {a} is not square."
        raise ValueError(msg)

    return a.scalar ** a.shape[-1] * ldet(a.operator)


@slogdet.dispatch
def _(a: ScaledLinearOperator) -> tuple[jax.Array, jax.Array]:
    """Compute the sign and log determinant of a scaled linear operator."""
    if not is_square(a):
        msg = f"Operator {a} is not square."
        raise ValueError(msg)

    sign, logdet = slogdet(a.operator)
    return sign, logdet + (jnp.log(a.scalar) * a.shape[-1])


@leigh.dispatch
def _(a: ScaledLinearOperator) -> tuple[LinearOperator, LinearOperator]:
    lams, Q = leigh(a.operator)
    return a.scalar * lams, Q


# inverse special behavior:
# ScaledLinearOperator(inverse(operator), inverse(Scalar))
def _broadcast_shapes(shapes: Iterable[ShapeLike]) -> ShapeLike:
    try:
        return jnp.broadcast_shapes(*shapes)
    except ValueError:
        msg = f"Shapes {shapes} cannot be broadcasted."
        raise ValueError(msg)  # noqa: B904


class AddLinearOperator(LinearOperator):
    r"""A linear operator formed by adding two or more linear operators together.

    For linear operators :math:`A_1, A_2, \ldots, A_n`, this represents
    :math:`A_1 + A_2 + \ldots + A_n`where
    :math:`(A_1 + A_2 + \ldots + A_n)x = A_1x + A_2x + \ldots + A_nx`
    for any vector :math:`x`

    Args:
        *operator_list: Variable number of linear operators to be added
    """

    def __init__(self, *operator_list: ArithmeticType) -> None:
        self.operator_list = [
            utils.as_linop(o) if isinstance(op, AddLinearOperator) else utils.as_linop(op)
            for op in operator_list
            for o in (op.operator_list if isinstance(op, AddLinearOperator) else [op])
        ]
        shape = _broadcast_shapes([op.shape for op in self.operator_list])
        super().__init__(shape=shape, dtype=self.operator_list[0].dtype)

    def _matmul(self, arr: jax.Array) -> jax.Array:
        return reduce(
            operator.add,
            (op @ arr for op in reversed(self.operator_list)),
        )

    def _todense(self) -> jax.Array:
        return reduce(operator.add, (op._todense() for op in self.operator_list))

    def transpose(self) -> "AddLinearOperator":
        """Return the transpose of this operator."""
        return AddLinearOperator(*(op.transpose() for op in self.operator_list))

    def tree_flatten(self) -> tuple[tuple[any, ...], dict[str, any]]:
        """Flatten this operator into JAX pytree children and static data."""
        children = tuple(self.operator_list)
        aux_data = {}
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: dict[str, any],
        children: tuple[any, ...],
    ) -> "AddLinearOperator":
        """Reconstruct this operator from JAX pytree children and static data."""
        del aux_data
        return cls(*children)


@diagonal.dispatch
def _(a: AddLinearOperator) -> jax.Array:
    return reduce(operator.add, (diagonal(op) for op in a.operator_list))


@diagonal.dispatch(precedence=1)
def _(a: "ProductLinearOperator") -> jax.Array:
    """Diagonal of a product.

    If all factors are diagonal-like (Diagonal, Identity, Scalar, and their
    scaled variants), the product remains diagonal and the diagonal equals the
    element-wise product of the factor diagonals. Otherwise, fall back to a
    dense computation for correctness.
    """

    def _is_diag_like(op: LinearOperator) -> bool:
        from linox.operators.diagonal import Diagonal as _Diag
        from linox.operators.special import Identity as _Id
        from linox.operators.special import Scalar as _Scal

        if isinstance(op, (_Diag, _Id, _Scal)):
            return True
        if isinstance(op, ScaledLinearOperator):
            return isinstance(op.operator, (_Diag, _Id, _Scal))
        return False

    if all(_is_diag_like(op) for op in a.operator_list):
        diags = []
        for op in a.operator_list:
            if isinstance(op, ScaledLinearOperator):
                diags.append(jnp.asarray(op.scalar) * jnp.asarray(diagonal(op.operator)))
            else:
                diags.append(jnp.asarray(diagonal(op)))
        return reduce(operator.mul, diags)

    # Fallback: compute dense diagonal for correctness
    _warn(f"Converting product of shape {a.shape} to dense array for diagonal.")
    dense = a._todense()
    if dense.ndim <= 2:
        return jnp.diag(dense)
    n = dense.shape[-1]
    idx = jnp.arange(n)
    return dense[..., idx, idx]


class ProductLinearOperator(LinearOperator):
    r"""Product of linear operators.

    For linear operators :math:`A_1, A_2, \ldots, A_n`, this represents
    :math:`A_1A_2\ldots A_n` where :math:`(A_1A_2\ldots A_n)x = A_1(A_2(\ldots(A_nx)))`
    for any vector :math:`x`

    Args:
        *operator_list: Variable number of linear operators to be multiplied
    """

    def __init__(self, *operator_list: LinearOperator) -> None:
        self.operator_list = [
            utils.as_linop(o) if isinstance(op, ProductLinearOperator) else utils.as_linop(op)
            for op in operator_list
            for o in (op.operator_list if isinstance(op, ProductLinearOperator) else [op])
        ]
        batch_shape = _broadcast_shapes([op.shape[:-2] for op in self.operator_list])
        self.__check_init__()
        result_dtype = jnp.result_type(*[op.dtype for op in self.operator_list])
        shape = utils.as_shape(
            (
                *batch_shape,
                self.operator_list[0].shape[-2],
                self.operator_list[-1].shape[-1],
            )
        )
        super().__init__(shape=shape, dtype=result_dtype)

    def __check_init__(self) -> None:
        for i, op1 in enumerate(self.operator_list[:-1]):
            op2 = self.operator_list[i + 1]
            if op1.shape[-1] != op2.shape[-2]:
                msg = f"Shape mismatch: Cannot multiply linear operators with shapes operator 1: ({op1.shape}) operator 2: ({op2.shape})"
                raise ValueError(msg)

    def _matmul(self, arr: jax.Array) -> jax.Array:
        return reduce(lambda x, y: y @ x, [arr, *reversed(self.operator_list)])

    def transpose(self) -> "ProductLinearOperator":
        """Return the transpose of this operator."""
        return ProductLinearOperator(*(op.transpose() for op in reversed(self.operator_list)))

    def _todense(self) -> jax.Array:
        return reduce(
            lambda x, y: y @ x,
            [
                self.operator_list[-1]._todense(),
                *reversed([op._todense() for op in self.operator_list[:-1]]),
            ],
        )

    def tree_flatten(self) -> tuple[tuple[any, ...], dict[str, any]]:
        """Flatten this operator into JAX pytree children and static data."""
        children = tuple(self.operator_list)
        aux_data = {}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: dict[str, any], children: tuple[any, ...]) -> "ProductLinearOperator":
        """Reconstruct this operator from JAX pytree children and static data."""
        del aux_data
        return cls(*children)


# not properly tested
class CongruenceTransform(ProductLinearOperator):
    r"""Congruence transformation of linear operators.

    For linear operators :math:`A` and :math:`B`, this represents :math:`ABA^T`
    where :math:`(ABA^T)x = A(B(A^T x))` for any vector :math:`x`

    Args:
        A: First linear operator
        B: Second linear operator
    """

    def __init__(self, A: ArithmeticType, B: ArithmeticType) -> None:
        self._A = utils.as_linop(A)
        self._B = utils.as_linop(B)

        super().__init__(self._A, self._B, self._A.T)

    def transpose(self) -> LinearOperator:
        """Return the transpose of this operator."""
        return CongruenceTransform(self._A, self._B.T)

    def tree_flatten(self) -> tuple[tuple[any, ...], dict[str, any]]:
        """Flatten this operator into JAX pytree children and static data."""
        children = (self._A, self._B)
        aux_data = {}
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: dict[str, any],
        children: tuple[any, ...],
    ) -> "CongruenceTransform":
        """Reconstruct this operator from JAX pytree children and static data."""
        del aux_data
        A, B = children
        return cls(A=A, B=B)


@plum.dispatch
def congruence_transform(A: ArithmeticType, B: ArithmeticType) -> LinearOperator:
    """Return the congruence transform ``A @ B @ A.T``."""
    return CongruenceTransform(A, B)


@congruence_transform.dispatch
def _(A: ArithmeticType, B: ScaledLinearOperator) -> LinearOperator:
    # A (cB) A^T = c (A B A^T): pull the scalar out so the inner transform can
    # pick up whatever structured rule exists for ``B.operator``.
    return B.scalar * congruence_transform(A, B.operator)


@diagonal.dispatch(precedence=5)
def _(a: CongruenceTransform) -> jax.Array:
    r"""Diagonal of :math:`A B A^\top` without forming the product.

    Sums (e.g. ``IsotropicScalingPlusSymmetricLowRank``) and scaled operators
    are split up so each piece can use its own structured rule. Otherwise
    ``A`` is densified (shape ``(n, k)``) and
    :math:`\operatorname{diag}(A B A^\top)_i = \sum_j A_{ij} (B A^\top)_{ji}`
    is evaluated at :math:`O(n k^2)` cost, never materialising the
    ``(n, n)`` product or a dense ``B``.
    """
    B = a._B
    if isinstance(B, AddLinearOperator):
        return reduce(operator.add, (diagonal(congruence_transform(a._A, op)) for op in B.operator_list))
    if isinstance(B, ScaledLinearOperator):
        return jnp.asarray(B.scalar) * diagonal(congruence_transform(a._A, B.operator))
    from linox.operators.dense import Matrix  # circular import

    # A dense-storage ``A`` is used as is; anything else has to be densified.
    A = a._A.A if isinstance(a._A, Matrix) else a._A.todense()
    BAt = B @ jnp.swapaxes(A, -1, -2)
    return jnp.sum(A * jnp.swapaxes(BAt, -1, -2), axis=-1)


class TransposedLinearOperator(LinearOperator):
    r"""Transpose of a linear operator.

    For a linear operator :math:`A`, this represents :math:`A^T`
    where :math:`(A^T)_{ij} = A_{ji}` for all :math:`i,j`

    Args:
        operator: A linear operator to be transposed
    """

    def __init__(self, operator: LinearOperator) -> None:
        self.operator = utils.as_linop(operator)
        batch_shape = operator.shape[:-2]
        super().__init__(
            shape=(*batch_shape, operator.shape[-1], operator.shape[-2]),
            dtype=operator.dtype,
        )

    def _matmul(self, arr: jnp.array) -> jax.Array:
        # Derive the adjoint from the forward matvec instead of asking the
        # wrapped operator to transpose itself. `LinearOperator.transpose`
        # materialises the dense matrix by default, so the old
        # `self.operator.transpose() @ arr` densified any operator that did
        # not override it -- exactly the operators a matrix-free library must
        # not densify.
        #
        # This wrapper is only ever constructed when there is *no* structured
        # transpose (`LinearOperator.T` returns the subclass's own transpose
        # when it provides one), so nothing structured loses its fast path.
        arr = jnp.asarray(arr)
        n = self.operator.shape[-1]

        # `linear_transpose` requires the cotangent's dtype to match the
        # forward output's exactly, so promote both to a common dtype rather
        # than silently narrowing a float64 rhs onto a float32 operator.
        dtype = jnp.promote_types(self.operator.dtype, arr.dtype)
        basis = jnp.zeros((n,), dtype=dtype)

        def forward(v: jax.Array) -> jax.Array:
            return self.operator @ v

        out_dtype = jax.eval_shape(forward, basis).dtype

        def adjoint(col: jax.Array) -> jax.Array:
            return jax.linear_transpose(forward, basis)(col.astype(out_dtype))[0]

        if arr.ndim == 1:
            return adjoint(arr)
        return jax.vmap(adjoint, in_axes=-1, out_axes=-1)(arr)

    def _todense(self) -> jax.Array:
        return self.operator._todense().swapaxes(-1, -2)

    def transpose(self) -> LinearOperator:
        """Return the transpose of this operator."""
        return self.operator

    def tree_flatten(self) -> tuple[tuple[any, ...], dict[str, any]]:
        """Flatten this operator into JAX pytree children and static data."""
        children = (self.operator,)
        aux_data = {}
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: dict[str, any],
        children: tuple[any, ...],
    ) -> "TransposedLinearOperator":
        """Reconstruct this operator from JAX pytree children and static data."""
        del aux_data
        (operator,) = children
        return cls(operator=operator)


@leigh.dispatch
def _(a: TransposedLinearOperator) -> tuple[LinearOperator, LinearOperator]:
    lams, Q = leigh(a.operator)
    return lams, Q.transpose()


class InverseLinearOperator(LinearOperator):
    """Inverse of a linear operator.

    For a linear operator :math:`A`, this represents :math:`A^{-1}`
    where :math:`A^{-1}` is the unique operator such that :math:`AA^{-1} = A^{-1}A = I`
    where :math:`I` is the identity operator

    Args:
        operator: A linear operator to be inverted
    """

    def __init__(
        self,
        operator: LinearOperator,
        method: str = "exact",
        solver_options: dict[str, any] | None = None,
    ) -> None:
        self.operator = operator
        self.method = method
        self.solver_options = solver_options if solver_options is not None else {}
        super().__init__(shape=operator.shape, dtype=operator.dtype)

    def _matmul(self, arr: jax.Array) -> jax.Array:
        if self.method == "exact":
            return lsolve(self.operator, arr)

        if self.method == "lsmr":
            from linox.linalg.approx.lsmr import lsmr_solve

            x, _ = lsmr_solve(self.operator, arr, **self.solver_options)
            return x

        if self.method in {"cg", "conjugate_gradient"}:
            x, _ = jax.scipy.sparse.linalg.cg(self.operator, arr, **self.solver_options)
            return x

        # Fallback
        return lsolve(self.operator, arr)

    def _todense(self) -> jax.Array:
        if self.method == "exact":
            config.warn(f"Linear operator {self.operator} is densed for inverse computation.")
            return jnp.linalg.inv(self.operator._todense())
        # If approx, we can't easily densify without solving against identity
        # Fallback to solving against Identity (expensive but correct)
        n = self.shape[0]
        I_op = jnp.eye(n, dtype=self.dtype)
        return self._matmul(I_op)

    def transpose(self) -> LinearOperator:
        """Return the transpose of this operator."""
        # If A is invertible, (A^-1)^T = (A^T)^-1
        # We propagate the solver method.
        # Note: if method was "cg" (expects SPD), A^T should also be SPD if A was.
        # If method was "lsmr", it works for A^T too.
        return InverseLinearOperator(
            self.operator.transpose(),
            method=self.method,
            solver_options=self.solver_options,
        )

    def tree_flatten(self) -> tuple[tuple[any, ...], dict[str, any]]:
        """Flatten this operator into JAX pytree children and static data."""
        children = (self.operator,)

        aux_data = {"method": self.method, "solver_options": self.solver_options} if self.method != "exact" else {}
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: dict[str, any],
        children: tuple[any, ...],
    ) -> "InverseLinearOperator":
        """Reconstruct this operator from JAX pytree children and static data."""
        (operator,) = children
        return cls(operator=operator, **aux_data)


@linverse.dispatch
def _(a: InverseLinearOperator) -> LinearOperator:
    return a.operator


@lsqrt.dispatch
def _(a: InverseLinearOperator) -> LinearOperator:
    # For PSD A: (A^{-1})^{1/2} = (A^{1/2})^{-1}
    return linverse(lsqrt(a.operator))


@ldet.dispatch
def ldet(a: InverseLinearOperator) -> jax.Array:
    """Compute the determinant of a linear operator."""
    if not is_square(a):
        msg = f"Operator {a} is not square."
        raise ValueError(msg)

    return 1 / ldet(a.operator)


class PseudoInverseLinearOperator(LinearOperator):
    """Moore-Penrose pseudo-inverse ``A^+`` of a linear operator."""

    def __init__(self, operator: LinearOperator, tol: float = 1e-12) -> None:
        self.operator = operator
        super().__init__(shape=operator.T.shape, dtype=operator.dtype)
        self.tol = tol

    def transpose(self) -> LinearOperator:
        """Return the transpose of this operator."""
        # (A^+)^T == (A^T)^+ -- transpose the operand, not the pseudo-inverse
        # of self, which recurses forever.
        return PseudoInverseLinearOperator(self.operator.transpose(), tol=self.tol)

    def _todense(self) -> jax.Array:
        r"""Materialize the pseudo-inverse densely.

        TODO: compute this from the SVD rather than via ``jnp.linalg.pinv``:
        ``U, S, Vh = svd(self.operator)``.

        Returns
        -------
            x_LS = \sum_i (u_i^T b) / s_i v_i
            -> U, S, Vh = svd(self.operator)
            return U @ jnp.diag(1 / S) @ Vh.
        """
        return jnp.linalg.pinv(self.operator._todense(), rtol=self.tol)

    def _matmul(self, arr: jax.Array) -> jax.Array:
        return lpsolve(self.operator, arr, rtol=self.tol)

    def tree_flatten(self) -> tuple[tuple[any, ...], dict[str, any]]:
        """Flatten this operator into JAX pytree children and static data."""
        children = (self.operator,)
        aux_data = {}
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: dict[str, any],
        children: tuple[any, ...],
    ) -> "PseudoInverseLinearOperator":
        """Reconstruct this operator from JAX pytree children and static data."""
        del aux_data
        (operator,) = children
        return cls(operator=operator)


@lpinverse.dispatch
def _(a: PseudoInverseLinearOperator) -> LinearOperator:
    return a.operator


@svd.dispatch
def _(
    a: PseudoInverseLinearOperator,
    **kwargs,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """SVD of pseudo-inverse operator: inverts singular values above tolerance.

    Notes
    -----
        Passes through all kwargs (k, full_matrices, compute_uv, etc.) to the
        underlying operator's SVD, then inverts singular values.
    """
    U, S, Vh = svd(a.operator, **kwargs)

    # Invert singular values (zero out those below tolerance)
    S_inv = jnp.where(a.tol < S, 1 / S, 0)

    # Create a mask for each singular value
    mask = (a.tol < S).astype(U.dtype)

    # Apply mask to relevant columns/rows
    # Note: With full_matrices=True, U can be (m, m) but S is (min(m,n),)
    # We only mask the first S.shape[0] columns of U and rows of Vh
    k = S.shape[0]
    if U.shape[1] == k:  # noqa: SIM108 - the per-branch comments carry the reasoning
        # Partial SVD or full_matrices=False: all columns correspond to singular values
        U_masked = U * mask[None, :]
    else:
        # full_matrices=True: only first k columns correspond to singular values
        U_masked = U.at[:, :k].multiply(mask[None, :])

    if Vh.shape[0] == k:  # noqa: SIM108 - as above
        # All rows correspond to singular values
        Vh_masked = Vh * mask[:, None]
    else:
        # Only first k rows correspond to singular values
        Vh_masked = Vh.at[:k, :].multiply(mask[:, None])

    return U_masked, S_inv, Vh_masked


# --------------------------------------------------------------------------- #
# Non-"l" prefixed functions (New API for linox 0.0.2+)
# Functions with "l" prefix are deprecated and will be removed in 0.0.3+
# --------------------------------------------------------------------------- #


@plum.dispatch
def add(a: LinearOperator, b: LinearOperator) -> LinearOperator:
    """Add two linear operators. See ladd for implementation details."""
    return ladd(a, b)


@add.dispatch
def _(a: LinearOperator, b: jax.Array) -> LinearOperator:
    return ladd(a, b)


@plum.dispatch
def sub(a: LinearOperator, b: LinearOperator) -> LinearOperator:
    """Subtract two linear operators. See lsub for implementation details."""
    return lsub(a, b)


@sub.dispatch
def _(a: LinearOperator, b: jax.Array) -> LinearOperator:
    return lsub(a, b)


@plum.dispatch
def mul(a: ScalarLike | jax.Array, b: LinearOperator) -> LinearOperator:
    """Multiply a linear operator by a scalar. See lmul for implementation details."""
    return lmul(a, b)


@plum.dispatch
def div(a: LinearOperator, b: LinearOperator) -> LinearOperator:
    """Divide linear operators. See ldiv for implementation details."""
    return ldiv(a, b)


@plum.dispatch
def matmul(a: LinearOperator, b: LinearOperator) -> ArithmeticType:
    """Matrix multiply linear operators. See lmatmul for implementation details."""
    return lmatmul(a, b)


@matmul.dispatch
def _(a: LinearOperator, b: jax.Array) -> jax.Array:
    return lmatmul(a, b)


@matmul.dispatch
def _(a: jax.Array, b: LinearOperator) -> LinearOperator:
    return lmatmul(a, b)


def neg(a: LinearOperator) -> LinearOperator:
    """Negate a linear operator. See lneg for implementation details."""
    return lneg(a)


@plum.dispatch
def sqrt(a: LinearOperator) -> LinearOperator:
    """Compute square root of a linear operator. See lsqrt for implementation details."""
    return lsqrt(a)


@plum.dispatch
def inverse(a: LinearOperator) -> ArithmeticType:
    """Compute inverse of a linear operator. See linverse for implementation details."""
    return linverse(a)


@plum.dispatch
def pinverse(a: LinearOperator) -> ArithmeticType:
    """Compute pseudo-inverse of a linear operator. See lpinverse for implementation details."""
    return lpinverse(a)


@plum.dispatch
def eigh(a: LinearOperator) -> tuple[jax.Array, LinearOperator]:
    """Compute eigendecomposition of a Hermitian operator. See leigh for implementation details."""
    return leigh(a)


@plum.dispatch
def qr(a: LinearOperator) -> tuple[jax.Array, jax.Array]:
    """QR decomposition of a linear operator. See lqr for implementation details."""
    return lqr(a)


@plum.dispatch
def solve(a: LinearOperator, b: jax.Array) -> jax.Array:
    """Solve the linear system Ax = b. See lsolve for implementation details."""
    return lsolve(a, b)


@plum.dispatch
def psolve(a: LinearOperator, b: jax.Array, rtol=1e-8) -> jax.Array:
    """Solve Ax = b using pseudo-inverse. See lpsolve for implementation details."""
    return lpsolve(a, b, rtol)


@plum.dispatch
def cholesky(a: LinearOperator) -> jax.Array:
    """Cholesky decomposition of a linear operator. See lcholesky for implementation details."""
    return lcholesky(a)


@plum.dispatch
def det(a: LinearOperator) -> jax.Array:
    """Compute determinant of a linear operator. See ldet for implementation details."""
    return ldet(a)


# Register all linear operators as PyTrees
jax.tree_util.register_pytree_node_class(ScaledLinearOperator)
jax.tree_util.register_pytree_node_class(AddLinearOperator)
jax.tree_util.register_pytree_node_class(ProductLinearOperator)
jax.tree_util.register_pytree_node_class(CongruenceTransform)
jax.tree_util.register_pytree_node_class(TransposedLinearOperator)
jax.tree_util.register_pytree_node_class(InverseLinearOperator)
jax.tree_util.register_pytree_node_class(PseudoInverseLinearOperator)
