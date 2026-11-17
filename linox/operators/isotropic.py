"""Isotropic additive operators of the form ``s*I + A``."""

# _isotropicadd.py

import jax
import jax.numpy as jnp

from linox import utils
from linox.operators.arithmetic import (
    AddLinearOperator,
    ScaledLinearOperator,
    diagonal,
    lcholesky,
    leigh,
    lexp,
    linverse,
    llog,
    lpinverse,
    lpow,
    lsolve,
    lsqrt,
    ltrace,
    slogdet,
)
from linox.operators.base import LinearOperator
from linox.operators.diagonal import Diagonal
from linox.operators.special import Identity
from linox.utils.checks import require


def _require_symmetric(A: LinearOperator, *, num_probes: int = 3) -> None:
    """Raise unless ``A`` is symmetric.

    Every spectral shortcut on :class:`IsotropicAdditiveLinearOperator` goes
    through ``jnp.linalg.eigh``, which reads only the lower triangle. Handing
    it a non-symmetric operand returns a wrong answer *silently* --
    ``todense()`` and ``solve()`` end up disagreeing about the same operator.

    The check applies under ``jax.jit`` too, where it becomes a runtime error
    rather than a trace-time one, and it never materialises the operand:

    * an operator that declares ``is_symmetric`` is taken at its word;
    * one that already holds a concrete dense array is compared exactly,
      which costs nothing since the array is in hand;
    * anything else -- Kronecker, kernel, block, any lazy composition -- is
      probed matrix-free through the bilinear form ``<x, Ay> == <Ax, y>``,
      which needs only *forward* matvecs. Densifying here would defeat the
      point: ``leigh`` has structured dispatches that never build the dense
      matrix, so a dense check would be the only O(n^2) step in the whole
      path. Note this deliberately avoids ``A.T``, whose base implementation
      materialises the dense matrix for any operator lacking a structured
      transpose.

    The probe is randomized, so an adversarially-constructed operand could
    slip through; for that, ``num_probes`` can be raised.
    """
    if getattr(A, "is_symmetric", False):
        return

    array = getattr(A, "A", None)
    if isinstance(array, jnp.ndarray) and array.ndim >= 2:
        ok = jnp.allclose(array, jnp.swapaxes(array, -1, -2))
    else:
        # Probe the bilinear form: A is self-adjoint iff <x, Ay> == <Ax, y>.
        #
        # Deliberately avoids `A.T`. For an operator that does not override
        # `transpose()`, the base implementation materialises the dense matrix,
        # so an `A.T @ x` formulation would densify precisely the operators
        # this check exists to keep matrix-free. The bilinear form needs only
        # forward matvecs and so never materialises anything.
        key = jax.random.PRNGKey(0)
        n = A.shape[-1]
        ok = jnp.asarray(True)
        for i in range(num_probes):
            kx, ky = jax.random.split(jax.random.fold_in(key, i))
            x = jax.random.normal(kx, (n,), dtype=A.dtype)
            y = jax.random.normal(ky, (n,), dtype=A.dtype)
            lhs = jnp.vdot(x, A @ y)
            rhs = jnp.vdot(A @ x, y)
            ok = ok & jnp.allclose(lhs, rhs, rtol=1e-5, atol=1e-8)

    require(
        ok,
        "IsotropicAdditiveLinearOperator (s*I + A) requires a symmetric A: its "
        "spectral shortcuts use eigh, which reads only the lower triangle and "
        "would silently return wrong inverses and solves. Symmetrise A first "
        "(e.g. linox.symmetrize(A)), or build the sum explicitly with "
        "AddLinearOperator to keep the general path.",
    )


def _is_integer_power(power: float) -> bool:
    """Whether ``power`` is a concrete integer, for which a negative base is fine."""
    try:
        return float(power).is_integer()
    except (TypeError, ValueError, jax.errors.ConcretizationTypeError):
        return False


def _require_positive_spectrum(a: "IsotropicAdditiveLinearOperator", op: str, *, strict: bool) -> None:
    """Raise unless the eigenvalues of ``s*I + A`` are non-negative.

    Symmetry alone makes the eigendecomposition valid, which is all that
    ``inverse``, ``eigh`` and ``slogdet`` need. Taking a square root, a
    Cholesky-like factor, a logarithm or a fractional power additionally
    requires the *shifted* spectrum ``s + lambda`` to be non-negative
    (positive, for the logarithm) -- otherwise the result is silently NaN.

    Free to check: the eigenvalues have already been computed by the time any
    of these dispatches run.
    """
    S = a.S
    eigs = diagonal(S) if isinstance(S, LinearOperator) else S
    shifted = eigs + a.scalar
    ok = jnp.all(shifted > 0) if strict else jnp.all(shifted >= 0)
    require(
        ok,
        f"{op} of (s*I + A) requires the shifted spectrum s + lambda to be "
        f"{'positive' if strict else 'non-negative'}, but s*I + A is "
        f"{'not positive definite' if strict else 'indefinite'}. Symmetry alone "
        "is enough for inverse/eigh/slogdet, which stay available; this "
        "operation is not defined on a spectrum that crosses zero. Increase "
        "the shift s so it lifts the whole spectrum, or use a method that "
        "does not require positivity.",
    )


class IsotropicAdditiveLinearOperator(AddLinearOperator):
    r"""Isotropic additive linear operator for matrices of the form.

        A_iso := s I + A,

    where ``s`` is a scalar (or a 0-arg scalar LinearOperator) and ``A`` is a
    symmetric LinearOperator. This class exposes fast, matrix-free implementations
    of common spectral transforms (inverse, pseudo-inverse, square root, log,
    powers, exp, Cholesky-like factor) by working in the eigenbasis of ``A``.

    ----------
    Core idea
    ----------
    If ``A = Q Λ Qᵀ`` is an eigendecomposition of ``A`` (with Λ diagonal and
    ``Qᵀ Q = I``), then

        s I + A = Q (Λ + s I) Qᵀ,

    so any spectral function ``f`` (e.g. inverse, sqrt, log, power, exp) satisfies

        f(s I + A) = Q f(Λ + s I) Qᵀ,

    which reduces the linear-algebra to elementwise operations on the eigenvalues.

    This class computes/caches an (optionally truncated) eigendecomposition via
    ``leigh(A)`` and then dispatches the following:

    * ``linverse``:       (s I + A)⁻¹ = (1/s) I − Q diag(λ / (s (λ + s))) Qᵀ
                          (Woodbury / projector–complement split)
    * ``lpinverse``:      pseudo-inverse using the same spectral formula with
                          safe handling of zero/near-zero modes.
    * ``lsqrt``:          (s I + A)^{1/2} = Q diag(√(λ + s)) Qᵀ
    * ``lcholesky``:      returns a factor L with L Lᵀ = s I + A, namely
                          L = Q diag(√(λ + s))   (orthonormal “spectral” factor)
    * ``llog``:           log(s I + A) = Q diag(log(λ + s)) Qᵀ
    * ``lpow``:           (s I + A)^p = Q diag((λ + s)^p) Qᵀ
    * ``diagonal``:       diag(s I + A) = s · 1 + diag(A)
    * ``ltrace``:         tr(s I + A) = s·n + tr(A)  (with Hutchinson if needed)
    * ``lexp``:           exp(s I + A) = Q diag(exp(λ + s)) Qᵀ

    -------------------------------
    Projector / anti-projector view
    -------------------------------
    When ``leigh`` returns a **truncated** eigenspace ``Q ∈ ℝ^{n×k}`` (k ≤ n),
    let P := Q Qᵀ be the projector onto the retained subspace and
    P⊥ := I − P the orthogonal complement. Then

        (s I + A)⁻¹
        = Q (Λ + s I)⁻¹ Qᵀ  +  (1/s) P⊥,

    i.e. the inverse acts as ``(Λ + s I)⁻¹`` on span(Q) and as ``(1/s) I`` on
    its orthogonal complement. The implementation of ``linverse`` uses the
    equivalent Woodbury form

        (s I + A)⁻¹ = (1/s) [ I − Q diag(λ / (λ + s)) Qᵀ ].

    If ``leigh`` is **full-rank**, then P = I and P⊥ = 0, which recovers the
    usual full spectral formulas.

    -------------
    Caching notes
    -------------
    * ``Q`` and ``S`` (eigenvectors/eigenvalues) are cached lazily by
      ``_ensure_eigh()``. Any operation that changes the operator should call
      ``_invalidate_cache()``.
    * ``projector`` (Q Qᵀ) and ``complement`` (I − Q Qᵀ) are also cached on demand.

    ----------

    Arguments:
    ----------
    s : jax.Array
        Scalar added to the diagonal (isotropic shift). May be wrapped into a
        scalar ``ScaledLinearOperator(Identity, s)``.
    A : LinearOperator
        Symmetric linear operator (square).

        Symmetry is **required and checked exactly**, including under
        ``jax.jit`` -- where it becomes a runtime error rather than a
        trace-time one. The check costs nothing asymptotically: it runs in
        :meth:`_ensure_eigh`, which is about to densify ``A`` for ``leigh``
        regardless.

        ``_matmul`` and ``_todense`` stay permissive, since ``s*I + A`` is
        computed correctly for any square ``A`` and ``smart_add`` routes every
        ``Identity + op`` sum through this class. Only the eigh-backed
        shortcuts refuse.

        Symmetry alone is enough for ``inverse``, ``eigh`` and ``slogdet``.
        ``sqrt``, ``lcholesky``, ``llog`` and fractional ``lpow`` additionally
        require the *shifted* spectrum ``s + lambda`` to be non-negative
        (strictly positive for the logarithm); that is checked per-operation
        against the eigenvalues, which have already been computed by then.

    -------

    Returns
    -------
    A LinearOperator supporting matrix-free application and spectral transforms
    of ``s I + A`` via the multipledispatch functions listed above.

    -------

    Example:
    -------
    >>> n = 100
    >>> s = jnp.array(0.1)
    >>> A = utils.as_linop(jnp.diag(jnp.linspace(0.0, 5.0, n)))  # symmetric
    >>> L = IsotropicAdditiveLinearOperator(s, A)
    >>> x = jnp.ones((n,))
    >>> y = (linverse(L) @ x)          # apply (s I + A)^{-1} to a vector
    >>> d = diagonal(L)                 # exact diagonal
    >>> z = (lsqrt(L) @ x)              # apply (s I + A)^{1/2} to a vector

    """

    def __init__(self, s: jax.Array, A: LinearOperator) -> None:
        self._A = utils.as_linop(A)
        if self._A.shape[-1] != self._A.shape[-2]:
            msg = "A must be a square matrix."
            raise ValueError(msg)
        self._s = ScaledLinearOperator(Identity(self._A.shape[0], dtype=self._A.dtype), s)
        self._Q = None
        self._S = None
        self._projector = None
        self._complement = None
        super().__init__(self._s, self._A)

    def _ensure_eigh(self) -> None:
        if (self._S is None) or (self._Q is None):
            # Guard here rather than in __init__: `_matmul`/`_todense` compute
            # s*I + A correctly for any square A, and `smart_add` rewrites
            # every `Identity + op` sum into this class. Only the eigh-based
            # shortcuts require symmetry, so only they need to refuse.
            _require_symmetric(self._A)
            self._S, self._Q = leigh(self._A)
            # invalidate derived caches
            self._projector = None
            self._complement = None

    def _invalidate_cache(self) -> None:
        self._Q = self._S = self._projector = self._complement = None

    @property
    def s(self) -> jax.Array:
        """Scalar operator component (s * I)."""
        return self._s

    @property
    def scalar(self) -> jax.Array:
        """Scalar value s from the isotropic shift."""
        return self._s.scalar

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the operator."""
        return self._A.shape

    @property
    def operator(self) -> LinearOperator:
        """The base linear operator A."""
        return self._A

    @property
    def Q(self) -> LinearOperator:
        """Eigenvectors of A (computed lazily via leigh)."""
        self._ensure_eigh()
        return self._Q

    @property
    def S(self) -> LinearOperator:
        """Eigenvalues of A (computed lazily via leigh)."""
        self._ensure_eigh()
        return self._S

    @property
    def projector(self) -> LinearOperator:
        """Projector onto the eigenspace Q Q^T (cached)."""
        self._ensure_eigh()
        if self._projector is None:
            self._projector = self._Q @ self._Q.T
        return self._projector

    @property
    def complement(self) -> LinearOperator:
        """Orthogonal complement projector I - Q Q^T (cached)."""
        self._ensure_eigh()
        if self._complement is None:
            self._complement = Identity(self.shape[0], dtype=self._A.dtype) - self.projector
        return self._complement

    def _matmul(self, arr: jax.Array):
        return self._s @ arr + self._A @ arr

    def _todense(self) -> jax.Array:
        return self._s._todense() + self._A._todense()

    def tree_flatten(self) -> tuple[tuple, dict]:
        """Flatten this operator into JAX pytree children and static data."""
        children = (self._s.scalar, self._A)
        aux_data = {}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Reconstruct this operator from JAX pytree children and static data."""
        s, A = children
        return cls(s, A)


jax.tree_util.register_pytree_node_class(IsotropicAdditiveLinearOperator)


@lcholesky.dispatch
def _(a: IsotropicAdditiveLinearOperator) -> LinearOperator:
    a._ensure_eigh()
    Q, S = a.Q, a.S  # cached
    s = a.s.scalar
    # Cholesky of A + sI = Q * sqrt(Λ + sI) where A = Q Λ Q^T
    _require_positive_spectrum(a, "Cholesky factor", strict=False)

    eigs = diagonal(S) if isinstance(S, LinearOperator) else S

    new_lam = Diagonal(jnp.sqrt(eigs + s))
    return Q @ new_lam


@lsqrt.dispatch(precedence=1)
def _(a: IsotropicAdditiveLinearOperator) -> LinearOperator:
    a._ensure_eigh()
    _require_positive_spectrum(a, "Square root", strict=False)
    Q, S = a.Q, a.S  # cached
    s = a.s.scalar

    eigs = diagonal(S) if isinstance(S, LinearOperator) else S

    new_lam = Diagonal(jnp.sqrt(eigs + s))
    return Q @ new_lam @ Q.T


# we need a log-determinant function here


@lsolve.dispatch
def _(a: IsotropicAdditiveLinearOperator, b: jax.Array) -> jax.Array:
    r"""Solve (sI + A)x = b using eigendecomposition.

    For A = Q Λ Q^T, we have (sI + A) = Q(sI + Λ)Q^T, so:
        x = Q diag(1/(s + λ)) Q^T b
    """
    # Optimization for LowRank operators
    from linox.linalg.woodbury import woodbury_solve
    from linox.operators.lowrank import SymmetricLowRank

    if isinstance(a.operator, SymmetricLowRank):
        # (sI + U S U^T) x = b
        # woodbury_solve(U, S, s, b)
        return woodbury_solve(a.operator.U, a.operator.S, a.s.scalar, b)

    a._ensure_eigh()
    Q, S = a.Q, a.S  # cached
    s = a.s.scalar

    eigs = diagonal(S) if isinstance(S, LinearOperator) else S

    # Q^T b
    Qt_b = Q.T @ b

    # diag(1/(s + λ)) @ Q^T b
    inv_eigs = 1.0 / (s + eigs)
    scaled = inv_eigs * Qt_b if Qt_b.ndim == 1 else inv_eigs[:, None] * Qt_b

    # Q @ scaled
    return Q @ scaled


@linverse.dispatch
def _(a: IsotropicAdditiveLinearOperator) -> LinearOperator:
    a._ensure_eigh()
    Q, S = a.Q, a.S  # cached
    s = a.s.scalar

    inv_iso = linverse(a.s)

    eigs = diagonal(S) if isinstance(S, LinearOperator) else S
    D = Diagonal(eigs / (s * (eigs + s)))

    return inv_iso - (Q @ D @ Q.T)

    inv_iso = lpinverse(a.s)

    eigs = diagonal(S) if isinstance(S, LinearOperator) else S
    D = Diagonal(eigs / (s * (eigs + s)))

    return inv_iso - (Q @ D @ Q.T)


@slogdet.dispatch
def _(a: IsotropicAdditiveLinearOperator) -> tuple[jax.Array, jax.Array]:
    """Compute sign and logdet of sI + A using eigenvalues."""
    a._ensure_eigh()

    _Q, S = a.Q, a.S
    s = a.s.scalar

    eigs = diagonal(S) if isinstance(S, LinearOperator) else S

    vals = eigs + s

    # sign = product of signs
    sign = jnp.prod(jnp.sign(vals))
    logdet = jnp.sum(jnp.log(jnp.abs(vals)))

    return sign, logdet


@leigh.dispatch
def _(a: IsotropicAdditiveLinearOperator) -> tuple[LinearOperator, LinearOperator]:
    a._ensure_eigh()
    Q, S = a.Q, a.S  # cached
    s = a.s.scalar
    if isinstance(S, LinearOperator):
        n = S.shape[0]
        new_lam = S + s * Identity(n, dtype=S.dtype)
    else:
        new_lam = utils.as_linop(S + s)
    return new_lam, Q


@diagonal.dispatch
def _(a: IsotropicAdditiveLinearOperator) -> jax.Array:
    # Sum of diagonals: diag(A) + s * 1
    return jnp.asarray(diagonal(a.operator)) + jnp.asarray(diagonal(a.s))


# New matrix-free function dispatches for IsotropicAdditive
@ltrace.dispatch
def _(
    a: IsotropicAdditiveLinearOperator,
    key: jax.Array | None = None,
    num_samples: int = 100,
    distribution: str = "rademacher",
) -> tuple[jax.Array, jax.Array]:
    """Trace of sI + A: trace(sI + A) = s*n + trace(A)."""
    n = a.shape[-1]
    s = a.s.scalar

    # Recursively compute trace of A
    trace_A, std_A = ltrace(a.operator, key=key, num_samples=num_samples, distribution=distribution)

    trace_value = s * n + trace_A
    trace_std = std_A  # std of constant + random variable = std of random variable

    return trace_value, trace_std


@lexp.dispatch
def _(
    a: IsotropicAdditiveLinearOperator,
    v: jax.Array | None = None,
    num_iters: int = 20,
    method: str = "lanczos",
) -> jax.Array | LinearOperator:
    """Matrix exponential of sI + A using eigendecomposition.

    exp(sI + A) = exp(s) * exp(A) since sI and A commute... NO, this is wrong!
    Actually: exp(sI + A) = U exp(s + λ) U^T where A = U λ U^T
    """
    a._ensure_eigh()
    s = a.s.scalar

    # Eigenvalues of sI + A are s + λ(A)
    S = a.S
    eigs = diagonal(S) if isinstance(S, LinearOperator) else S

    eigvals = eigs + s

    if v is None:
        # Return lazy operator: U @ Diagonal(exp(s + λ)) @ U^T
        exp_eigvals = Diagonal(jnp.exp(eigvals))
        from linox.operators.arithmetic import congruence_transform

        return congruence_transform(a.Q, exp_eigvals)
    # exp(sI + A) @ v = U @ exp(s + λ) @ U^T @ v
    return a.Q @ (jnp.exp(eigvals) * (a.Q.T @ v))


@llog.dispatch
def _(
    a: IsotropicAdditiveLinearOperator,
    v: jax.Array | None = None,
    num_iters: int = 20,
    method: str = "lanczos",
) -> jax.Array | LinearOperator:
    """Matrix logarithm of sI + A using eigendecomposition.

    log(sI + A) = U log(s + λ) U^T where A = U λ U^T
    """
    a._ensure_eigh()
    _require_positive_spectrum(a, "Logarithm", strict=True)
    s = a.s.scalar

    # Eigenvalues of sI + A are s + λ(A)
    S = a.S
    eigs = diagonal(S) if isinstance(S, LinearOperator) else S

    eigvals = eigs + s

    if v is None:
        # Return lazy operator: U @ Diagonal(log(s + λ)) @ U^T
        log_eigvals = Diagonal(jnp.log(eigvals))
        from linox.operators.arithmetic import congruence_transform

        return congruence_transform(a.Q, log_eigvals)
    # log(sI + A) @ v = U @ log(s + λ) @ U^T @ v
    return a.Q @ (jnp.log(eigvals) * (a.Q.T @ v))


@lpow.dispatch
def _(
    a: IsotropicAdditiveLinearOperator,
    *,
    power: float,
    v: jax.Array | None = None,
    num_iters: int = 20,
    method: str = "lanczos",
) -> jax.Array | LinearOperator:
    """Matrix power of sI + A using eigendecomposition.

    (sI + A)^p = U (s + λ)^p U^T where A = U λ U^T
    """
    a._ensure_eigh()
    if not _is_integer_power(power):
        _require_positive_spectrum(a, f"Power {power}", strict=False)
    s = a.s.scalar

    # Eigenvalues of sI + A are s + λ(A)
    S = a.S
    eigs = diagonal(S) if isinstance(S, LinearOperator) else S

    eigvals = eigs + s

    if v is None:
        # Return lazy operator: U @ Diagonal((s + λ)^p) @ U^T
        pow_eigvals = Diagonal(eigvals**power)
        from linox.operators.arithmetic import congruence_transform

        return congruence_transform(a.Q, pow_eigvals)
    # (sI + A)^p @ v = U @ (s + λ)^p @ U^T @ v
    return a.Q @ ((eigvals**power) * (a.Q.T @ v))
