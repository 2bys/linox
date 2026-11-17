"""Regression tests for bugs reported against linox 0.0.3 (found via luno).

Each test names the bug it guards against. The dtype tests use float32 inputs
on purpose: the suite as a whole runs under ``jax_enable_x64`` (several test
modules switch it on at import), so a result that silently falls back to
JAX's default dtype shows up as float64 here.
"""

import contextlib
import io
import subprocess
import sys

import jax
import jax.numpy as jnp
import linox
import plum
import pytest
from linox import config
from linox.operators import arithmetic
from linox.operators.arithmetic import CongruenceTransform, InverseLinearOperator, ScaledLinearOperator

F32 = jnp.float32


@pytest.fixture
def densify_events():
    """Collect the "converted to dense" debug events emitted during a test."""
    events = []
    config.set_debug_hook(events.append)
    try:
        yield events
    finally:
        config.set_debug_hook(None)


def _dense_msgs(events) -> list[str]:
    return [e.msg for e in events if "dense" in e.msg.lower()]


def _rand(seed: int, shape: tuple[int, ...], dtype=F32) -> jax.Array:
    return jax.random.normal(jax.random.key(seed), shape, dtype=dtype)


# --------------------------------------------------------------------------- #
# 1. Importing linox must not change JAX's global precision
# --------------------------------------------------------------------------- #


def test_import_does_not_enable_x64() -> None:
    # Needs a fresh interpreter: in this process x64 is already on and linox
    # is already imported.
    code = "import jax; before = jax.config.jax_enable_x64; import linox; print(before, jax.config.jax_enable_x64)"
    env = {k: v for k, v in __import__("os").environ.items() if k != "JAX_ENABLE_X64"}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)
    assert out.stdout.split() == ["False", "False"]


# --------------------------------------------------------------------------- #
# 2. CongruenceTransform defined once; its structured diagonal rule is used
# --------------------------------------------------------------------------- #


def test_congruence_transform_class_is_unique() -> None:
    import inspect

    source = inspect.getsource(arithmetic)
    assert source.count("class CongruenceTransform(") == 1
    precedence_5 = [m for m in arithmetic.diagonal.methods if m.signature.precedence == 5]
    assert [m.signature.types[0] for m in precedence_5] == [CongruenceTransform]


@pytest.mark.parametrize(
    "B",
    [
        pytest.param(lambda: linox.Matrix(_rand(1, (3, 3))), id="Matrix"),
        pytest.param(lambda: linox.Diagonal(jnp.arange(1.0, 4.0, dtype=F32)), id="Diagonal"),
        pytest.param(lambda: linox.Identity(3, dtype=F32), id="Identity"),
        pytest.param(
            lambda: linox.IsotropicScalingPlusSymmetricLowRank(F32(0.5), _rand(2, (3, 2)), jnp.array([1.0, 2.0], F32)),
            id="ISPLR",
        ),
        pytest.param(
            lambda: linox.PositiveDiagonalPlusSymmetricLowRank(
                linox.Diagonal(jnp.arange(1.0, 4.0, dtype=F32)),
                linox.SymmetricLowRank(_rand(3, (3, 2)), jnp.array([1.0, 2.0], F32)),
            ),
            id="PDPSLR",
        ),
        pytest.param(lambda: ScaledLinearOperator(linox.Matrix(_rand(4, (3, 3))), F32(2.0)), id="Scaled"),
    ],
)
def test_congruence_diagonal_is_exact_without_densifying(B, densify_events) -> None:
    # A Kronecker A has no dense-storage shortcut, so this goes through the
    # generic CongruenceTransform rule rather than a SymmetricLowRank rewrite.
    A = linox.Kronecker(linox.Matrix(_rand(5, (4, 1))), linox.Matrix(_rand(6, (5, 3))))
    B = B()
    C = linox.congruence_transform(A, B)
    densify_events.clear()

    d = linox.diagonal(C)

    assert not [m for m in _dense_msgs(densify_events) if "product" in m.lower()]
    expected = jnp.diag(A.todense() @ B.todense() @ A.todense().T)
    assert d.shape == (20,)
    assert d.dtype == F32
    assert jnp.allclose(d, expected, rtol=1e-4, atol=1e-4)


# --------------------------------------------------------------------------- #
# 3. Legacy names keep `.dispatch`; inv/linverse use the structured inverse
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ["linverse", "lsqrt", "lpinverse"])
def test_legacy_names_are_dispatchable(name) -> None:
    fn = getattr(linox, name)
    assert isinstance(fn, plum.Function)
    assert fn is getattr(arithmetic, name)


def _orth(seed: int, shape: tuple[int, int]) -> jax.Array:
    """Orthonormal columns: ISPLR's Woodbury inverse treats (U, S) as an eigendecomposition."""
    return jnp.linalg.qr(_rand(seed, shape))[0]


@pytest.fixture
def isplr():
    return linox.IsotropicScalingPlusSymmetricLowRank(F32(0.5), _orth(1, (6, 2)), jnp.array([1.0, 2.0], F32))


@pytest.mark.parametrize("fn", ["linverse", "inv", "inverse"])
def test_inverse_entry_points_use_woodbury(fn, isplr) -> None:
    inv = getattr(linox, fn)(isplr)
    assert isinstance(inv, linox.IsotropicScalingPlusSymmetricLowRank)
    assert inv.dtype == F32
    assert jnp.allclose(inv.todense() @ isplr.todense(), jnp.eye(6), atol=1e-4)


def test_user_rule_registered_via_linverse_reaches_inv() -> None:
    class MyOp(linox.Matrix):
        pass

    jax.tree_util.register_pytree_node_class(MyOp)
    sentinel = linox.Identity(2, dtype=F32)

    @linox.linverse.dispatch
    def _(a: MyOp) -> linox.LinearOperator:
        return sentinel

    op = MyOp(jnp.eye(2, dtype=F32))
    assert linox.inv(op) is sentinel
    assert linox.inverse(op) is sentinel


# --------------------------------------------------------------------------- #
# 4. inverse(method="auto") on large operators prefers exact structure
# --------------------------------------------------------------------------- #


@pytest.fixture
def small_dense_threshold():
    old = config.get_max_dense_n()
    config.set_max_dense_n(4)  # stand in for n > 2000 without the cost
    try:
        yield
    finally:
        config.set_max_dense_n(old)


def test_auto_inverse_large_structured_is_exact(isplr, small_dense_threshold) -> None:
    assert isinstance(linox.inverse(isplr), linox.IsotropicScalingPlusSymmetricLowRank)
    assert type(linox.inverse(isplr)) is type(linox.sqrt(isplr))


@pytest.mark.parametrize(
    "op",
    [
        pytest.param(lambda: linox.Matrix(_rand(0, (6, 6))), id="Matrix"),
        pytest.param(lambda: ScaledLinearOperator(linox.Matrix(_rand(0, (6, 6))), F32(2.0)), id="Scaled-Matrix"),
    ],
)
def test_auto_inverse_large_dense_still_approximates(op, small_dense_threshold) -> None:
    # `linverse(Matrix)` is a full dense inverse; auto must not pick it for large n.
    r = linox.inverse(op())
    assert isinstance(r, InverseLinearOperator)
    assert r.method == "lsmr"


def _spd(seed: int, n: int) -> jax.Array:
    X = _rand(seed, (n, n))
    return X @ X.T / n + jnp.eye(n, dtype=F32)


@pytest.mark.parametrize(
    "op",
    [
        pytest.param(lambda: linox.Matrix(_spd(0, 30)), id="Matrix"),
        pytest.param(lambda: ScaledLinearOperator(linox.Matrix(_spd(0, 30)), F32(2.0)), id="Scaled-Matrix"),
    ],
)
def test_auto_sqrt_large_dense_uses_lanczos(op, small_dense_threshold) -> None:
    # `lsqrt(Matrix)` is a full dense Cholesky; auto must not pick it for large n.
    from linox.operators.functional import MatrixFunctionLinearOperator

    op = op()
    S = linox.sqrt(op)
    assert isinstance(S, MatrixFunctionLinearOperator)
    assert S.dtype == F32
    v = _rand(1, (30,))
    # Lanczos gives the principal (symmetric) root, so S @ S == op.
    assert jnp.allclose(S @ (S @ v), op @ v, rtol=1e-3, atol=1e-3)


def test_auto_sqrt_large_structured_stays_exact(isplr, small_dense_threshold) -> None:
    assert isinstance(linox.sqrt(isplr), linox.IsotropicScalingPlusSymmetricLowRank)
    assert isinstance(linox.sqrt(linox.Diagonal(jnp.arange(1.0, 7.0, dtype=F32))), linox.Diagonal)


def test_explicit_exact_sqrt_of_large_matrix_is_cholesky(small_dense_threshold) -> None:
    A = _spd(0, 30)
    L = linox.sqrt(linox.Matrix(A), method="exact")
    assert not isinstance(L, linox.operators.functional.MatrixFunctionLinearOperator)
    assert jnp.allclose(L.todense() @ L.todense().T, A, rtol=1e-4, atol=1e-4)


def test_explicit_approx_inverse_is_honoured(isplr) -> None:
    r = linox.inverse(isplr, method="lsmr")
    assert isinstance(r, InverseLinearOperator)
    assert r.method == "lsmr"


# --------------------------------------------------------------------------- #
# 5. config.warn prints when debug is on, and only then
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("debug", [True, False])
def test_warn_prints_only_in_debug(debug) -> None:
    old = config.is_debug()
    config.set_debug(debug)
    buf = io.StringIO()
    try:
        with contextlib.redirect_stderr(buf), contextlib.redirect_stdout(buf):
            config.warn("hello")
    finally:
        config.set_debug(old)
    assert buf.getvalue() == ("Warning: hello\n" if debug else "")


# --------------------------------------------------------------------------- #
# 6. Public typing module
# --------------------------------------------------------------------------- #


def test_public_typing_module() -> None:
    import typing

    from linox import _types
    from linox import typing as ltyping

    for name in ltyping.__all__:
        assert getattr(ltyping, name) is getattr(_types, name)
    # The forward reference must resolve to a class that actually exists.
    (ref,) = [a for a in typing.get_args(ltyping.MatrixType) if isinstance(a, typing.ForwardRef)]
    assert eval(ref.__forward_arg__, {"linox": linox}) is linox.LinearOperator


# --------------------------------------------------------------------------- #
# 7. Structured diagonal / congruence rules (formerly provided by luno)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "op",
    [
        pytest.param(lambda: linox.SymmetricLowRank(_rand(1, (7, 2)), jnp.array([1.0, 3.0], F32)), id="SymmetricLowRank"),
        pytest.param(lambda: linox.LowRank(_rand(1, (7, 2)), jnp.array([1.0, 3.0], F32), _rand(2, (7, 2))), id="LowRank"),
        pytest.param(lambda: linox.SymmetricLowRank(_rand(1, (7, 2))), id="default-S"),
    ],
)
def test_lowrank_diagonal_without_densifying(op, densify_events) -> None:
    op = op()
    d = linox.diagonal(op)
    assert _dense_msgs(densify_events) == []
    assert d.dtype == F32
    assert jnp.allclose(d, jnp.diag(op.todense()), rtol=1e-5, atol=1e-5)


def test_lowrank_default_S_follows_U_dtype() -> None:
    op = linox.SymmetricLowRank(_rand(1, (5, 2)))
    assert op.S.dtype == F32
    assert op.dtype == F32


def test_lowrank_dtype_promotes_with_S() -> None:
    op = linox.SymmetricLowRank(_rand(1, (5, 2)), jnp.ones(2, dtype=jnp.float64))
    assert op.dtype == jnp.result_type(F32, jnp.float64)


A_DENSE = lambda: linox.Matrix(_rand(10, (8, 3)))  # noqa: E731


@pytest.mark.parametrize(
    ("A", "B", "expected_type"),
    [
        pytest.param(A_DENSE, lambda: linox.SymmetricLowRank(_rand(11, (3, 2)), jnp.array([1.0, 2.0], F32)),
                     linox.SymmetricLowRank, id="SymmetricLowRank"),
        pytest.param(A_DENSE, lambda: linox.Identity(3, dtype=F32), linox.SymmetricLowRank, id="Matrix-Identity"),
        pytest.param(A_DENSE, lambda: linox.Diagonal(jnp.array([1.0, 2.0, 3.0], F32)), linox.SymmetricLowRank,
                     id="Matrix-Diagonal"),
        pytest.param(A_DENSE, lambda: ScaledLinearOperator(linox.Matrix(_rand(14, (3, 3))), F32(3.0)),
                     ScaledLinearOperator, id="Scaled"),
        pytest.param(A_DENSE, lambda: linox.IsotropicScalingPlusSymmetricLowRank(
            F32(0.5), _rand(12, (3, 2)), jnp.array([1.0, 2.0], F32)), CongruenceTransform, id="ISPLR"),
        pytest.param(A_DENSE, lambda: linox.PositiveDiagonalPlusSymmetricLowRank(
            linox.Diagonal(jnp.array([1.0, 2.0, 3.0], F32)),
            linox.SymmetricLowRank(_rand(13, (3, 2)), jnp.array([1.0, 2.0], F32))),
            CongruenceTransform, id="PDPSLR"),
    ],
)
def test_congruence_rules(A, B, expected_type, densify_events) -> None:
    A, B = A(), B()
    C = linox.congruence_transform(A, B)
    expected = A.todense() @ B.todense() @ A.todense().T

    assert isinstance(C, expected_type)
    assert C.shape == (8, 8)
    assert C.dtype == F32
    densify_events.clear()
    d = linox.diagonal(C)
    assert _dense_msgs(densify_events) == []
    assert d.dtype == F32
    assert jnp.allclose(d, jnp.diag(expected), rtol=1e-4, atol=1e-4)
    assert jnp.allclose(C.todense(), expected, rtol=1e-4, atol=1e-4)


def test_matrix_identity_congruence_ignores_identity_default_dtype() -> None:
    # An Identity built without a dtype follows JAX's default (float64 here);
    # it carries no data, so it must not upcast the user's float32 A.
    C = linox.congruence_transform(A_DENSE(), linox.Identity(3))
    assert C.dtype == F32


@pytest.mark.parametrize(
    ("scalar", "expected"),
    [
        pytest.param(2.0, F32, id="python-float"),
        pytest.param(jnp.asarray(2.0, F32), F32, id="float32"),
        pytest.param(jnp.asarray(2.0, jnp.float64), jnp.result_type(F32, jnp.float64), id="float64-array"),
    ],
)
def test_scaled_operator_keeps_operator_dtype(scalar, expected) -> None:
    # A Python scalar is weakly typed and must not upcast the user's float32
    # operator (it did under x64); an explicitly typed array still promotes.
    op = ScaledLinearOperator(linox.Matrix(_rand(0, (3, 3))), scalar)
    assert op.dtype == expected
    assert (op @ jnp.ones(3, F32)).dtype == expected
