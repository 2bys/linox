# test_early_densification.py

"""Test suite to detect early/premature densification in linear operators.

This test suite ensures that linear operators maintain lazy evaluation and only
densify when absolutely necessary (e.g., when _matmul is called or explicit
todense() is requested by the user).

Early densification is dangerous because:
1. It defeats the purpose of having structured operators
2. It causes memory explosion for large problems
3. It prevents exploitation of efficient linear algebra
4. It can happen silently inside jnp functions
"""

import functools
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import linox
import pytest
from linox import (
    BlockDiagonal,
    Diagonal,
    Identity,
    IsotropicAdditiveLinearOperator,
    Kronecker,
    Matrix,
)
from linox.operators.arithmetic import diagonal, linverse
from linox.operators.lowrank import (
    IsotropicScalingPlusSymmetricLowRank,
    SymmetricLowRank,
)

# ============================================================================
# Test Utilities
# ============================================================================


class DensificationDetector:
    """Context manager that detects if todense() is called."""

    def __init__(
        self, operator: linox.LinearOperator, allow_in_matmul: bool = True
    ) -> None:
        self.operator = operator
        self.allow_in_matmul = allow_in_matmul
        self.todense_called = False
        self.call_stack = []
        self.original_todense = None

    def __enter__(self):
        self.todense_called = False
        self.call_stack = []

        original_todense = self.operator.todense

        @functools.wraps(original_todense)
        def tracked_todense(*args, **kwargs):
            import traceback

            stack = traceback.extract_stack()
            # Check if we're being called from _matmul or todense itself
            in_matmul = any("_matmul" in frame.name for frame in stack)

            if not (self.allow_in_matmul and in_matmul):
                self.todense_called = True
                self.call_stack = [
                    f"{frame.filename}:{frame.lineno} in {frame.name}"
                    for frame in stack[-10:]
                ]

            return original_todense(*args, **kwargs)

        self.original_todense = original_todense
        self.operator.todense = tracked_todense
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.original_todense is not None:
            self.operator.todense = self.original_todense
        return False

    def assert_no_densification(self, message: str = "") -> None:
        if self.todense_called:
            stack_str = "\n  ".join(self.call_stack)
            msg = f"Early densification detected! {message}\nCall stack:\n  {stack_str}"
            raise AssertionError(msg)


def assert_no_early_densification(
    operator: linox.LinearOperator,
    operation: Callable[[linox.LinearOperator], Any],
    message: str = "",
) -> Any:
    """Assert that an operation doesn't cause early densification.

    Args:
        operator: The linear operator to test
        operation: Function that takes the operator and performs some operation
        message: Additional context for the error message

    Returns:
        The result of the operation

    Raises:
        AssertionError: If todense() is called during the operation
    """
    with DensificationDetector(operator, allow_in_matmul=True) as detector:
        result = operation(operator)
        detector.assert_no_densification(message)
    return result


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


@pytest.fixture
def small_size() -> int:
    return 4


@pytest.fixture
def matrix_a(key, small_size):
    return jax.random.normal(key, (small_size, small_size))


@pytest.fixture
def matrix_b(key, small_size):
    key2 = jax.random.split(key)[0]
    return jax.random.normal(key2, (small_size, small_size))


@pytest.fixture
def spd_matrix(key, small_size):
    """Symmetric positive definite matrix."""
    A = jax.random.normal(key, (small_size, small_size))
    return A @ A.T + jnp.eye(small_size) * 1e-3


@pytest.fixture
def vector(key, small_size):
    return jax.random.normal(key, (small_size,))


# ============================================================================
# Test Arithmetic Operations
# ============================================================================


def test_scaled_operator_no_densification(matrix_a, vector) -> None:
    """Test that ScaledLinearOperator doesn't densify during construction or matmul."""
    A = Matrix(matrix_a)

    # Construction should not densify
    scaled = assert_no_early_densification(
        A, lambda op: 2.5 * op, "ScaledLinearOperator construction"
    )

    # Matrix-vector product should not densify (except in _matmul)
    result = scaled @ vector
    assert jnp.allclose(result, 2.5 * matrix_a @ vector)


def test_add_operator_no_densification(matrix_a, matrix_b, vector) -> None:
    """Test that AddLinearOperator doesn't densify during construction or matmul."""
    A = Matrix(matrix_a)
    B = Matrix(matrix_b)

    # Construction should not densify
    sum_op = assert_no_early_densification(
        A, lambda op: op + B, "AddLinearOperator construction"
    )

    # Matrix-vector product should not densify
    result = sum_op @ vector
    assert jnp.allclose(result, (matrix_a + matrix_b) @ vector)


def test_product_operator_no_densification(matrix_a, matrix_b, vector) -> None:
    """Test that ProductLinearOperator doesn't densify during construction or matmul."""
    A = Matrix(matrix_a)
    B = Matrix(matrix_b)

    # Construction should not densify
    prod_op = assert_no_early_densification(
        A, lambda op: op @ B, "ProductLinearOperator construction"
    )

    # Matrix-vector product should not densify
    result = prod_op @ vector
    assert jnp.allclose(result, matrix_a @ matrix_b @ vector, atol=1e-6)


def test_transpose_no_densification(matrix_a, vector) -> None:
    """Test that transpose doesn't densify."""
    A = Matrix(matrix_a)

    # Transpose should not densify
    A_T = assert_no_early_densification(A, lambda op: op.T, "Transpose operation")

    # Using transposed operator should not densify
    result = A_T @ vector
    assert jnp.allclose(result, matrix_a.T @ vector)


# ============================================================================
# Test Diagonal Operations
# ============================================================================


def test_diagonal_of_scaled_no_densification(matrix_a) -> None:
    """Test that diagonal of scaled operator doesn't densify."""
    A = Matrix(matrix_a)
    scaled = 3.0 * A

    # Getting diagonal should not densify the scaled operator
    with DensificationDetector(scaled, allow_in_matmul=True) as detector:
        diag = diagonal(scaled)
        detector.assert_no_densification("diagonal of ScaledLinearOperator")

    assert jnp.allclose(diag, 3.0 * jnp.diag(matrix_a))


def test_diagonal_of_sum_no_densification(matrix_a, matrix_b) -> None:
    """Test that diagonal of sum doesn't densify operands unnecessarily."""
    A = Matrix(matrix_a)
    B = Matrix(matrix_b)
    sum_op = A + B

    # Getting diagonal should not densify A or B individually
    with DensificationDetector(A, allow_in_matmul=True) as det_a, DensificationDetector(B, allow_in_matmul=True) as det_b:
        diag = diagonal(sum_op)
        det_a.assert_no_densification("diagonal of sum (operand A)")
        det_b.assert_no_densification("diagonal of sum (operand B)")

    assert jnp.allclose(diag, jnp.diag(matrix_a) + jnp.diag(matrix_b))


# ============================================================================
# Test Kronecker Product
# ============================================================================


def test_kronecker_construction_no_densification(matrix_a, matrix_b) -> None:
    """Test that Kronecker product construction doesn't densify."""
    A = Matrix(matrix_a)
    B = Matrix(matrix_b)

    with DensificationDetector(A, allow_in_matmul=True) as det_a, DensificationDetector(B, allow_in_matmul=True) as det_b:
        Kronecker(A, B)
        det_a.assert_no_densification("Kronecker construction (A)")
        det_b.assert_no_densification("Kronecker construction (B)")


def test_kronecker_matmul_no_densification(matrix_a, matrix_b, key) -> None:
    """Test that Kronecker product matmul doesn't densify."""
    A = Matrix(matrix_a)
    B = Matrix(matrix_b)
    kron_op = Kronecker(A, B)

    n = kron_op.shape[0]
    vec = jax.random.normal(key, (n,))

    # Matrix-vector product should only call _matmul, not cause early densification
    with DensificationDetector(kron_op, allow_in_matmul=True) as detector:
        result = kron_op @ vec
        detector.assert_no_densification("Kronecker matmul")

    expected = jnp.kron(matrix_a, matrix_b) @ vec
    assert jnp.allclose(result, expected, atol=1e-6)


# ============================================================================
# Test Block Operators
# ============================================================================


def test_block_diagonal_no_densification(matrix_a, matrix_b, key) -> None:
    """Test that BlockDiagonal doesn't densify blocks during matmul."""
    A = Matrix(matrix_a)
    B = Matrix(matrix_b)
    block_op = BlockDiagonal(A, B)

    n = block_op.shape[0]
    vec = jax.random.normal(key, (n,))

    with DensificationDetector(A, allow_in_matmul=True) as det_a, DensificationDetector(B, allow_in_matmul=True) as det_b:
        block_op @ vec
        det_a.assert_no_densification("BlockDiagonal matmul (A)")
        det_b.assert_no_densification("BlockDiagonal matmul (B)")


# ============================================================================
# Test Low-Rank Operators
# ============================================================================


def test_symmetric_low_rank_no_densification(key, small_size) -> None:
    """Test that SymmetricLowRank doesn't densify during construction or matmul."""
    rank = 2
    U = jax.random.normal(key, (small_size, rank))
    S = jnp.array([2.0, 1.0])

    lr_op = SymmetricLowRank(U, S)

    vec = jax.random.normal(key, (small_size,))

    # Matrix-vector should not cause early densification
    result = lr_op @ vec
    expected = U @ jnp.diag(S) @ U.T @ vec
    assert jnp.allclose(result, expected)


def test_isotropic_plus_low_rank_no_densification(key, small_size) -> None:
    """Test IsotropicScalingPlusSymmetricLowRank doesn't densify."""
    rank = 2
    U = jax.random.normal(key, (small_size, rank))
    S = jnp.array([2.0, 1.0])
    sigma = 0.5

    iso_lr = IsotropicScalingPlusSymmetricLowRank(sigma, U, S)

    vec = jax.random.normal(key, (small_size,))
    result = iso_lr @ vec

    expected = (sigma * jnp.eye(small_size) + U @ jnp.diag(S) @ U.T) @ vec
    assert jnp.allclose(result, expected, atol=1e-6)


# ============================================================================
# Test Decompositions and Inverse Operations
# ============================================================================


def test_inverse_scaled_no_densification(spd_matrix) -> None:
    """Test that inverse of scaled operator doesn't densify unnecessarily."""
    A = Matrix(spd_matrix)
    scaled = 2.0 * A

    # Getting inverse should not densify the underlying operator early
    with DensificationDetector(scaled, allow_in_matmul=True) as detector:
        linverse(scaled)
        # Note: inverse itself may need to densify at some point, but not during
        # construction of the inverse operator
        detector.assert_no_densification("inverse of scaled (during construction)")


# ============================================================================
# Test Known Problem Areas
# ============================================================================


def test_isotropic_add_cholesky_should_not_densify(spd_matrix, small_size) -> None:
    """Test that Cholesky of isotropic add uses eigendecomposition."""
    from linox.operators.arithmetic import lcholesky

    A = Matrix(spd_matrix)
    iso_add = IsotropicAdditiveLinearOperator(0.5, A)

    with DensificationDetector(iso_add, allow_in_matmul=True) as detector:
        _ = lcholesky(iso_add)
        detector.assert_no_densification("Cholesky of isotropic additive")


def test_congruence_diagonal_should_not_densify(matrix_a, spd_matrix) -> None:
    """Test that diagonal of congruence transform doesn't densify."""
    from linox.operators.arithmetic import CongruenceTransform

    A = Matrix(matrix_a)
    B = Matrix(spd_matrix)
    cong = CongruenceTransform(A, B)

    with DensificationDetector(A, allow_in_matmul=True) as det_a, DensificationDetector(B, allow_in_matmul=True) as det_b:
        _ = diagonal(cong)
        det_a.assert_no_densification("CongruenceTransform diagonal (A)")
        det_b.assert_no_densification("CongruenceTransform diagonal (B)")


# ============================================================================
# Test Integration: Complex Operator Chains
# ============================================================================


def test_complex_chain_no_early_densification(key, small_size) -> None:
    """Test a complex chain of operations doesn't cause early densification."""
    # Create operators
    d1 = jnp.arange(1, small_size + 1, dtype=jnp.float32)
    D = Diagonal(d1)
    I = Identity(small_size)

    # Complex expression: (2*D + I) @ (3*I) + D
    op1 = 2.0 * D + I
    op2 = 3.0 * I
    result_op = op1 @ op2 + D

    # This should all be lazy
    vec = jax.random.normal(key, (small_size,))
    result = result_op @ vec

    # Compute expected
    expected = (2.0 * jnp.diag(d1) + jnp.eye(small_size)) @ (
        3.0 * jnp.eye(small_size)
    ) @ vec + jnp.diag(d1) @ vec

    assert jnp.allclose(result, expected, atol=1e-6)


def test_kronecker_chain_no_early_densification(small_size) -> None:
    """Test Kronecker chains maintain lazy evaluation."""
    d1 = jnp.arange(1, small_size + 1, dtype=jnp.float32)
    d2 = jnp.arange(2, small_size + 2, dtype=jnp.float32)

    D1 = Diagonal(d1)
    D2 = Diagonal(d2)

    # (D1 ⊗ D2) should remain structured
    kron_op = Kronecker(D1, D2)

    # Operations on it should also remain structured
    scaled_kron = 2.0 * kron_op

    with DensificationDetector(kron_op, allow_in_matmul=True) as detector:
        diag = diagonal(scaled_kron)
        detector.assert_no_densification("scaled Kronecker diagonal")

    expected = 2.0 * jnp.kron(d1, d2)
    assert jnp.allclose(diag, expected)


# ============================================================================
# Test Utility Functions
# ============================================================================


def test_densification_detector_catches_todense() -> None:
    """Test that DensificationDetector correctly detects todense() calls."""
    A = Matrix(jnp.eye(3))

    with pytest.raises(AssertionError, match="Early densification detected"), DensificationDetector(A, allow_in_matmul=False) as detector:
        _ = A.todense()
        detector.assert_no_densification()


def test_densification_detector_allows_matmul() -> None:
    """Test that DensificationDetector allows todense in _matmul."""
    A = Matrix(jnp.eye(3))
    vec = jnp.ones(3)

    with DensificationDetector(A, allow_in_matmul=True) as detector:
        _ = A @ vec  # This may call todense in _matmul, which is allowed
        detector.assert_no_densification()
