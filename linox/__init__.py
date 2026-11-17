"""linox: Linear operators in JAX."""

from . import api
from .api import *  # noqa: F403

# Deprecated "l"-prefixed aliases, kept for backwards compatibility and
# scheduled for removal in 0.0.4. These are renames rather than plain
# re-exports, so they are listed in `__all__` below instead of using the
# redundant `X as X` form that marks an intentional re-export.
from .api import _broadcast_shapes as _broadcast_shapes
from .api import eigh as leigh  # noqa: F401
from .api import eye as identity  # noqa: F401
from .api import lexp as lexp
from .api import llog as llog
from .api import lpow as lpow
from .api import solve as lsolve  # noqa: F401
from .api import trace as ltrace  # noqa: F401

# `linverse`, `lpinverse` and `lsqrt` are the plum-dispatched implementations
# rather than aliases of `inv`/`pinv`/`sqrt`: downstream code (e.g. luno)
# registers its own rules through `linverse.dispatch` etc., and those rules are
# what `inv`, `inverse`, `pinv` and `sqrt` dispatch to.
from .operators.arithmetic import linverse as linverse
from .operators.arithmetic import lpinverse as lpinverse
from .operators.arithmetic import lsqrt as lsqrt

# Backward compatibility imports for kernel operators
from .operators.kernel import ArrayKernel as ArrayKernel
from .operators.kernel import Kernel as Kernel
from .operators.kernel import KernelLinearOperator as KernelLinearOperator
from .operators.kernel import ToeplitzKernel as ToeplitzKernel
from .operators.kernel import kernel_operator as kernel_operator

__version__ = "0.0.3"


_LEGACY_ALIASES = [
    "identity",
    "leigh",
    "lexp",
    "linverse",
    "llog",
    "lpinverse",
    "lpow",
    "lsolve",
    "lsqrt",
    "ltrace",
]

_KERNEL_EXPORTS = [
    "ArrayKernel",
    "Kernel",
    "KernelLinearOperator",
    "ToeplitzKernel",
    "kernel_operator",
]

# `api.__all__` already carries several of the legacy alias names, so
# de-duplicate rather than concatenating blindly: a name repeated in `__all__`
# breaks star-import tooling. `_broadcast_shapes` stays importable for
# backwards compatibility but is deliberately not advertised as public.
__all__ = sorted({*api.__all__, *_KERNEL_EXPORTS, *_LEGACY_ALIASES})
