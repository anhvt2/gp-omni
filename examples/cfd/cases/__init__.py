"""All design cases by name."""

from . import axial, external, external_structured, internal, internal_cad, radial

_MODULES = (axial, radial, internal, internal_cad, external_structured, external)
_ALL = [case for module in _MODULES for case in module.CASES]
CASES = {case.name: case for case in _ALL}
if len(CASES) != len(_ALL):
    raise RuntimeError("case names must be unique")
