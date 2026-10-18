"""Source adapters, grouped by team, and the catalog that lists them.

Each module (crypto, stocks, web, and sim for the simulator) exports a list of adapter
classes. `catalog()` loads them defensively: a module that fails to import is skipped and
its error is kept in IMPORT_ERRORS (the health panel shows it), so one broken file never
stops the research service. Roles decide what runs: primaries run from the start, standbys
wait for their primary to fail, dormant adapters are the reserve the gap detector grows into.
"""
from __future__ import annotations

import importlib

MODULES = ("crypto", "stocks", "web")
ROLES = ("primary", "standby", "dormant")

IMPORT_ERRORS = {}   # module (or module.Class) -> "ExceptionType: message"


def _classes(modname, attr):
    try:
        mod = importlib.import_module(f"{__name__}.{modname}")
    except Exception as e:   # ImportError, or a syntax/name error in a half-written module
        IMPORT_ERRORS[modname] = f"{type(e).__name__}: {e}"[:300]
        return []
    IMPORT_ERRORS.pop(modname, None)
    try:
        return list(getattr(mod, attr, None) or [])
    except TypeError:
        IMPORT_ERRORS[modname] = f"{attr} is not a list"
        return []


def role_of(adapter):
    r = getattr(adapter, "role", "primary")
    return r if r in ROLES else "primary"


def catalog(cfg, sim=False):
    """Adapter instances: every class in ADAPTERS of crypto/stocks/web (or SIM_ADAPTERS of sim),
    minus the names in cfg["disable"]. Duplicate names keep the first."""
    cfg = cfg or {}
    disable = set(cfg.get("disable") or [])
    classes = _classes("sim", "SIM_ADAPTERS") if sim else [c for m in MODULES for c in _classes(m, "ADAPTERS")]
    out, seen = [], set()
    for cls in classes:
        label = getattr(cls, "__name__", type(cls).__name__)
        try:
            a = cls() if isinstance(cls, type) else cls
        except Exception as e:
            IMPORT_ERRORS[label] = f"{type(e).__name__}: {e}"[:300]
            continue
        name = getattr(a, "name", None)
        if not isinstance(name, str) or not callable(getattr(a, "fetch", None)):
            IMPORT_ERRORS[label] = "not an adapter (needs a name and fetch())"
            continue
        if name in disable or name in seen:
            continue
        seen.add(name)
        out.append(a)
    return out


def count_roles(adapters):
    out = dict(primary=0, standby=0, dormant=0)
    for a in adapters:
        out[role_of(a)] += 1
    return out
