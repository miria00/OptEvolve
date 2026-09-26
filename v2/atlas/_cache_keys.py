"""Compile-cache keys shared by every executable cache in the package.

There is exactly one definition. A second, device-blind copy in
atlas.backend.precision is what let a GPU solve receive a CPU-compiled
program after the copy in atlas.wrappers.km had already been fixed.
It lives directly under atlas (whose __init__ only configures jax) so that
importing it can never form a cycle through atlas.backend.

A compiled executable is bound to a device, so a key records each leaf's
placement as well as its shape and dtype. Host (numpy) leaves key as
"host"; a mix of host and device leaves costs an extra compile, never a
stale reuse.
"""
import jax


def placement(leaf):
    devices = getattr(leaf, "devices", None)
    if callable(devices):
        try:
            return tuple(sorted((d.platform, d.id) for d in devices()))
        except Exception:
            pass
    return ("host",)


def abstract(tree):
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    return (treedef, tuple((tuple(l.shape), str(l.dtype), placement(l)) for l in leaves))
