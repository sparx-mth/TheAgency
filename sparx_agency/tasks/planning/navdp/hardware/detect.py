"""Re-export shim: canonical hardware detection now lives in
:mod:`sparx_agency.tasks.common.hardware.detect`. Kept so existing imports of
``sparx_agency.tasks.planning.navdp.hardware.detect`` keep working unchanged.

Note: the canonical ``HardwareProfile.allow_dla`` reports hardware DLA
*capability*, not a per-model recommendation. NavDP's own build path never
reads ``allow_dla`` (its graphs are ViT/LayerNorm transformers, a poor DLA
fit) -- that choice lives in this task's own build code, not in the shared
hardware profile.
"""
from sparx_agency.tasks.common.hardware.detect import (  # noqa: F401
    HardwareProfile,
    detect,
    main,
    _is_jetson,
    _jetson_fill,
    _nvpmodel,
    _power_budget_w,
    _read,
    _run,
    _target_tag,
    _workspace_bytes,
    _x86_compute_capability,
    _x86_gpu,
)

if __name__ == "__main__":
    main()
