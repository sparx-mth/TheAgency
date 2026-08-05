"""Re-export shim: canonical hardware detection now lives in
:mod:`sparx_agency.tasks.common.hardware.detect`. Kept so existing imports of
``sparx_agency.tasks.mapping.yolo_world_trt.hardware`` keep working unchanged.
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
