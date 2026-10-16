"""Gibson's import path for the shared perception-service options (``..detector_options``)."""
from sparx_agency.tasks.planning.objnav_benchmark_runtime.detector_options import *  # noqa: F401,F403
from sparx_agency.tasks.planning.objnav_benchmark_runtime.detector_options import (  # noqa: F401
    DEFAULT_DETECTOR_BACKEND, add_detector_options, add_segmenter_options, detector_flags, segmenter_flags)
