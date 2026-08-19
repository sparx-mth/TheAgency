"""Run the TensorRT optimizer CLI: ``python -m sparx_agency.tasks.common.trt_optimizer``."""
from __future__ import annotations

import sys

from sparx_agency.tasks.common.trt_optimizer.cli import main

sys.exit(main())
