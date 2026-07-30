"""Allow ``python -m sparx_agency.tools.pdf_parser``."""
from __future__ import annotations

import sys

from sparx_agency.tools.pdf_parser.cli import main

if __name__ == "__main__":
    sys.exit(main())
