#!/usr/bin/env python3
"""Check that a PDF does not exceed a page limit.

Prints the page count. Exit code 0 if the PDF has at most ``--max-pages``
pages, 1 if it has more, 2 if the file cannot be read.

Example::

    python check_pages.py run/report.pdf --max-pages 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pypdf import PdfReader


def count_pages(path: Path) -> int:
    """Return the number of pages in the PDF at ``path``."""
    return len(PdfReader(str(path)).pages)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("pdf", type=Path, help="PDF to check")
    parser.add_argument("--max-pages", type=int, default=2, help="page limit (default 2)")
    args = parser.parse_args()
    try:
        pages = count_pages(args.pdf)
    except Exception as exc:  # unreadable or missing file
        print(f"cannot read {args.pdf}: {exc}", file=sys.stderr)
        sys.exit(2)
    status = "OK" if pages <= args.max_pages else "OVER LIMIT"
    print(f"{args.pdf}: {pages} page(s), limit {args.max_pages}: {status}")
    sys.exit(0 if pages <= args.max_pages else 1)


if __name__ == "__main__":
    main()
