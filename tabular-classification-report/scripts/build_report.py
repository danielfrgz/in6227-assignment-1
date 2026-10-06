#!/usr/bin/env python3
"""Render a report written in a small Markdown subset to PDF.

This script only *renders*. It never changes wording or numbers; the agent
writes ``report.md`` and this script lays it out.

Layout follows the course report template: A4, Times New Roman 10 pt body
text (or the closest embedded serif font available), justified, bold upper-case
section headings, tables and figures inline. Tables and captions use 9 pt.

Supported Markdown subset:

* A header block at the very top between two ``---`` lines, with
  ``key: value`` lines: ``title``, ``name``, ``matric``, ``assignment``,
  ``variant``, ``model``, ``interface``, ``skill``.
* ``## Heading`` for section headings (rendered upper-case).
* Paragraphs separated by blank lines; ``**bold**``, ``*italic*`` and
  ```code``` inline.
* Bullet lists with ``- ``.
* Pipe tables (header row, separator row, body rows).
* Figures: ``![caption](relative/path.png){width=60%}`` (width optional,
  default 100% of the text width).
* HTML comments ``<!-- ... -->`` are removed (the template uses them for
  instructions).

Example::

    python build_report.py run/report.md --out run/report.pdf
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, ListFlowable, ListItem,
                                Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

# Serif TrueType fonts to embed, in order of preference: the template's Times New
# Roman (macOS, Windows), then metric-compatible or common Linux serif fonts.
# Embedding makes the PDF render identically in every viewer and covers symbols
# such as the "greater or equal" sign that the built-in PDF fonts lack.
FONT_CANDIDATES = [
    ("Times New Roman", ["/System/Library/Fonts/Supplemental", "/Library/Fonts",
                         "C:/Windows/Fonts", "/usr/share/fonts/truetype/msttcorefonts"],
     ["Times New Roman.ttf", "Times New Roman Bold.ttf",
      "Times New Roman Italic.ttf", "Times New Roman Bold Italic.ttf"]),
    ("Times New Roman", ["C:/Windows/Fonts"],
     ["times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"]),
    ("Liberation Serif", ["/usr/share/fonts/truetype/liberation",
                          "/usr/share/fonts/liberation"],
     ["LiberationSerif-Regular.ttf", "LiberationSerif-Bold.ttf",
      "LiberationSerif-Italic.ttf", "LiberationSerif-BoldItalic.ttf"]),
    ("DejaVu Serif", ["/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/dejavu"],
     ["DejaVuSerif.ttf", "DejaVuSerif-Bold.ttf",
      "DejaVuSerif-Italic.ttf", "DejaVuSerif-BoldItalic.ttf"]),
]


def register_fonts() -> tuple[str, str, str]:
    """Register the first available serif family; return (regular, bold, italic).

    Falls back to the built-in Times fonts (not embedded) with a warning.
    """
    for family, folders, files in FONT_CANDIDATES:
        for folder in folders:
            paths = [Path(folder) / f for f in files]
            if all(p.exists() for p in paths):
                names = ["Serif", "Serif-Bold", "Serif-Italic", "Serif-BoldItalic"]
                for name, path in zip(names, paths):
                    pdfmetrics.registerFont(TTFont(name, str(path)))
                pdfmetrics.registerFontFamily("Serif", normal=names[0], bold=names[1],
                                              italic=names[2], boldItalic=names[3])
                print(f"font: {family} (embedded)", file=sys.stderr)
                return names[0], names[1], names[2]
    print("warning: no serif TrueType font found; using built-in Times (not embedded,"
          " symbols such as >= may not render)", file=sys.stderr)
    return "Times-Roman", "Times-Bold", "Times-Italic"


REGULAR, BOLD, ITALIC = register_fonts()

BODY_SIZE = 10          # template: body text 10 pt, never reduced to fit
LEADING = 11.5          # template line spacing (about 1.15)
SMALL_SIZE = 9          # table cells and figure captions
MARGIN = 2 * cm

STYLES = {
    "title": ParagraphStyle("title", fontName=BOLD, fontSize=16,
                            leading=19, alignment=TA_CENTER, spaceAfter=3),
    "author": ParagraphStyle("author", fontName=REGULAR, fontSize=11,
                             leading=13, alignment=TA_CENTER),
    "meta": ParagraphStyle("meta", fontName=REGULAR, fontSize=BODY_SIZE,
                           leading=LEADING, alignment=TA_CENTER, spaceAfter=6),
    "heading": ParagraphStyle("heading", fontName=BOLD, fontSize=BODY_SIZE,
                              leading=LEADING, spaceBefore=6, spaceAfter=2),
    "body": ParagraphStyle("body", fontName=REGULAR, fontSize=BODY_SIZE,
                           leading=LEADING, alignment=TA_JUSTIFY, spaceAfter=4),
    "cell": ParagraphStyle("cell", fontName=REGULAR, fontSize=SMALL_SIZE,
                           leading=SMALL_SIZE + 1.5),
    "caption": ParagraphStyle("caption", fontName=ITALIC, fontSize=SMALL_SIZE,
                              leading=SMALL_SIZE + 1.5, alignment=TA_CENTER,
                              spaceAfter=4),
}

HEADER_KEYS = ["title", "name", "matric", "assignment", "variant",
               "model", "interface", "skill"]


def inline(text: str) -> str:
    """Escape XML characters and convert inline Markdown to reportlab markup."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"`([^`]+)`", r'<font face="Courier">\1</font>', text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", text)
    return text


def split_header(lines: list[str]) -> tuple[dict, list[str]]:
    """Separate the ``---`` header block from the body lines."""
    if not lines or lines[0].strip() != "---":
        return {}, lines
    try:
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    except StopIteration:
        sys.exit("header block opened with --- but never closed")
    header = {}
    for line in lines[1:end]:
        key, sep, value = line.partition(":")
        if sep:
            header[key.strip().lower()] = value.strip()
    return header, lines[end + 1:]


def header_flowables(header: dict) -> list:
    """Title, author line, assignment line and metadata line."""
    missing = [k for k in HEADER_KEYS if not header.get(k)]
    if missing:
        print(f"warning: header fields missing: {', '.join(missing)}", file=sys.stderr)
    out = []
    if header.get("title"):
        out.append(Paragraph(inline(header["title"]), STYLES["title"]))
    author = " · ".join(v for v in (header.get("matric"), header.get("name")) if v)
    course = " · ".join(v for v in (header.get("assignment"), header.get("variant")) if v)
    if author:
        out.append(Paragraph(inline(author), STYLES["author"]))
    if course:
        out.append(Paragraph(inline(course), STYLES["author"]))
    meta = []
    if header.get("model"):
        meta.append(f"Model: {header['model']}")
    if header.get("interface"):
        meta.append(f"Interface: {header['interface']}")
    if header.get("skill"):
        meta.append(f"Skill: {header['skill']}")
    if meta:
        out.append(Paragraph(inline(" · ".join(meta)), STYLES["meta"]))
    return out


def parse_table(rows: list[str], width: float) -> Table:
    """Build a table from pipe-table lines (the separator row is skipped)."""
    cells = [[c.strip() for c in row.strip().strip("|").split("|")] for row in rows]
    cells = [r for i, r in enumerate(cells) if not (i == 1 and all(re.fullmatch(r":?-+:?", c) for c in r))]
    ncols = max(len(r) for r in cells)
    data = []
    for i, row in enumerate(cells):
        row = row + [""] * (ncols - len(row))
        text = [f"<b>{inline(c)}</b>" if i == 0 else inline(c) for c in row]
        data.append([Paragraph(t, STYLES["cell"]) for t in text])
    table = Table(data, colWidths=[width / ncols] * ncols, repeatRows=1, hAlign="CENTER")
    table.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 1.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
    ]))
    return table


def figure(caption: str, path: Path, width_frac: float, text_width: float) -> KeepTogether:
    """Scaled image with a caption underneath."""
    if not path.exists():
        sys.exit(f"figure not found: {path}")
    img = Image(str(path))
    scale = text_width * width_frac / img.imageWidth
    img.drawWidth, img.drawHeight = img.imageWidth * scale, img.imageHeight * scale
    parts = [img]
    if caption:
        parts.append(Paragraph(inline(caption), STYLES["caption"]))
    return KeepTogether(parts)


FIGURE_RE = re.compile(r"!\[(?P<cap>[^\]]*)\]\((?P<src>[^)]+)\)(\{width=(?P<w>\d+)%\})?\s*$")


def body_flowables(lines: list[str], base: Path, text_width: float) -> list:
    """Convert the body lines to flowables, block by block."""
    out: list = []
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line.strip():
            i += 1
            continue
        if line.startswith("## ") or line.startswith("# "):
            out.append(Paragraph(inline(line.lstrip("#").strip().upper()), STYLES["heading"]))
            i += 1
        elif line.lstrip().startswith("|"):
            block = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                block.append(lines[i])
                i += 1
            out += [parse_table(block, text_width), Spacer(1, 4)]
        elif FIGURE_RE.match(line.strip()):
            m = FIGURE_RE.match(line.strip())
            frac = int(m["w"]) / 100 if m["w"] else 1.0
            out.append(figure(m["cap"], base / m["src"], frac, text_width))
            i += 1
        elif line.lstrip().startswith("- "):
            items = []
            while i < len(lines) and lines[i].lstrip().startswith("- "):
                items.append(lines[i].lstrip()[2:])
                i += 1
                while i < len(lines) and lines[i].startswith("  ") and lines[i].strip() \
                        and not lines[i].lstrip().startswith("- "):
                    items[-1] += " " + lines[i].strip()
                    i += 1
            out.append(ListFlowable(
                [ListItem(Paragraph(inline(t), STYLES["body"]), leftIndent=10) for t in items],
                bulletType="bullet", start="•", leftIndent=10, bulletFontName=REGULAR,
                bulletFontSize=BODY_SIZE))
        else:
            para = []
            while i < len(lines) and lines[i].strip() and not re.match(r"\s*(#|\||- |!\[)", lines[i]):
                para.append(lines[i].strip())
                i += 1
            out.append(Paragraph(inline(" ".join(para)), STYLES["body"]))
    return out


def page_number(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont(REGULAR, 8)
    canvas.drawCentredString(A4[0] / 2, MARGIN / 2, str(doc.page))
    canvas.restoreState()


def build(source: Path, target: Path) -> None:
    text = re.sub(r"<!--.*?-->", "", source.read_text(encoding="utf-8"), flags=re.S)
    header, body = split_header(text.splitlines())
    doc = SimpleDocTemplate(str(target), pagesize=A4, leftMargin=MARGIN,
                            rightMargin=MARGIN, topMargin=MARGIN, bottomMargin=MARGIN,
                            title=header.get("title", ""), author=header.get("name", ""))
    story = header_flowables(header) + body_flowables(body, source.parent, doc.width)
    doc.build(story, onFirstPage=page_number, onLaterPages=page_number)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("report", type=Path, help="report.md to render")
    parser.add_argument("--out", type=Path, required=True, help="output PDF path")
    args = parser.parse_args()
    build(args.report, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
