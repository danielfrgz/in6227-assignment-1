#!/usr/bin/env python3
"""Render a report written in a small Markdown subset to a two-column PDF.

This script only *renders*. It never changes wording or numbers; the agent
writes ``report.md`` and this script lays it out.

Layout (all values are named settings in the LAYOUT section below): A4, a
full-width title block on page 1, then two justified columns of Times New Roman
10 pt (or the closest embedded serif font), bold upper-case section headings,
paragraphs separated by a first-line indent, a grey running header (the course
header by default) and
a two-field footer (document title left, author right).

Placement rules:

* Body text, lists and figures flow through the columns. A figure's width is a
  percentage of the full text width and is capped at the column width.
* Tables, and figures marked ``{span=full}``, are full-width *floats*: they are
  placed at the top of the next page, above the columns (the usual placement
  for spanning tables in two-column papers). Refer to them by name in the text,
  not as "below".

Supported Markdown subset:

* A header block at the very top between two ``---`` lines, with
  ``key: value`` lines. Title block: ``title``, ``name``, ``matric``,
  ``assignment``, ``variant``, ``model``, ``interface``, ``skill``, ``repo``
  (a missing field produces a warning). Optional page furniture:
  ``running_header`` (top-left text on every page; defaults to the course
  header RUNNING_HEADER_DEFAULT with the current year); the footer shows
  ``title`` and ``name``, each omitted if absent.
* ``## Heading`` for section headings (rendered upper-case).
* Paragraphs separated by blank lines; ``**bold**``, ``*italic*`` and
  ```code``` inline.
* Bullet lists with ``- ``.
* Pipe tables (header row, separator row, body rows).
* Figures: ``![caption](relative/path.png){width=60%}`` or
  ``{width=60%,span=full}``.
* HTML comments ``<!-- ... -->`` are removed (the template uses them for
  instructions).

Example::

    python build_report.py run/report.md --out run/report.pdf
"""

from __future__ import annotations

import argparse
import datetime
import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, FrameBreak, Image,
                                KeepTogether, ListFlowable, ListItem,
                                NextPageTemplate, PageBreak, PageTemplate,
                                Paragraph, Spacer, Table, TableStyle)

# ---------------------------------------------------------------------------
# LAYOUT. Values measured from the course report template (points, 1/72 in).
# ---------------------------------------------------------------------------
PAGE_SIZE = A4
MARGIN_LEFT = 54.0                 # measured
MARGIN_RIGHT = 54.0                # measured (53.5, rounded)
MARGIN_TOP = 72.0                  # inferred from title position
MARGIN_BOTTOM = 72.0               # not measurable (template page not full); mirrors top
COLUMN_GUTTER = 28.2               # measured
TEXT_WIDTH = PAGE_SIZE[0] - MARGIN_LEFT - MARGIN_RIGHT
COLUMN_WIDTH = (TEXT_WIDTH - COLUMN_GUTTER) / 2          # = 229.5, measured 229.6
TEXT_HEIGHT = PAGE_SIZE[1] - MARGIN_TOP - MARGIN_BOTTOM

TITLE_SIZE = 24                    # measured 24.0, bold, centred
TITLE_LEADING = 27.6
TITLE_SPACE_BEFORE = 12            # .doc paragraph space-before
TITLE_SPACE_AFTER = 3              # .doc paragraph space-after
AUTHOR_SIZE = 11                   # measured 11.04, centred
AUTHOR_LEADING = 13.5              # measured line pitch 13.55
META_SIZE = 10
META_LEADING = 11.5
TITLE_BLOCK_SPACE_AFTER = 23.0     # measured gap to the first heading (2 x body leading)

BODY_SIZE = 10                     # measured 10.08; never reduced to fit
BODY_LEADING = 11.52               # measured
FIRST_LINE_INDENT = 12.0           # measured; not applied after a heading
HEADING_SPACE_BEFORE = 11.52       # one empty line, measured
HEADING_SPACE_AFTER = 11.52        # one empty line, measured
HEADING_KEEP_WITH_NEXT = True
LIST_INDENT = 10.0

SMALL_SIZE = 9                     # table cells and figure captions
SMALL_LEADING = 10.5
FLOAT_GAP = 6.0                    # between stacked floats
FLOAT_SPACE_AFTER = 11.52          # between the floats and the columns below

RUNNING_HEADER_DEFAULT = "IN6227 Data Mining {year} WKWSCI"   # course header; {year} = build year
RUNNING_HEADER_FONT = "Helvetica-Bold"   # measured
RUNNING_HEADER_SIZE = 10                  # measured 10.08
RUNNING_HEADER_COLOR = colors.Color(0.651, 0.651, 0.651)   # measured grey
RUNNING_HEADER_BASELINE_FROM_TOP = 44.4   # measured
FOOTER_SIZE = 9                           # measured 9.12
FOOTER_COLOR = colors.Color(0.267, 0.447, 0.769)            # measured blue
FOOTER_BASELINE_FROM_BOTTOM = 48.5        # measured (box bottom 46.5 pt above edge)
FOOTER_MIN_GAP = 18.0                     # between the left and right footer fields
INLINE_CODE_FONT = None                   # None = body font; the template has no monospace
                                          # font, and Courier is too wide for a column
TABLE_CELL_PADDING = 4.0                  # added to each column's natural width
# ---------------------------------------------------------------------------

# Serif TrueType fonts to embed, in order of preference: Times New Roman
# (macOS, Windows), then metric-compatible or common Linux serif fonts.
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

STYLES = {
    "title": ParagraphStyle("title", fontName=BOLD, fontSize=TITLE_SIZE,
                            leading=TITLE_LEADING, alignment=TA_CENTER,
                            spaceBefore=TITLE_SPACE_BEFORE, spaceAfter=TITLE_SPACE_AFTER),
    "author": ParagraphStyle("author", fontName=REGULAR, fontSize=AUTHOR_SIZE,
                             leading=AUTHOR_LEADING, alignment=TA_CENTER),
    "meta": ParagraphStyle("meta", fontName=REGULAR, fontSize=META_SIZE,
                           leading=META_LEADING, alignment=TA_CENTER),
    "heading": ParagraphStyle("heading", fontName=BOLD, fontSize=BODY_SIZE,
                              leading=BODY_LEADING, spaceBefore=HEADING_SPACE_BEFORE,
                              spaceAfter=HEADING_SPACE_AFTER,
                              keepWithNext=int(HEADING_KEEP_WITH_NEXT)),
    "body_first": ParagraphStyle("body_first", fontName=REGULAR, fontSize=BODY_SIZE,
                                 leading=BODY_LEADING, alignment=TA_JUSTIFY),
    "body": ParagraphStyle("body", fontName=REGULAR, fontSize=BODY_SIZE,
                           leading=BODY_LEADING, alignment=TA_JUSTIFY,
                           firstLineIndent=FIRST_LINE_INDENT),
    "cell": ParagraphStyle("cell", fontName=REGULAR, fontSize=SMALL_SIZE,
                           leading=SMALL_LEADING),
    "caption": ParagraphStyle("caption", fontName=ITALIC, fontSize=SMALL_SIZE,
                              leading=SMALL_LEADING, alignment=TA_CENTER,
                              spaceBefore=2, spaceAfter=BODY_LEADING / 2),
}

HEADER_KEYS = ["title", "name", "matric", "assignment", "variant",
               "model", "interface", "skill", "repo"]
OPTIONAL_KEYS = ["running_header"]


def inline(text: str) -> str:
    """Escape XML characters and convert inline Markdown to reportlab markup."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    if INLINE_CODE_FONT:
        text = re.sub(r"`([^`]+)`", rf'<font face="{INLINE_CODE_FONT}">\1</font>', text)
    else:
        text = re.sub(r"`([^`]+)`", r"\1", text)
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
        if sep and value.strip():
            header[key.strip().lower()] = value.strip()
    return header, lines[end + 1:]


def title_block(header: dict) -> list:
    """Title, author line, assignment line, metadata and repository lines."""
    missing = [k for k in HEADER_KEYS if not header.get(k)]
    if missing:
        print(f"warning: header fields missing: {', '.join(missing)}", file=sys.stderr)
    out = []
    if header.get("title"):
        out.append(Paragraph(inline(header["title"]), STYLES["title"]))
    author = " · ".join(v for v in (header.get("matric"), header.get("name")) if v)
    course = " · ".join(v for v in (header.get("assignment"), header.get("variant")) if v)
    for line in (author, course):
        if line:
            out.append(Paragraph(inline(line), STYLES["author"]))
    meta = [f"{label}: {header[key]}" for key, label in
            (("model", "Model"), ("interface", "Interface"), ("skill", "Skill"))
            if header.get(key)]
    if meta:
        out.append(Paragraph(inline(" · ".join(meta)), STYLES["meta"]))
    if header.get("repo"):
        out.append(Paragraph(f"Repository: {inline(header['repo'])}", STYLES["meta"]))
    return out


def block_height(flowables: list, width: float) -> float:
    """Height the flowables occupy when stacked at ``width``, including spacing."""
    total = 0.0
    for f in flowables:
        total += f.wrap(width, TEXT_HEIGHT)[1] + f.getSpaceBefore() + f.getSpaceAfter()
    return total


def column_widths(cells: list[list[str]], ncols: int) -> list[float]:
    """Share TEXT_WIDTH between columns in proportion to their widest cell."""
    natural = []
    for c in range(ncols):
        texts = [row[c] if c < len(row) else "" for row in cells]
        widest = max(pdfmetrics.stringWidth(re.sub(r"[*`]", "", t), BOLD, SMALL_SIZE)
                     for t in texts)
        natural.append(widest + TABLE_CELL_PADDING)
    scale = TEXT_WIDTH / sum(natural)
    return [w * scale for w in natural]


def parse_table(rows: list[str]) -> Table:
    """Full-width table from pipe-table lines (the separator row is skipped)."""
    cells = [[c.strip() for c in row.strip().strip("|").split("|")] for row in rows]
    cells = [r for i, r in enumerate(cells)
             if not (i == 1 and all(re.fullmatch(r":?-+:?", c) for c in r))]
    ncols = max(len(r) for r in cells)
    data = []
    for i, row in enumerate(cells):
        row = row + [""] * (ncols - len(row))
        text = [f"<b>{inline(c)}</b>" if i == 0 else inline(c) for c in row]
        data.append([Paragraph(t, STYLES["cell"]) for t in text])
    table = Table(data, colWidths=column_widths(cells, ncols), hAlign="CENTER")
    table.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 1.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
    ]))
    return table


def figure(caption: str, path: Path, width: float) -> list:
    """Image scaled to ``width`` with an optional caption underneath."""
    if not path.exists():
        sys.exit(f"figure not found: {path}")
    img = Image(str(path))
    scale = width / img.imageWidth
    img.drawWidth, img.drawHeight = img.imageWidth * scale, img.imageHeight * scale
    parts = [img]
    if caption:
        parts.append(Paragraph(inline(caption), STYLES["caption"]))
    return parts


FIGURE_RE = re.compile(r"!\[(?P<cap>[^\]]*)\]\((?P<src>[^)]+)\)(\{(?P<opts>[^}]*)\})?\s*$")


def figure_options(opts: str | None) -> tuple[float, bool]:
    """Return (width fraction of the text width, full-width float?)."""
    frac, full = 1.0, False
    for opt in (opts or "").split(","):
        key, _, value = opt.strip().partition("=")
        if key == "width" and value.endswith("%"):
            frac = int(value[:-1]) / 100
        elif key == "span" and value == "full":
            full = True
    return frac, full


def body_flowables(lines: list[str], base: Path) -> tuple[list, list]:
    """Convert body lines to (column flowables, full-width floats)."""
    flow: list = []
    floats: list = []
    after_heading = True
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line.strip():
            i += 1
            continue
        if line.startswith("## ") or line.startswith("# "):
            flow.append(Paragraph(inline(line.lstrip("#").strip().upper()), STYLES["heading"]))
            after_heading = True
            i += 1
            continue
        if line.lstrip().startswith("|"):
            block = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                block.append(lines[i])
                i += 1
            floats.append([parse_table(block)])
        elif FIGURE_RE.match(line.strip()):
            m = FIGURE_RE.match(line.strip())
            frac, full = figure_options(m["opts"])
            if full:
                floats.append(figure(m["cap"], base / m["src"], TEXT_WIDTH * frac))
            else:
                width = min(TEXT_WIDTH * frac, COLUMN_WIDTH)
                flow.append(KeepTogether(figure(m["cap"], base / m["src"], width)))
            i += 1
        elif line.lstrip().startswith("- "):
            items = []
            while i < len(lines) and lines[i].lstrip().startswith("- "):
                items.append(lines[i].lstrip()[2:])
                i += 1
                while (i < len(lines) and lines[i].startswith("  ") and lines[i].strip()
                       and not lines[i].lstrip().startswith("- ")):
                    items[-1] += " " + lines[i].strip()
                    i += 1
            flow.append(ListFlowable(
                [ListItem(Paragraph(inline(t), STYLES["body_first"]), leftIndent=LIST_INDENT)
                 for t in items],
                bulletType="bullet", start="•", leftIndent=LIST_INDENT,
                bulletFontName=REGULAR, bulletFontSize=BODY_SIZE))
        else:
            para = []
            while (i < len(lines) and lines[i].strip()
                   and not re.match(r"\s*(#|\||- |!\[)", lines[i])):
                para.append(lines[i].strip())
                i += 1
            style = STYLES["body_first"] if after_heading else STYLES["body"]
            flow.append(Paragraph(inline(" ".join(para)), style))
        after_heading = False
    return flow, floats


def column_frames(top: float, prefix: str) -> list[Frame]:
    """Two column frames whose tops are at ``top`` (measured from the page bottom)."""
    height = top - MARGIN_BOTTOM
    return [Frame(MARGIN_LEFT + n * (COLUMN_WIDTH + COLUMN_GUTTER), MARGIN_BOTTOM,
                  COLUMN_WIDTH, height, id=f"{prefix}-col{n}",
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
            for n in (0, 1)]


def fit_left(text: str, font: str, size: float, max_width: float) -> str:
    """Shorten ``text`` with an ellipsis until it fits ``max_width``."""
    if pdfmetrics.stringWidth(text, font, size) <= max_width:
        return text
    while text and pdfmetrics.stringWidth(text + "…", font, size) > max_width:
        text = text[:-1]
    return text.rstrip() + "…"


class ReportDoc(BaseDocTemplate):
    """Two-column document: title page, an optional float page, later pages."""

    def __init__(self, target: Path, header: dict, title_height: float,
                 floats: list, **kwargs):
        super().__init__(str(target), pagesize=PAGE_SIZE, leftMargin=MARGIN_LEFT,
                         rightMargin=MARGIN_RIGHT, topMargin=MARGIN_TOP,
                         bottomMargin=MARGIN_BOTTOM, title=header.get("title", ""),
                         author=header.get("name", ""), **kwargs)
        self.header = header
        self.floats = floats
        self.floats_drawn = False
        page_top = PAGE_SIZE[1] - MARGIN_TOP
        title_frame = Frame(MARGIN_LEFT, page_top - title_height, TEXT_WIDTH, title_height,
                            id="title", leftPadding=0, rightPadding=0,
                            topPadding=0, bottomPadding=0)
        first_cols = column_frames(page_top - title_height, "first")
        self.float_height = (sum(block_height(f, TEXT_WIDTH) for f in floats)
                             + FLOAT_GAP * max(len(floats) - 1, 0))
        templates = [PageTemplate("first", [title_frame] + first_cols, onPage=self.furniture)]
        if floats:
            float_cols = column_frames(page_top - self.float_height - FLOAT_SPACE_AFTER, "float")
            templates.append(PageTemplate("floats", float_cols, onPage=self.float_page,
                                          autoNextPageTemplate="later"))
        templates.append(PageTemplate("later", column_frames(page_top, "later"),
                                      onPage=self.furniture))
        self.addPageTemplates(templates)

    def furniture(self, canvas, doc) -> None:
        """Running header and footer; each part only if its field is present."""
        canvas.saveState()
        width, height = PAGE_SIZE
        text = self.header.get("running_header") or RUNNING_HEADER_DEFAULT.format(
            year=datetime.date.today().year)
        if text:
            canvas.setFont(RUNNING_HEADER_FONT, RUNNING_HEADER_SIZE)
            canvas.setFillColor(RUNNING_HEADER_COLOR)
            canvas.drawString(MARGIN_LEFT, height - RUNNING_HEADER_BASELINE_FROM_TOP,
                              text.upper())
        right = self.header.get("name", "")
        left = self.header.get("title", "")
        canvas.setFont(REGULAR, FOOTER_SIZE)
        canvas.setFillColor(FOOTER_COLOR)
        if right:
            canvas.drawRightString(width - MARGIN_RIGHT, FOOTER_BASELINE_FROM_BOTTOM, right)
        if left:
            room = (TEXT_WIDTH - FOOTER_MIN_GAP
                    - (pdfmetrics.stringWidth(right, REGULAR, FOOTER_SIZE) if right else 0))
            canvas.drawString(MARGIN_LEFT, FOOTER_BASELINE_FROM_BOTTOM,
                              fit_left(left, REGULAR, FOOTER_SIZE, room))
        canvas.restoreState()

    def float_page(self, canvas, doc) -> None:
        """Page furniture plus the full-width floats stacked above the columns."""
        self.furniture(canvas, doc)
        y = PAGE_SIZE[1] - MARGIN_TOP
        for parts in self.floats:
            for part in parts:
                w, h = part.wrap(TEXT_WIDTH, TEXT_HEIGHT)
                y -= part.getSpaceBefore() + h
                part.drawOn(canvas, MARGIN_LEFT + (TEXT_WIDTH - w) / 2, y)
                y -= part.getSpaceAfter()
            y -= FLOAT_GAP
        self.floats_drawn = True


def build(source: Path, target: Path) -> None:
    text = re.sub(r"<!--.*?-->", "", source.read_text(encoding="utf-8"), flags=re.S)
    header, body = split_header(text.splitlines())
    title = title_block(header)
    title_height = block_height(title, TEXT_WIDTH) + TITLE_BLOCK_SPACE_AFTER
    flow, floats = body_flowables(body, source.parent)
    next_template = "floats" if floats else "later"
    story = title + [FrameBreak(), NextPageTemplate(next_template)] + flow
    doc = ReportDoc(target, header, title_height, floats)
    if doc.float_height > TEXT_HEIGHT / 2:
        print("warning: full-width floats take more than half a page", file=sys.stderr)
    doc.build(story)
    if floats and not doc.floats_drawn:
        # Everything fitted on page 1: add a page so the floats are still shown.
        doc = ReportDoc(target, header, title_height, floats)
        # A trailing PageBreak alone is dropped, so a 1-pt spacer forces the page.
        doc.build(title + [FrameBreak(), NextPageTemplate(next_template)] + flow
                  + [PageBreak(), Spacer(1, 1)])
        if not doc.floats_drawn:
            sys.exit("error: full-width floats could not be placed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("report", type=Path, help="report.md to render")
    parser.add_argument("--out", type=Path, required=True, help="output PDF path")
    args = parser.parse_args()
    build(args.report, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
