#!/usr/bin/env python3
"""Конвертация JSON-отчёта бенчмарка в аккуратно оформленный DOCX."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

DEFAULT_FONT = "Times New Roman"
DEFAULT_FONT_SIZE = 12
DEFAULT_SOURCE_MAX_CHARS = 800
GRAY = RGBColor(0x55, 0x55, 0x55)
RED = RGBColor(0xCC, 0x00, 0x00)
BLACK = RGBColor(0x00, 0x00, 0x00)


# ---------- Общие помощники ----------

def clean_text(text: Any) -> str:
    """Нормализация пробелов, переносов и служебных символов."""
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def truncate(text: str, limit: int) -> str:
    if limit and len(text) > limit:
        return text[:limit].rstrip() + "…"
    return text


def set_run_font(
    run,
    *,
    name: str = DEFAULT_FONT,
    size: int = DEFAULT_FONT_SIZE,
    bold: bool | None = None,
    italic: bool | None = None,
    color: RGBColor | None = None,
) -> None:
    """Задаёт шрифт прогона, включая кириллицу (w:cs, w:eastAsia)."""
    run.font.name = name
    run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if color is not None:
        run.font.color.rgb = color

    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rfonts.set(qn(attr), name)


def _fmt_float(value: Any, digits: int) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def _fmt_pct(value: Any) -> str:
    try:
        return f"{float(value):.2f} %"
    except (TypeError, ValueError):
        return "—"


def _fmt_sec(value: Any) -> str:
    try:
        return f"{float(value):.2f} с"
    except (TypeError, ValueError):
        return "—"


def _add_heading(doc: Document, text: str, *, level: int, size: int) -> None:
    p = doc.add_heading(level=level)
    run = p.add_run(text)
    set_run_font(run, size=size, bold=True, color=BLACK)


def _add_labeled_paragraph(
    doc: Document,
    label: str,
    text: str,
    *,
    label_size: int = 11,
    text_size: int = 11,
    text_italic: bool = False,
) -> None:
    p = doc.add_paragraph()
    run = p.add_run(f"{label}: ")
    set_run_font(run, size=label_size, bold=True)
    run = p.add_run(text)
    set_run_font(run, size=text_size, italic=text_italic)


# ---------- Настройка документа ----------

def _setup_document(doc: Document) -> None:
    style = doc.styles["Normal"]
    style.font.name = DEFAULT_FONT
    style.font.size = Pt(DEFAULT_FONT_SIZE)

    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rfonts.set(qn(attr), DEFAULT_FONT)

    for section in doc.sections:
        section.top_margin = Cm(2)
        section.bottom_margin = Cm(2)
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(2)
        _add_page_number_footer(section)


def _add_page_number_footer(section) -> None:
    p = section.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = "PAGE"
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_end)
    set_run_font(run, size=10)


# ---------- Блоки отчёта ----------

def _add_title(doc: Document, data: dict) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("Отчёт по бенчмарку")
    set_run_font(run, size=20, bold=True)

    created = data.get("created_at")
    if created:
        try:
            created = datetime.fromisoformat(created).strftime("%d.%m.%Y %H:%M")
        except ValueError:
            pass

    meta = " · ".join(
        part for part in (
            f"Модель: {data.get('model', '—')}",
            f"Дата: {created}" if created else None,
        ) if part
    )
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(meta)
    set_run_font(run, size=11, italic=True, color=GRAY)


def _add_summary(doc: Document, data: dict) -> None:
    _add_heading(doc, "Сводка", level=1, size=16)

    rows = [
        ("Всего вопросов",
         str(data.get("question_count", len(data.get("results", []))))),
        ("Средняя точность", _fmt_pct(data.get("average_accuracy"))),
        ("Средняя косинусная дистанция",
         _fmt_float(data.get("average_distance"), 4)),
        ("Среднее время ответа", _fmt_sec(data.get("average_seconds"))),
    ]

    table = doc.add_table(rows=len(rows), cols=2)
    try:
        table.style = "Light Grid Accent 1"
    except KeyError:
        table.style = "Table Grid"

    for row_idx, (key, value) in enumerate(rows):
        cell_k = table.cell(row_idx, 0)
        cell_v = table.cell(row_idx, 1)
        cell_k.text = ""
        run = cell_k.paragraphs[0].add_run(key)
        set_run_font(run, size=DEFAULT_FONT_SIZE, bold=True)
        cell_v.text = ""
        run = cell_v.paragraphs[0].add_run(value)
        set_run_font(run, size=DEFAULT_FONT_SIZE)

    doc.add_paragraph()


def _add_question(
    doc: Document,
    item: dict,
    *,
    source_max_chars: int,
    include_sources: bool,
) -> None:
    number = item.get("number", "?")
    question = clean_text(item.get("question")) or "(без вопроса)"

    # Заголовок вопроса
    heading = doc.add_heading(level=2)
    run = heading.add_run(f"{number}. {question}")
    set_run_font(run, size=14, bold=True, color=BLACK)

    # Строка метрик
    meta_parts: list[str] = []
    if item.get("accuracy") is not None:
        meta_parts.append(f"Точность: {_fmt_pct(item['accuracy'])}")
    if item.get("cosine_similarity") is not None:
        meta_parts.append(
            f"Сходство: {_fmt_float(item['cosine_similarity'], 2)}"
        )
    if item.get("cosine_distance") is not None:
        meta_parts.append(
            f"Дистанция: {_fmt_float(item['cosine_distance'], 3)}"
        )
    if item.get("seconds") is not None:
        meta_parts.append(f"Время: {_fmt_sec(item['seconds'])}")
    if "relevant_sources_count" in item:
        rel = item.get("relevant_sources_count") or 0
        raw = item.get("raw_sources_count") or 0
        meta_parts.append(f"Источников: {rel}/{raw}")
    if meta_parts:
        p = doc.add_paragraph()
        run = p.add_run(" | ".join(meta_parts))
        set_run_font(run, size=10, italic=True, color=GRAY)

    # Ошибка
    if item.get("error"):
        p = doc.add_paragraph()
        run = p.add_run(f"Ошибка: {clean_text(item['error'])}")
        set_run_font(run, size=11, bold=True, color=RED)

    # Эталон
    reference = clean_text(item.get("reference"))
    if reference:
        _add_labeled_paragraph(
            doc, "Эталонный ответ", reference, text_italic=True
        )

    # Ответ модели
    answer = clean_text(item.get("answer"))
    if answer:
        _add_labeled_paragraph(doc, "Ответ модели", answer)

    # Источники
    if include_sources:
        sources = item.get("sources") or []
        if sources:
            p = doc.add_paragraph()
            run = p.add_run("Источники:")
            set_run_font(run, size=11, bold=True)
            for idx, src in enumerate(sources, 1):
                src_text = truncate(clean_text(src), source_max_chars)
                bullet = doc.add_paragraph(style="List Bullet")
                run = bullet.add_run(f"[{idx}] {src_text}")
                set_run_font(run, size=10)


def _add_questions(
    doc: Document,
    results: list[dict],
    *,
    source_max_chars: int,
    include_sources: bool,
    page_break_per_item: bool,
) -> None:
    for index, item in enumerate(results):
        if page_break_per_item and index > 0:
            doc.add_page_break()
        _add_question(
            doc,
            item,
            source_max_chars=source_max_chars,
            include_sources=include_sources,
        )


# ---------- Публичный API ----------

def build_report(
    data: dict,
    *,
    source_max_chars: int = DEFAULT_SOURCE_MAX_CHARS,
    include_sources: bool = True,
    page_break_per_item: bool = False,
) -> Document:
    doc = Document()
    _setup_document(doc)

    doc.core_properties.title = "Отчёт по бенчмарку"
    doc.core_properties.subject = f"Модель: {data.get('model', '—')}"
    doc.core_properties.author = "benchmark_to_docx.py"

    _add_title(doc, data)
    _add_summary(doc, data)
    _add_questions(
        doc,
        data.get("results", []),
        source_max_chars=source_max_chars,
        include_sources=include_sources,
        page_break_per_item=page_break_per_item,
    )
    return doc


# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", type=Path, default=Path("benchmark.json"),
                        help="Путь к JSON (по умолчанию: benchmark.json)")
    parser.add_argument("-o", "--output", type=Path, default=Path("output.docx"),
                        help="Путь к DOCX (по умолчанию: output.docx)")
    parser.add_argument("--source-max-chars", type=int,
                        default=DEFAULT_SOURCE_MAX_CHARS,
                        help="Обрезать каждый источник до N символов (0 — не обрезать)")
    parser.add_argument("--no-sources", action="store_true",
                        help="Не включать блок источников")
    parser.add_argument("--page-break-per-item", action="store_true",
                        help="Каждый вопрос с новой страницы")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        raw = args.input.read_text(encoding="utf-8")
    except FileNotFoundError:
        sys.exit(f"Файл не найден: {args.input}")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        sys.exit(f"Ошибка разбора JSON ({args.input}): {exc}")

    doc = build_report(
        data,
        source_max_chars=args.source_max_chars,
        include_sources=not args.no_sources,
        page_break_per_item=args.page_break_per_item,
    )
    doc.save(args.output)

    total = len(data.get("results", []))
    print(f"Готово: {args.output} ({total} вопросов)")


if __name__ == "__main__":
    main()