#!/usr/bin/env python3
"""
generate_latex_summary.py

For each specified folder, reads evaluation_results.txt,
parses the SUMMARY TABLE (per-document) and MEAN PER-LABEL BREAKDOWN (per-label)
sections, recomputes precision/recall/F1 from raw counts, and writes a
ready-to-paste LaTeX page (results_summary.tex) into that folder.

Usage:
    python generate_latex_summary.py
"""

import re
import os

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

llm_results_folder = r"C:\Users\zakga\OneDrive\Documents\code\LeREaD_annotation_process\data\Documents_Annotés\llm"

FOLDERS = [
    fr"{llm_results_folder}\p2_c500_fsselected-30_mgpt-5.2",
    fr"{llm_results_folder}\p2_c500_fsselected-0_mgpt-5.2",
    fr"{llm_results_folder}\TEST_CONTEXT_p2_c500_fsselected-30_mgpt-5.2",
    fr"{llm_results_folder}\PARACHUNKER_p2_c500_fsselected-30_mgpt-5.2",
    fr"{llm_results_folder}\PARACHUNKER_CONTEXT_p2_c500_fsselected-30_mgpt-5.2",
    fr"{llm_results_folder}\PARACHUNKER_ALLINONE_p2_c500_fsselected-30_mgpt-5.2",
    fr"{llm_results_folder}\PARACHUNKER_ALLINONE_p2_c500_fspattern-30_mgpt-5.2",
    fr"{llm_results_folder}\PARACHUNKER_ALLINONE_p2_c500_fsrandom-30_mgpt-5.2",
    fr"{llm_results_folder}\PARACHUNKER_ALLINONE_p2_c500_fsrandom-15_mgpt-5.2",
    fr"{llm_results_folder}\PARACHUNKER_ALLINONE_p2_c500_fspattern-15_mgpt-5.2",
    fr"{llm_results_folder}\PARACHUNKER_ALLINONE_p2_c500_fspattern-5_mgpt-5.2",
    fr"{llm_results_folder}\PARACHUNKER_ALLINONE_p2_c500_fspattern-5-25_mgpt-5.2",
    fr"{llm_results_folder}\F_PARACHUNKER_ALLINONE_p2_c500_fspattern-5_mgpt-5.2",
    fr"{llm_results_folder}\F_PARACHUNKER_ALLINONE_p2_c500_fspattern-5-25_mgpt-5.2",
    fr"{llm_results_folder}\F+_PARACHUNKER_ALLINONE_p2_c500_fspattern-5-25_mgpt-5.2"


    

]

# Labels classified as "parent" (top-level) vs "sub"
PARENT_LABELS = {"decision", "legislation", "secondary sources"}

# Short document name extraction: keep the case identifier up to first _annotated / _LLM / end
_DOC_RE = re.compile(r"^(\d{4}[A-Za-z0-9]+)")


# ──────────────────────────────────────────────────────────────────────────────
# Parsing helpers
# ──────────────────────────────────────────────────────────────────────────────

def extract_section_text(full_text: str, header_keyword: str) -> str:
    """
    Returns the block of lines that follow a section whose header line
    contains `header_keyword`.  Stops at the next ={10,} separator.
    """
    lines = full_text.splitlines()
    sep = re.compile(r"^={10,}$")
    found_header = False
    past_closing_sep = False
    collected = []

    for line in lines:
        stripped = line.strip()

        if not found_header:
            if header_keyword.lower() in stripped.lower():
                found_header = True
            continue

        if found_header and not past_closing_sep:
            # Skip the closing === of the section header
            if sep.match(stripped):
                past_closing_sep = True
            continue

        # We are now inside the section body
        if sep.match(stripped):
            break  # Next section starts
        collected.append(line)

    return "\n".join(collected)


def parse_doc_table(text: str):
    """
    Parses lines like:
      1989CanLII1415CITT_annotated_GL_revRL_te   237   206   159   67.09%  77.18%  71.78%  38.97%
    Returns list of dicts: {filename, human, llm, match}
    """
    pattern = re.compile(
        r"^(\S+)\s+(\d+)\s+(\d+)\s+(\d+)\s+[\d.]+%\s+[\d.]+%\s+[\d.]+%\s+[\d.]+%"
    )
    rows = []
    for line in text.splitlines():
        m = pattern.match(line.strip())
        if m:
            rows.append({
                "filename": m.group(1),
                "human":    int(m.group(2)),
                "llm":      int(m.group(3)),
                "match":    int(m.group(4)),
            })
    return rows


def parse_label_table(text: str):
    """
    Parses lines like:
      authors    222   206   164   79.61%   73.87%   77.21%
    Returns list of dicts: {label, a1, a2, matched}
    """
    pattern = re.compile(
        r"^(.+?)\s+(\d+)\s+(\d+)\s+(\d+)\s+[\d.]+%\s+[\d.]+%\s+[\d.]+%"
    )
    rows = []
    for line in text.splitlines():
        m = pattern.match(line.strip())
        if m:
            rows.append({
                "label":   m.group(1).strip(),
                "a1":      int(m.group(2)),   # Human (reference)
                "a2":      int(m.group(3)),   # LLM
                "matched": int(m.group(4)),
            })
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Metric computation
# ──────────────────────────────────────────────────────────────────────────────

def metrics(human: int, llm: int, matched: int):
    p  = matched / llm   if llm   > 0 else 0.0
    r  = matched / human if human > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


# ──────────────────────────────────────────────────────────────────────────────
# Short document name
# ──────────────────────────────────────────────────────────────────────────────

def short_doc_name(filename: str) -> str:
    """
    Extract a readable short name from the filename stem.
    e.g. '1989CanLII1415CITT_annotated_GL_revRL_te' → '1989CanLII1415CITT'
    """
    # Remove trailing truncation artifact (no extension, truncated at 40 chars)
    base = filename.split("_annotated")[0].split("_LLMv")[0].split("_v1")[0]
    # Also strip trailing underscores
    return base.rstrip("_")


# ──────────────────────────────────────────────────────────────────────────────
# LaTeX builders
# ──────────────────────────────────────────────────────────────────────────────

def build_label_table(label_rows: list, caption: str) -> str:
    """
    Builds a single LaTeX table with:
      - parent labels (decision, legislation, secondary sources) + their TOTAL
      - sub labels (everything else) + their TOTAL
    """
    parent_rows = [r for r in label_rows if r["label"].lower() in PARENT_LABELS]
    sub_rows    = [r for r in label_rows if r["label"].lower() not in PARENT_LABELS]

    def fmt_rows(rows):
        lines = []
        tot_h = tot_l = tot_m = 0
        for r in rows:
            p, rc, f1 = metrics(r["a1"], r["a2"], r["matched"])
            tot_h += r["a1"]; tot_l += r["a2"]; tot_m += r["matched"]
            lines.append(
                f"{r['label']:<25} & {r['a1']:>5} & {r['a2']:>5} & {r['matched']:>5}"
                f" & {p:.2f} & {rc:.2f} & {f1:.2f} \\\\"
            )
        p, rc, f1 = metrics(tot_h, tot_l, tot_m)
        lines.append("\\midrule")
        lines.append(
            f"\\rowcolor{{totalrow}}\n"
            f"\\textbf{{TOTAL}} & \\textbf{{{tot_h}}} & \\textbf{{{tot_l}}} & "
            f"\\textbf{{{tot_m}}} & \\textbf{{{p:.2f}}} & \\textbf{{{rc:.2f}}} & "
            f"\\textbf{{{f1:.2f}}} \\\\"
        )
        return "\n".join(lines)

    body = fmt_rows(parent_rows) + "\n\\midrule\n" + fmt_rows(sub_rows)

    return (
        "\\begin{table}[h]\n"
        "\\centering\n"
        #f"\\caption{{{caption}}}\n"
        "\\begin{tabular}{lrrr|rrr}\n"
        "\\toprule\n"
        "\\textbf{Mention} & \\textbf{H} & \\textbf{LLM} & \\textbf{EM} "
        "& \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1} \\\\\n"
        "\\midrule\n"
        + body + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}"
    )


def build_doc_table(doc_rows: list, caption: str) -> str:
    """
    Builds a per-document LaTeX table with a TOTAL row.
    """
    lines = []
    tot_h = tot_l = tot_m = 0

    for r in doc_rows:
        name = short_doc_name(r["filename"])
        p, rc, f1 = metrics(r["human"], r["llm"], r["match"])
        tot_h += r["human"]; tot_l += r["llm"]; tot_m += r["match"]
        lines.append(
            f"{name:<30} & {r['human']:>5} & {r['llm']:>5} & {r['match']:>5}"
            f" & {p:.2f} & {rc:.2f} & {f1:.2f} \\\\"
        )

    p, rc, f1 = metrics(tot_h, tot_l, tot_m)
    lines.append("\\midrule")
    lines.append(
        f"\\rowcolor{{totalrow}}\n"
        f"\\textbf{{Total}} & \\textbf{{{tot_h}}} & \\textbf{{{tot_l}}} & "
        f"\\textbf{{{tot_m}}} & \\textbf{{{p:.2f}}} & \\textbf{{{rc:.2f}}} & "
        f"\\textbf{{{f1:.2f}}} \\\\"
    )

    return (
        "\\begin{table}[h]\n"
        "\\centering\n"
        #f"\\caption{{{caption}}}\n"
        "\\begin{tabular}{lrrr rrr}\n"
        "\\toprule\n"
        "\\textbf{Document} & \\textbf{Human} & \\textbf{LLM} & \\textbf{Match} "
        "& \\textbf{P} & \\textbf{R} & \\textbf{F1} \\\\\n"
        "\\midrule\n"
        + "\n".join(lines) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Full LaTeX page
# ──────────────────────────────────────────────────────────────────────────────

LATEX_PREAMBLE = r"""\documentclass[12pt,a4paper]{article}
\usepackage[margin=2cm]{geometry}
\usepackage{booktabs}
\usepackage{colortbl}
\usepackage[table]{xcolor}
\usepackage{caption}

% Row color for totals
\definecolor{totalrow}{gray}{0.88}

\begin{document}
"""

LATEX_POSTAMBLE = "\n\\end{document}\n"


def build_latex_page(folder_name: str, doc_rows: list, label_rows: list) -> str:
    section_title = folder_name.replace("_", r"\_")
    label_caption = f"Per-mention-type evaluation results — {folder_name}."
    doc_caption   = f"Per-document evaluation results — {folder_name}."

    label_table = build_label_table(label_rows, label_caption)
    doc_table   = build_doc_table(doc_rows, doc_caption)

    page = (
        "\\newpage\n"
        + "%" * 50 + "\n"
        + f"\\section{{{section_title}}}\n"
        + "%" * 50 + "\n\n"
        + "\\subsection{Results by Mention Type}\n"
        + label_table + "\n\n"
        + "\\subsection{Results by Document}\n"
        + doc_table + "\n"
        #+ LATEX_POSTAMBLE
    )
    return page


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def process_folder(folder: str):
    eval_path = os.path.join(folder, "evaluation_results.txt")
    out_path  = os.path.join(folder, "results_summary.tex")

    if not os.path.isfile(eval_path):
        print(f"  [SKIP] Not found: {eval_path}")
        return

    with open(eval_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Extract sections
    doc_text   = extract_section_text(content, "SUMMARY TABLE")
    label_text = extract_section_text(content, "MEAN PER-LABEL BREAKDOWN")

    if not doc_text.strip():
        print(f"  [WARN] SUMMARY TABLE not found in: {eval_path}")
        return
    if not label_text.strip():
        print(f"  [WARN] MEAN PER-LABEL BREAKDOWN not found in: {eval_path}")
        return

    doc_rows   = parse_doc_table(doc_text)
    label_rows = parse_label_table(label_text)

    if not doc_rows:
        print(f"  [WARN] No document rows parsed from: {eval_path}")
        return
    if not label_rows:
        print(f"  [WARN] No label rows parsed from: {eval_path}")
        return

    folder_name = os.path.basename(folder)
    latex = build_latex_page(folder_name, doc_rows, label_rows)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(latex)

    print(f"  [OK]  Written: {out_path}")
    print(f"        Docs parsed : {len(doc_rows)}")
    print(f"        Labels parsed: {len(label_rows)}")


def main():
    for folder in FOLDERS:
        print(f"\nProcessing: {os.path.basename(folder)}")
        if not os.path.isdir(folder):
            print(f"  [SKIP] Folder does not exist: {folder}")
            continue
        process_folder(folder)

    print("\nDone — copy the generated results_summary.tex files into Overleaf.")


if __name__ == "__main__":
    main()