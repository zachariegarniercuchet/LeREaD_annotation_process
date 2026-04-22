import re

def performance_per_label(txt: str):

    def parse_line(line):
        match = re.match(r"(.+?)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)%", line.strip())
        if not match:
            return None
        label   = match.group(1).strip()
        A1      = int(match.group(2))
        A2      = int(match.group(3))
        matched = int(match.group(4))
        return label, A1, A2, matched

    def compute_metrics(A1, A2, matched):
        precision = matched / A2 if A2 > 0 else 0
        recall    = matched / A1 if A1 > 0 else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
        return precision, recall, f1

    def build_table(rows, tab_label):
        lines = []
        total_A1 = total_A2 = total_matched = 0

        for label, A1, A2, matched in rows:
            p, r, f1 = compute_metrics(A1, A2, matched)
            total_A1      += A1
            total_A2      += A2
            total_matched += matched
            lines.append(f"{label} & {A1} & {A2} & {matched} & {p:.2f} & {r:.2f} & {f1:.2f} \\\\")

        p, r, f1 = compute_metrics(total_A1, total_A2, total_matched)
        lines.append("\\midrule")
        lines.append(
            f"\\textbf{{TOTAL}} & \\textbf{{{total_A1}}} & \\textbf{{{total_A2}}} & "
            f"\\textbf{{{total_matched}}} & \\textbf{{{p:.2f}}} & \\textbf{{{r:.2f}}} & \\textbf{{{f1:.2f}}} \\\\"
        )

        return (
            "\\begin{table}[h]\n"
            "\\centering\n"
            "\\small\n"
            "\\begin{tabular}{lrrr|rrr}\n"
            "\\toprule\n"
            "Mention & H & LLM & EM & Precision & Recall & F1 \\\\\n"
            "\\midrule\n"
            + "\n".join(lines) + "\n"
            "\\bottomrule\n"
            "\\end{tabular}\n"
            f"\\label{{tab:{tab_label}}}\n"
            "\\end{table}"
        )

    # --- Parse all rows ---
    rows = [r for line in txt.split("\n") if (r := parse_line(line))]

    # --- Split into two groups ---
    TOP_LEVEL = {"decision", "legislation", "secondary sources"}
    parent_rows = [r for r in rows if r[0].lower() in TOP_LEVEL]
    sub_rows    = [r for r in rows if r[0].lower() not in TOP_LEVEL]

    return {
        "parent_labels": build_table(parent_rows, "parent_labels_results"),
        "sub_labels":    build_table(sub_rows,    "sub_labels_results"),
    }

import re

def performance_per_document(
    txt: str,
    tab_label="document_results",
    caption="Per-document evaluation results.",
):
    """
    Parses a per-document result table and outputs a LaTeX table.

    Input columns: File, Human, LLM, Match, Precision, Recall, L1 F1, L2 F1
    Output columns: Document, Type, Jur., Human, LLM, Match, P, R, F1

    Precision, Recall, F1 are recomputed from Match.
    """

    def parse_line(line):
        match = re.match(
            r"(\S+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%",
            line.strip()
        )
        if not match:
            return None

        filename  = match.group(1)
        human     = int(match.group(2))
        llm       = int(match.group(3))
        match_cnt = int(match.group(4))

        return filename, human, llm, match_cnt

    def compute_f1(p, r):
        return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    # --- Parse ---
    rows = [r for line in txt.split("\n") if (r := parse_line(line))]

    # --- Build body lines ---
    body_lines = []
    total_human = total_llm = total_match = 0

    for filename, human, llm, match_count in rows:
        p = match_count / llm   if llm   > 0 else 0.0
        r = match_count / human if human > 0 else 0.0
        f1 = compute_f1(p, r)

        total_human += human
        total_llm   += llm
        total_match += match_count

        body_lines.append(
            f"{filename} & {human} & {llm} & {match_count} "
            f"& {p:.2f} & {r:.2f} & {f1:.2f} \\\\"
        )

    # --- Total row ---
    p_tot  = total_match / total_llm   if total_llm   > 0 else 0.0
    r_tot  = total_match / total_human if total_human > 0 else 0.0
    f1_tot = compute_f1(p_tot, r_tot)

    body_lines.append("\\midrule")
    body_lines.append(
        f"\\textbf{{Total}} & "
        f"\\textbf{{{total_human}}} & \\textbf{{{total_llm}}} & \\textbf{{{total_match}}} & "
        f"\\textbf{{{p_tot:.2f}}} & \\textbf{{{r_tot:.2f}}} & \\textbf{{{f1_tot:.2f}}} \\\\"
    )

    return (
        "\\begin{table}\n"
        "\\centering\n"
        "\\small\n"
        "\\begin{tabular}{lccrrrrrr}\n"
        "\\toprule\n"
        "Document & Human & LLM & Match & P & R & F1 \\\\\n"
        "\\midrule\n"
        + "\n".join(body_lines) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{tab:{tab_label}}}\n"
        "\\end{table}"
    )


    

    


if __name__ == "__main__":

    latex_tables = "label" #label or document
    data = """
Label                            Total A1   Total A2    Matched    Precision       Recall      Mean F1
──────────────────────────────────────────────────────────────────────────────────────────────────────────────
authors                               222        206        164       79.61%       73.87%       77.21%
citation                             1371       1364       1288       94.43%       93.95%       86.30%
decision                             1574       1509       1267       83.96%       80.50%       76.19%
fragment                             1924       1714       1577       92.01%       81.96%       78.86%
legislation                          1095       1061        930       87.65%       84.93%       81.43%
secondary sources                     269        297        211       71.04%       78.44%       66.95%
source                                168        191        143       74.87%       85.12%       72.24%
title                                2202       2102       1851       88.06%       84.06%       80.96%
unable to classify                      5          0          0        0.00%        0.00%        0.00%


"""

    if latex_tables == "label":
        tables = performance_per_label(data)
        for name, latex in tables.items():
            print(f"% ── {name} ──────────────────────────")
            print(latex)
            print()

    if latex_tables == "document":

        print(performance_per_document(data))