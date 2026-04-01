from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import fmean, median, stdev
from xml.etree import ElementTree as ET
import zipfile

NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
HEADER_PATTERN = re.compile(r"^var2_(?P<n>\d+)_(?P<method>.+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize the bivariate VAR(2) L2 benchmark spreadsheet and emit "
            "LaTeX-ready and JSON summaries."
        )
    )
    here = Path(__file__).resolve().parent
    parser.add_argument(
        "--xlsx",
        type=Path,
        default=here / "var2_L2_errors_VI_VNPC_P_SPLINE.xlsx",
        help="Path to the benchmark .xlsx file.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path for the JSON summary.",
    )
    parser.add_argument(
        "--latex-output",
        type=Path,
        default=None,
        help="Optional path for the LaTeX table snippet.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print the JSON summary to stdout.",
    )
    parser.add_argument(
        "--print-latex",
        action="store_true",
        help="Print the LaTeX table snippet to stdout.",
    )
    return parser.parse_args()


def load_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    values: list[str] = []
    for item in root.findall("a:si", NS):
        text = "".join(node.text or "" for node in item.iterfind(".//a:t", NS))
        values.append(text)
    return values


def excel_col_to_index(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    value = 0
    for char in letters:
        value = value * 26 + (ord(char.upper()) - ord("A") + 1)
    return value - 1


def load_first_sheet_rows(xlsx_path: Path) -> list[list[str | float | None]]:
    with zipfile.ZipFile(xlsx_path) as archive:
        shared_strings = load_shared_strings(archive)
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))

    rows: list[list[str | float | None]] = []
    for row in sheet.findall(".//a:sheetData/a:row", NS):
        current: list[str | float | None] = []
        for cell in row.findall("a:c", NS):
            ref = cell.attrib.get("r")
            if ref:
                idx = excel_col_to_index(ref)
                if idx >= len(current):
                    current.extend([None] * (idx - len(current) + 1))
            else:
                idx = len(current)

            cell_type = cell.attrib.get("t")
            value = cell.find("a:v", NS)
            if value is None:
                current[idx] = None
            elif cell_type == "s":
                current[idx] = shared_strings[int(value.text)]
            else:
                current[idx] = float(value.text)
        rows.append(current)
    return rows


def column_summary(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": fmean(values),
        "std": stdev(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def build_summary(
    rows: list[list[str | float | None]],
    *,
    source: Path,
) -> dict[str, object]:
    headers = [str(value) for value in rows[0]]
    data_rows = rows[1:]
    grouped: dict[int, dict[str, dict[str, float | int]]] = {}

    for index, header in enumerate(headers):
        values = [
            float(row[index])
            for row in data_rows
            if index < len(row) and row[index] is not None
        ]
        match = HEADER_PATTERN.match(header)
        if match is None:
            raise ValueError(f"Unexpected spreadsheet header: {header}")

        n_value = int(match.group("n"))
        method = match.group("method")
        grouped.setdefault(n_value, {})[method] = column_summary(values)

    comparisons: dict[int, dict[str, float]] = {}
    for n_value, metrics in grouped.items():
        p_mean = float(metrics["p_spline"]["mean"])
        vb_mean = float(metrics["VB"]["mean"])
        vnpc_mean = float(metrics["VNPC"]["mean"])
        comparisons[n_value] = {
            "relative_gain_vs_vb": (vb_mean - p_mean) / vb_mean,
            "relative_gain_vs_vnpc": (vnpc_mean - p_mean) / vnpc_mean,
        }

    return {
        "source": str(source),
        "grouped": grouped,
        "comparisons": comparisons,
    }


def render_latex_table(summary: dict[str, object]) -> str:
    grouped = summary["grouped"]
    lines = [
        r"\begin{table}[t]",
        r"    \centering",
        r"    \caption{Bivariate VAR(2) benchmark using the $L_{2}$ error metric from",
        r"    \citet{Liu2023}. Entries are mean $\pm$ standard deviation over 500",
        r"    independent realisations; smaller values are better.}",
        r"    \label{tab:var2_l2_results}",
        r"    \begin{NiceTabular}{lccc}",
        r"        \toprule",
        r"        {$n$} & {P-spline} & {VB} & {VNPC} \\",
        r"        \midrule",
    ]
    for n_value in sorted(grouped):
        row = grouped[n_value]
        lines.append(
            "        "
            f"{n_value} & "
            f"{row['p_spline']['mean']:.3f} \\pm {row['p_spline']['std']:.3f} & "
            f"{row['VB']['mean']:.3f} \\pm {row['VB']['std']:.3f} & "
            f"{row['VNPC']['mean']:.3f} \\pm {row['VNPC']['std']:.3f} \\\\"
        )
    lines.extend(
        [
            r"        \bottomrule",
            r"    \end{NiceTabular}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def main(args: argparse.Namespace) -> None:
    rows = load_first_sheet_rows(args.xlsx)
    summary = build_summary(rows, source=args.xlsx)
    latex_table = render_latex_table(summary)

    if args.json_output is not None:
        args.json_output.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.latex_output is not None:
        args.latex_output.write_text(latex_table + "\n", encoding="utf-8")

    if args.print_json or (
        args.json_output is None and args.latex_output is None and not args.print_latex
    ):
        print(json.dumps(summary, indent=2, sort_keys=True))
    if args.print_latex or (args.json_output is None and args.latex_output is None):
        print()
        print(latex_table)


if __name__ == "__main__":
    parsed_args = parse_args()
    main(parsed_args)
