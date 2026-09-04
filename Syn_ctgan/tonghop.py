import argparse
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "tonghop" / "per_attack_results.csv"
DEFAULT_OUTPUT = BASE_DIR / "tonghop" / "per_attack_results_mean_std.xlsx"

ID_COLUMNS = ["Dataset", "Seed", "Attack", "Model", "Samples"]
PREFERRED_METRICS = ["AUC", "F1", "MCC", "PR-AUC", "TPR@FAR=1%", "TPR@FAR=10%"]
PREFERRED_MODELS = ["DIF", "VAE", "AutoEncoder", "DeepSVDD", "CLAN", "SSAE"]
MODEL_DISPLAY_NAMES = {
    "AutoEncoder": "AE",
    "SSAE": "SSA-PAA",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create Excel mean ± std tables from per_attack_results.csv."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help=f"Input CSV path. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Output XLSX path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=2,
        help="Number of decimal places for mean and std. Default: 2",
    )
    return parser.parse_args()


def ordered_unique(series):
    return list(dict.fromkeys(series.dropna().astype(str)))


def metric_columns(df):
    preferred = [col for col in PREFERRED_METRICS if col in df.columns]
    numeric_extra = [
        col
        for col in df.columns
        if col not in ID_COLUMNS
        and col not in preferred
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    return preferred + numeric_extra


def model_columns(df):
    models = ordered_unique(df["Model"])
    ordered = [model for model in PREFERRED_MODELS if model in models]
    ordered.extend([model for model in models if model not in ordered])
    return ordered


def display_model_name(model):
    return MODEL_DISPLAY_NAMES.get(model, model)


def display_dataset_name(dataset):
    return str(dataset).replace("_", "-").upper()


def format_mean_std(mean_value, std_value, digits):
    if pd.isna(mean_value):
        return ""
    if pd.isna(std_value):
        std_value = 0.0
    return f"{float(mean_value):.{digits}f} ± {float(std_value):.{digits}f}"


def sample_text(group, digits):
    samples = pd.to_numeric(group["Samples"], errors="coerce").dropna()
    if samples.empty:
        return ""

    unique_samples = sorted(samples.unique())
    if len(unique_samples) == 1:
        value = float(unique_samples[0])
        return str(int(value)) if value.is_integer() else f"{value:.{digits}f}"

    return format_mean_std(samples.mean(), samples.std(ddof=1), digits)


def get_stat(agg, key, model, metric, stat):
    try:
        return agg.loc[key + (model,), (metric, stat)]
    except KeyError:
        return None


def build_by_attack(df, metrics, models, digits):
    agg = df.groupby(["Dataset", "Attack", "Model"], sort=False)[metrics].agg(["mean", "std"])
    attack_rows = df[["Dataset", "Attack"]].drop_duplicates()
    rows = []

    for _, attack_row in attack_rows.iterrows():
        dataset = str(attack_row["Dataset"])
        attack = str(attack_row["Attack"])
        attack_group = df[
            (df["Dataset"].astype(str) == dataset)
            & (df["Attack"].astype(str) == attack)
        ]
        samples = sample_text(attack_group, digits)

        row = {
            "Dataset": display_dataset_name(dataset),
            "Samples": samples,
            "Attack": attack,
        }
        for metric in metrics:
            for model in models:
                mean_value = get_stat(agg, (dataset, attack), model, metric, "mean")
                std_value = get_stat(agg, (dataset, attack), model, metric, "std")
                column_name = f"{metric}_{display_model_name(model)}"
                row[column_name] = format_mean_std(mean_value, std_value, digits)
        rows.append(row)

    return pd.DataFrame(rows)


def build_by_dataset(df, metrics, models, digits):
    agg = df.groupby(["Dataset", "Model"], sort=False)[metrics].agg(["mean", "std"])
    datasets = ordered_unique(df["Dataset"])
    rows = []

    for dataset in datasets:
        for metric in metrics:
            row = {
                "Dataset": display_dataset_name(dataset),
                "Metric": metric,
            }
            for model in models:
                mean_value = get_stat(agg, (dataset,), model, metric, "mean")
                std_value = get_stat(agg, (dataset,), model, metric, "std")
                row[display_model_name(model)] = format_mean_std(mean_value, std_value, digits)
            rows.append(row)

    return pd.DataFrame(rows)


def parse_mean(value):
    if not isinstance(value, str) or "±" not in value:
        return None
    try:
        return float(value.split("±", 1)[0].strip())
    except ValueError:
        return None


def col_name(col_idx):
    name = ""
    while col_idx:
        col_idx, remainder = divmod(col_idx - 1, 26)
        name = chr(65 + remainder) + name
    return name


def cell_ref(row_idx, col_idx):
    return f"{col_name(col_idx)}{row_idx}"


def xml_text(value):
    return escape(str(value), {"'": "&apos;", '"': "&quot;"})


def dataframe_to_rows(df):
    rows = [list(df.columns)]
    rows.extend(df.fillna("").astype(str).values.tolist())
    return rows


def merge_repeated_ranges(rows, column, start_row=2):
    ranges = []
    row = start_row
    max_row = len(rows)

    while row <= max_row:
        value = rows[row - 1][column - 1]
        end_row = row
        while end_row + 1 <= max_row and rows[end_row][column - 1] == value:
            end_row += 1
        if value not in (None, "") and end_row > row:
            ranges.append((row, column, end_row, column))
            for clear_row in range(row + 1, end_row + 1):
                rows[clear_row - 1][column - 1] = ""
        row = end_row + 1

    return ranges


def best_cells_by_row(rows, first_model_col, compare_group_size=None, start_row=2):
    best_cells = set()
    for row_idx in range(start_row, len(rows) + 1):
        last_col = len(rows[row_idx - 1])
        group_size = compare_group_size or (last_col - first_model_col + 1)
        for group_start in range(first_model_col, last_col + 1, group_size):
            group_end = min(group_start + group_size - 1, last_col)
            means = []
            for col_idx in range(group_start, group_end + 1):
                mean_value = parse_mean(rows[row_idx - 1][col_idx - 1])
                if mean_value is not None:
                    means.append(mean_value)
            if not means:
                continue

            best = max(means)
            for col_idx in range(group_start, group_end + 1):
                mean_value = parse_mean(rows[row_idx - 1][col_idx - 1])
                if mean_value is not None and abs(mean_value - best) < 1e-12:
                    best_cells.add((row_idx, col_idx))

    return best_cells


def column_widths(rows):
    widths = []
    for col_idx in range(1, len(rows[0]) + 1):
        max_len = max(len(str(row[col_idx - 1])) for row in rows)
        widths.append(min(max(max_len + 2, 11), 28))
    return widths


def build_sheet_xml(rows, first_model_col, merge_columns, compare_group_size=None):
    rows = [list(row) for row in rows]
    merge_ranges = []
    for column in merge_columns:
        merge_ranges.extend(merge_repeated_ranges(rows, column, start_row=2))

    best_cells = best_cells_by_row(
        rows,
        first_model_col=first_model_col,
        compare_group_size=compare_group_size,
        start_row=2,
    )
    widths = column_widths(rows)
    max_row = len(rows)
    max_col = len(rows[0])
    dimension = f"A1:{cell_ref(max_row, max_col)}"

    col_xml = "".join(
        f'<col min="{idx}" max="{idx}" width="{width}" customWidth="1"/>'
        for idx, width in enumerate(widths, start=1)
    )

    row_xml_parts = []
    for row_idx, row in enumerate(rows, start=1):
        cell_xml_parts = []
        for col_idx, value in enumerate(row, start=1):
            if value == "":
                continue
            style_id = 1 if row_idx == 1 else 2
            if (row_idx, col_idx) in best_cells:
                style_id = 3
            ref = cell_ref(row_idx, col_idx)
            cell_xml_parts.append(
                f'<c r="{ref}" s="{style_id}" t="inlineStr"><is><t>{xml_text(value)}</t></is></c>'
            )
        row_xml_parts.append(f'<row r="{row_idx}">{"".join(cell_xml_parts)}</row>')

    merge_xml = ""
    if merge_ranges:
        items = [
            f'<mergeCell ref="{cell_ref(sr, sc)}:{cell_ref(er, ec)}"/>'
            for sr, sc, er, ec in merge_ranges
        ]
        merge_xml = f'<mergeCells count="{len(items)}">{"".join(items)}</mergeCells>'

    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<dimension ref="{dimension}"/>'
        '<sheetViews><sheetView workbookViewId="0">'
        '<pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>'
        '</sheetView></sheetViews>'
        f'<cols>{col_xml}</cols>'
        f'<sheetData>{"".join(row_xml_parts)}</sheetData>'
        f'{merge_xml}'
        '</worksheet>'
    )


def styles_xml():
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="2">'
        '<font><sz val="11"/><color theme="1"/><name val="Calibri"/><family val="2"/></font>'
        '<font><b/><sz val="11"/><color theme="1"/><name val="Calibri"/><family val="2"/></font>'
        '</fonts>'
        '<fills count="3">'
        '<fill><patternFill patternType="none"/></fill>'
        '<fill><patternFill patternType="gray125"/></fill>'
        '<fill><patternFill patternType="solid"><fgColor rgb="FFD9D9D9"/><bgColor indexed="64"/></patternFill></fill>'
        '</fills>'
        '<borders count="2">'
        '<border><left/><right/><top/><bottom/><diagonal/></border>'
        '<border><left style="thin"><color rgb="FF000000"/></left>'
        '<right style="thin"><color rgb="FF000000"/></right>'
        '<top style="thin"><color rgb="FF000000"/></top>'
        '<bottom style="thin"><color rgb="FF000000"/></bottom>'
        '<diagonal/></border>'
        '</borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="4">'
        '<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
        '<xf numFmtId="0" fontId="1" fillId="2" borderId="1" xfId="0" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1">'
        '<alignment horizontal="center" vertical="center"/></xf>'
        '<xf numFmtId="0" fontId="0" fillId="0" borderId="1" xfId="0" applyBorder="1" applyAlignment="1">'
        '<alignment horizontal="center" vertical="center"/></xf>'
        '<xf numFmtId="0" fontId="1" fillId="0" borderId="1" xfId="0" applyFont="1" applyBorder="1" applyAlignment="1">'
        '<alignment horizontal="center" vertical="center"/></xf>'
        '</cellXfs>'
        '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
        '<dxfs count="0"/>'
        '<tableStyles count="0" defaultTableStyle="TableStyleMedium2" defaultPivotStyle="PivotStyleLight16"/>'
        '</styleSheet>'
    )


def write_xlsx(output_path, sheets):
    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets>'
        + "".join(
            f'<sheet name="{xml_text(name)}" sheetId="{idx}" r:id="rId{idx}"/>'
            for idx, name in enumerate(sheets.keys(), start=1)
        )
        + '</sheets></workbook>'
    )

    workbook_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(
            f'<Relationship Id="rId{idx}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{idx}.xml"/>'
            for idx in range(1, len(sheets) + 1)
        )
        + f'<Relationship Id="rId{len(sheets) + 1}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
        '</Relationships>'
    )

    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        + "".join(
            f'<Override PartName="/xl/worksheets/sheet{idx}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            for idx in range(1, len(sheets) + 1)
        )
        + '</Types>'
    )

    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        '</Relationships>'
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", root_rels)
        zf.writestr("xl/workbook.xml", workbook_xml)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        zf.writestr("xl/styles.xml", styles_xml())
        for idx, sheet_xml in enumerate(sheets.values(), start=1):
            zf.writestr(f"xl/worksheets/sheet{idx}.xml", sheet_xml)


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    missing = [col for col in ID_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    metrics = metric_columns(df)
    if not metrics:
        raise ValueError("No metric columns found.")

    for metric in metrics:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")

    models = model_columns(df)
    by_dataset = build_by_dataset(df, metrics, models, args.digits)
    by_attack = build_by_attack(df, metrics, models, args.digits)

    sheets = {
        "ByAttack": build_sheet_xml(
            dataframe_to_rows(by_attack),
            first_model_col=4,
            merge_columns=[],
            compare_group_size=len(models),
        ),
        "ByDataset": build_sheet_xml(
            dataframe_to_rows(by_dataset),
            first_model_col=3,
            merge_columns=[1],
            compare_group_size=len(models),
        ),
    }
    try:
        write_xlsx(output_path, sheets)
    except PermissionError as exc:
        raise PermissionError(
            f"Cannot write output file: {output_path}. "
            "Close the Excel file if it is open, or pass a different --output path."
        ) from exc

    print(f"[SAVE] {output_path}")
    print("Sheets: ByAttack, ByDataset")
    print(f"Metrics: {', '.join(metrics)}")
    print(f"Models: {', '.join(display_model_name(model) for model in models)}")


if __name__ == "__main__":
    main()
