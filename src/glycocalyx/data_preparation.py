from pathlib import Path
import numpy as np
import polars as pl

ROOT = Path(__file__).parent.parent.parent
RAW_DATA_FILE = ROOT / "data" / "raw" / "GlycocalyxPaper_DifferentLectins.csv"
OUTPUT_FILE = ROOT / "data" / "prepared" / "measurements.csv"
OUTPUT_FILE_GROUPED = ROOT / "data" / "prepared" / "measurements_grouped.csv"
GROUP_COLS = ["lectin", "mouse", "vessel_type", "contour_in_vessel_type"]


def standardise(expr: pl.Expr):
    return (expr - expr.mean()) / expr.std()


def main():
    raw = pl.read_csv(RAW_DATA_FILE, infer_schema_length=20000)
    out = raw.with_columns(
        y_std=standardise(pl.col("I_gcx")),
        y_norm_std=standardise(pl.col("I_gcx_norm")),
        ln_y_std=standardise(np.log(pl.col("I_gcx"))),
        ln_y_norm_std=standardise(np.log(pl.col("I_gcx_norm"))),
    )
    group = out.group_by(GROUP_COLS)
    grouped = (
        group.agg(pl.col("ln_y_std").mean().alias("ln_y_std_mean"))
        .with_columns(size=group.len()["len"])
        .sort(*GROUP_COLS)
    )
    grouped.write_csv(OUTPUT_FILE_GROUPED)
    out.write_csv(OUTPUT_FILE)


if __name__ == "__main__":
    main()
