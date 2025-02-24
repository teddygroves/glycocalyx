from pathlib import Path
import numpy as np
import polars as pl

ROOT = Path(__file__).parent.parent.parent
RAW_DATA_FILE = ROOT / "data" / "raw" / "GlycocalyxPaper_DifferentLectins.csv"
OUTPUT_FILE = ROOT / "data" / "prepared" / "pixels.csv"
OUTPUT_FILE_GROUPED = ROOT / "data" / "prepared" / "contours.csv"
GROUP_COLS = ["lectin", "mouse", "vessel_type", "contour_in_vessel_type"]


def standardise(expr: pl.Expr):
    return (expr - expr.mean()) / expr.std()


def get_ln_mean(df: pl.DataFrame) -> pl.Series:
    group = df.group_by("mouse", "vessel_type")
    mean = group.agg(pl.col("ln_y").mean().alias("mean"))
    return df.join(mean.filter(vessel_type="pa"), on="mouse")["mean"]


def main():
    pxl_raw = pl.read_csv(RAW_DATA_FILE, infer_schema_length=20000)
    pixel = (
        pxl_raw.with_columns(ln_y=np.log(pl.col("I_gcx")))
        .pipe(lambda df: df.with_columns(mouse_vt_ln_mean=get_ln_mean(df)))
        .with_columns(ln_y_norm=pl.col("ln_y") - pl.col("mouse_vt_ln_mean"))
    )
    contour_group = pixel.group_by(GROUP_COLS)
    contour = (
        contour_group.agg(pl.col("ln_y_norm").mean().alias("ln_y_norm_mean"))
        .with_columns(ln_y_norm_mean_std=standardise(pl.col("ln_y_norm_mean")))
        .with_columns(size=contour_group.len()["len"])
        .sort(*GROUP_COLS)
    )
    contour.write_csv(OUTPUT_FILE_GROUPED)
    pixel.write_csv(OUTPUT_FILE)


if __name__ == "__main__":
    main()
