import polars as pl


def standardise(expr: pl.Expr):
    return (expr - expr.mean()) / expr.std()


def get_ln_mean(df: pl.DataFrame) -> pl.Series:
    group = df.group_by("mouse", "vessel_type")
    mean = group.agg(pl.col("ln_y").mean().alias("mean"))
    return df.join(mean.filter(vessel_type="pa"), on="mouse")["mean"]
