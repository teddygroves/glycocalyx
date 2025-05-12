from pathlib import Path

import arviz as az
import bambi as bmb
import numpy as np
import polars as pl
import xarray as xr
from scipy.special import logit

ROOT = Path(__file__).parent.parent
RAW_DATA_FILE = ROOT / "data" / "FRAP_data.csv"
PREPARED_DATA_FILE = ROOT / "data" / "FRAP_data_prepared.csv"
IDATA_DIR = ROOT / "data"
TREATMENT_CODES = {
    "No injection": "N",
    "Enzyme": "E",
    "Saline": "S",
}
MODEL_FORMULA = "{y} ~ 1 + (1|mouse) + (1|treatment)"
YCOLS = [
    "logit_alpha",
    "log_c0",
    "cr",
    "log_k1",
    "log_k2",
    "log_r1",
    "log_r2",
    "log_I_plasma",
]


def prepare_data(raw_data: pl.DataFrame) -> pl.DataFrame:
    return (
        raw_data.with_columns(
            treatment=pl.col("treatment").map_elements(
                TREATMENT_CODES.get, return_dtype=pl.String
            )
        )
        .with_columns(mouse=pl.concat_str(pl.col("treatment"), pl.col("mouse")))
        .with_columns(
            roi=pl.concat_str(
                pl.col("treatment"), pl.col("mouse"), pl.col("roi")
            )
        )
        .with_columns(
            logit_alpha=logit(pl.col("alpha")),
            log_c0=np.log(pl.col("c0")),
            log_k1=np.log(pl.col("k1")),
            log_k2=np.log(pl.col("k2")),
            log_r1=np.log(pl.col("r1")),
            log_r2=np.log(pl.col("r2")),
            log_I_plasma=np.log(pl.col("I_plasma")),
        )
    )


def main():
    raw_data = pl.read_csv(RAW_DATA_FILE)
    print("preparing data...")
    prepared_data = prepare_data(raw_data)
    prepared_data.write_csv(PREPARED_DATA_FILE)
    print(prepared_data)
    for ycol in YCOLS:
        print(f"Fitting dependent variable {ycol}...")
        model = bmb.Model(
            MODEL_FORMULA.format(y=ycol),
            data=prepared_data.to_pandas(),
        )
        idata: az.InferenceData = model.fit(target_accept=0.999)
        print(az.summary(idata))
        idata.to_netcdf(str(IDATA_DIR / f"{ycol}.nc"))
        posterior: xr.Dataset = idata.posterior
        t: xr.DataArray = posterior["1|treatment"].sel(
            treatment__factor_dim="S"
        ) - posterior["1|treatment"].sel(treatment__factor_dim="E")
        print((t > 0).mean())


if __name__ == "__main__":
    main()
