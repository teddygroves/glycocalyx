from pathlib import Path

import arviz as az
import bambi as bmb
import numpy as np
import polars as pl
import xarray as xr

from glycocalyx.util import standardise

ROOT = Path(__file__).parent.parent
RAW_DATA_FILE = (
    ROOT
    / "data"
    / "raw"
    / "pa_pea_protocols_1and2_all_results_forTEDDY_updated.csv"
)
PREPARED_DATA_FILE = ROOT / "data" / "prepared" / "igcx.csv"
IDATA_DIR = ROOT / "data" / "results" / "igcx"
TREATMENT_CODES = {"Enzyme": "E", "Saline": "S"}
FORMULA_I_MAX = (
    "{y} ~ 1"
    " + treatment"
    " + vessel_type"
    " + treatment:vessel_type"
    " + (1|mouse)"
    " + (1|vessel)"
)
FORMULA_FWHM = "{y} ~ 1 + treatment + (1|mouse)"
YCOLS = ["ln_i_max", "ln_fwhm"]
YCOL_TO_FORMULA = dict(zip(YCOLS, [FORMULA_I_MAX, FORMULA_FWHM]))

PRIORS = {
    "treatment": bmb.Prior("Normal", mu=0.0, sigma=0.5),
    "vessel_type": bmb.Prior("Normal", mu=0.0, sigma=0.5),
    "treatment:vessel_type": bmb.Prior("Normal", mu=0.0, sigma=0.25),
    "1|mouse": bmb.Prior(
        "Normal",
        mu=0.0,
        sigma=bmb.Prior("HalfNormal", sigma=0.3),
    ),
    "1|vessel": bmb.Prior(
        "Normal",
        mu=0.0,
        sigma=bmb.Prior("HalfNormal", sigma=0.3),
    ),
}


def prepare_data(raw_data: pl.DataFrame) -> pl.DataFrame:
    return (
        raw_data.with_columns(
            treatment=pl.col("treatment").map_elements(
                TREATMENT_CODES.get, return_dtype=pl.String
            )
        )
        .with_columns(
            mouse=pl.concat_str(
                pl.col("treatment"), pl.col("mouse"), separator="-"
            )
        )
        .with_columns(
            vessel=pl.concat_str(
                pl.col("mouse"), pl.col("vessel").cast(pl.Int8), separator="-"
            )
        )
        .with_columns(
            ln_i_max=np.log(pl.col("i_max")),
            ln_fwhm=np.log(pl.col("fwhm")),
        )
        .with_columns(
            standardise(pl.col(colname)).alias(colname + "_stnd")
            for colname in YCOLS
        )
    )


def main():
    raw_data = pl.read_csv(RAW_DATA_FILE)
    print("preparing data...")
    prepared_data = prepare_data(raw_data)
    prepared_data.write_csv(PREPARED_DATA_FILE)
    print(prepared_data)
    for ycol, formula in YCOL_TO_FORMULA.items():
        print(f"Fitting dependent variable {ycol}...")
        formula = bmb.Formula(formula.format(y=ycol))
        model = bmb.Model(
            formula, data=prepared_data.to_pandas(), priors=PRIORS
        )
        idata: az.InferenceData = model.fit(target_accept=0.99)
        print(az.summary(idata))
        idata.to_netcdf(str(IDATA_DIR / f"{ycol}.nc"))
        posterior: xr.Dataset = idata.posterior
        t: xr.DataArray = posterior["treatment"]
        p = (t > 0).mean().to_numpy().item()
        print("Pr(saline effect > enzyme effect): " + str(round(p, 3)))


if __name__ == "__main__":
    main()
