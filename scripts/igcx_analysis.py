from pathlib import Path

import arviz as az
import bambi as bmb
import numpy as np
import polars as pl
import xarray as xr
from matplotlib import pyplot as plt

from glycocalyx.util import standardise
from glycocalyx.plotting import forestplot

ROOT = Path(__file__).parent.parent
RAW_DATA_FILE = ROOT / "data" / "raw" / "gcx_project_imax_width.csv"
PREPARED_DATA_FILE = ROOT / "data" / "prepared" / "igcx.csv"
IDATA_DIR = ROOT / "data" / "results" / "igcx"
PLOT_DIR = ROOT / "plots" / "igcx"
TREATMENT_CODES = {"Enzyme": "E", "Saline": "S"}
FORMULA_I_MAX = (
    "{y} ~ 1"
    "+ treatment"
    "+ vessel_type"
    "+ treatment:vessel_type"
    "+ (1|mouse)"
    "+ (1|vessel)"
)
FORMULA_FWHM = "{y} ~ 1 + treatment + (1|mouse)"
YCOLS = ["ln_i_max"]
YCOL_TO_FORMULA = dict(zip(YCOLS, [FORMULA_I_MAX]))
BAD_MICE = ["20230511"]
PRIORS = {
    "treatment": bmb.Prior("Normal", mu=0.0, sigma=0.5),
    "vessel_type": bmb.Prior("Normal", mu=0.0, sigma=0.5),
    "treatment:vessel_type": bmb.Prior("Normal", mu=0.0, sigma=0.25),
    "1|mouse": bmb.Prior(
        "Normal",
        mu=0.0,
        sigma=bmb.Prior("HalfNormal", sigma=0.5),
    ),
    "1|vessel": bmb.Prior(
        "Normal",
        mu=0.0,
        sigma=bmb.Prior("HalfNormal", sigma=0.5),
    ),
}
SEED = 1234


def prepare_data(raw_data: pl.DataFrame) -> pl.DataFrame:
    return (
        raw_data.filter(~pl.col("mouse").cast(str).str.contains_any(BAD_MICE))
        .with_columns(
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
            ln_width=np.log(pl.col("width")),
        )
        .with_columns(
            standardise(pl.col(colname)).alias(colname + "_stnd")
            for colname in YCOLS
        )
        .sort(["treatment", "vessel_type"])
    )


def get_overall_treatment_effect(posterior):
    cap = posterior["treatment"]
    other = posterior["treatment:vessel_type"].mean(
        dim=["treatment:vessel_type_dim"]
    )
    return cap + other


def main():
    raw_data = pl.read_csv(RAW_DATA_FILE)
    print("preparing data...")
    prepared_data = prepare_data(raw_data)
    prepared_data.write_csv(PREPARED_DATA_FILE)
    print(prepared_data)
    for ycol, formula in YCOL_TO_FORMULA.items():
        print(f"Fitting dependent variable {ycol}...")
        formula = bmb.Formula(formula.format(y=ycol + "_stnd"))
        model = bmb.Model(
            formula, data=prepared_data.to_pandas(), priors=PRIORS
        )
        idata: az.InferenceData = model.fit(target_accept=0.99, seed=SEED)
        print(az.summary(idata))
        idata.to_netcdf(str(IDATA_DIR / f"{ycol}.nc"))
        posterior: xr.Dataset = idata.posterior
        te_overall: xr.DataArray = get_overall_treatment_effect(posterior)
        k = (te_overall > 0).mean().to_numpy().item()
        sp = max(k, 1 - k)
        print("Pr(Saline effect > Enzyme effect): " + str(round(sp, 3)))
        te_for_vt = {"cap": posterior["treatment"]}
        for vt in posterior.coords["vessel_type_dim"].values:
            te_for_vt[vt] = (
                posterior["treatment"]
                + posterior["treatment:vessel_type"].loc[
                    {"treatment:vessel_type_dim": f"S, {vt}"}
                ]
            )
        te_for_vt = {
            f"Saline effect on {ycol} for vessel type {k} (pr +ve = {(v > 0).mean().round(2).item()})": v
            for k, v in te_for_vt.items()
        }
        te_for_vt[f"Overall Saline effect (pr +ve = {round(sp, 2)})"] = (
            te_overall
        )
        f, ax = plt.subplots(figsize=(10, 6))
        ax = forestplot(ax, te_for_vt, xlabel="")
        ax.legend(frameon=False)
        f.savefig(PLOT_DIR / f"{ycol}_treatment_by_vt.svg", bbox_inches="tight")


if __name__ == "__main__":
    main()
