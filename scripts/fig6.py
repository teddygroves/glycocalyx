from pathlib import Path

import arviz as az
import bambi as bmb
import numpy as np
import polars as pl
import xarray as xr
from matplotlib import pyplot as plt
from scipy.special import logit

from glycocalyx.util import standardise
from glycocalyx.plotting import forestplot

ROOT = Path(__file__).parent.parent
RAW_DATA_FILE = ROOT / "data" / "raw" / "FRAP" / "FRAP_data.csv"
PREPARED_DATA_FILE = ROOT / "data" / "prepared" / "frap.csv"
IDATA_DIR = ROOT / "data" / "results" / "fig6"
PLOT_DIR = ROOT / "plots" / "fig6"
TREATMENT_CODES = {"No injection": "N", "Enzyme": "E", "Saline": "S"}
MODEL_FORMULA = "{y} ~ 0 + treatment + (1|mouse) + (1|roi)"
YCOLS = ["alpha", "c0", "cr", "k1", "k2", "r1", "r2", "I_plasma"]
SEED = 1234


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
            roi=pl.concat_str(pl.col("mouse"), pl.col("roi"), separator="-")
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
    ts = {}
    for ycol in YCOLS:
        print(f"Fitting dependent variable {ycol}...")
        formula = bmb.Formula(MODEL_FORMULA.format(y=ycol))
        model = bmb.Model(
            formula,
            data=prepared_data.to_pandas(),
            # priors=priors,
        )
        idata: az.InferenceData = model.fit(target_accept=0.999, seed=SEED)
        print(az.summary(idata))
        idata.to_netcdf(str(IDATA_DIR / f"{ycol}.nc"))
        posterior: xr.Dataset = idata.posterior
        t: xr.DataArray = (
            posterior["treatment"].sel(treatment_dim="E")
            - posterior["treatment"].sel(treatment_dim="S")
        ) / prepared_data[ycol].mean()
        p = (t > 0).mean().to_numpy().item()
        ts[ycol + f" (SP={round(p, 2)})"] = t
    f, ax = plt.subplots(figsize=(10, 6))
    ax = forestplot(
        ax,
        ts,
        xlabel="Normalised enzyme effect vs saline",
        qlow=0.05,
        qhigh=0.95,
    )
    ax.legend(frameon=False)
    f.savefig(PLOT_DIR / "effects.svg", bbox_inches="tight")


if __name__ == "__main__":
    main()
