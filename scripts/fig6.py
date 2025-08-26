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
YCOLS_TRANSFORMED = [
    "logit_alpha",
    "log_c0",
    "log_cr",
    "log_k1",
    "log_k2",
    "log_r1",
    "log_r2",
    "log_I_plasma",
]
SEED = 1234


def plot_ppc(ax, idata, data, model, ycol):
    treatment_to_x = {"S": 0.4, "E": 0.6, "N": 0.8}
    plot_df = data.with_columns(
        x_mean=pl.col("treatment").map_elements(
            treatment_to_x.get, return_dtype=pl.Float32
        ),
        jitter=np.random.normal(loc=0, scale=0.01, size=len(data)),
    ).with_columns(x=pl.col("x_mean") + pl.col("jitter"))
    for (mouse,), subdf in plot_df.with_row_index().group_by("mouse"):
        x = subdf["x"]
        y = subdf[ycol]
        ix = subdf["index"]
        sct = ax.scatter(x, y)
        qlow, qhigh = (
            idata.posterior_predictive[ycol]
            .quantile([0.025, 0.975], dim=["chain", "draw"])
            .to_numpy()[:, ix]
        )
        lines = ax.vlines(
            x,
            qlow,
            qhigh,
            zorder=-1,
            color="gainsboro",
        )
    ax.set(xlabel="Treatment", ylabel=ycol)
    ax.set_xticks(list(treatment_to_x.values()), list(treatment_to_x.keys()))
    ax.set_xlim(0.3, 0.9)
    ax.legend(
        [sct, lines],
        [
            "Observation (color indicates mouse)",
            "2.5%-97.5% posterior predictive interval",
        ],
        frameon=False,
    )
    return ax


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
        .filter(pl.col("cr") > 0.0)
        .with_columns(
            logit_alpha=logit(pl.col("alpha")),
            log_c0=np.log(pl.col("c0")),
            log_cr=np.log(pl.col("cr")),
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
    for ycol, ycol_transformed in zip(YCOLS, YCOLS_TRANSFORMED):
        print(f"Fitting dependent variable {ycol_transformed}...")
        idata_file = IDATA_DIR / f"{ycol}.nc"
        formula = bmb.Formula(MODEL_FORMULA.format(y=ycol_transformed))
        model = bmb.Model(formula, data=prepared_data.to_pandas())
        if not idata_file.exists():
            idata: az.InferenceData = model.fit(target_accept=0.99, seed=SEED)
            model.predict(
                idata,
                data=prepared_data.to_pandas(),
                kind="response",
                inplace=True,
            )
            print(az.summary(idata))
            idata.to_netcdf(str(idata_file))
        else:
            idata = az.from_netcdf(idata_file)
        posterior: xr.Dataset = idata.posterior
        t: xr.DataArray = (
            posterior["treatment"].sel(treatment_dim="E")
            - posterior["treatment"].sel(treatment_dim="S")
        ) / prepared_data[ycol_transformed].mean()
        k = (t > 0).mean().to_numpy().item()
        sp = max(k, 1 - k)
        ts[ycol + f" (SP={round(sp, 2)})"] = t
        f, ax = plt.subplots(figsize=(10, 6))
        ax = plot_ppc(ax, idata, prepared_data, model, ycol_transformed)
        f.savefig(PLOT_DIR / f"ppc_{ycol}.svg", bbox_inches="tight")
    f, ax = plt.subplots(figsize=(10, 6))
    ax = forestplot(ax, ts, xlabel="Normalised enzyme effect vs saline")
    ax.legend(frameon=False)
    f.savefig(PLOT_DIR / "effects.svg", bbox_inches="tight")


if __name__ == "__main__":
    main()
