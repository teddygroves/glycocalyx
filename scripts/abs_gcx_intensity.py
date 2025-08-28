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
RAW_DATA_FILE = ROOT / "data" / "raw" / "abs_gcx_intensity.csv"
PREPARED_DATA_FILE = ROOT / "data" / "prepared" / "abs_gcx_intensity.csv"
IDATA_DIR = ROOT / "data" / "results" / "abs_gcx_intensity"
PLOT_DIR = ROOT / "plots" / "abs_gcx_intensity"
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


def plot_ppc(fig, axes, idata, data, model, ycol):
    plot_df_enzyme, plot_df_saline = (
        data.with_row_index(name="idata_index")
        .filter(treatment=t)
        .sort("mouse")
        .with_row_index()
        .with_columns(x=pl.col("index") / pl.col("index").len())
        for t in ("E", "S")
    )
    for ax, plot_df, title in zip(
        axes,
        (plot_df_enzyme, plot_df_saline),
        ("Enzyme", "Saline"),
    ):
        for (mouse,), subdf in plot_df.group_by("mouse"):
            x = subdf["x"]
            y = subdf[ycol]
            sct = ax.scatter(x, y, s=8)
            qlow, qhigh = (
                idata.posterior_predictive[ycol]
                .quantile([0.025, 0.975], dim=["chain", "draw"])
                .to_numpy()[:, subdf["idata_index"]]
            )
            lines = ax.vlines(
                x,
                qlow,
                qhigh,
                zorder=-1,
                color="gainsboro",
            )
        ax.set(xlabel="Arbitrary order", ylabel=ycol, title=title)
        ax.set_xticks([])
        # ax.set_xlim(0.3, 0.7)
    fig.legend(
        [sct, lines],
        [
            "Observation (color indicates mouse)",
            "2.5%-97.5% posterior predictive interval",
        ],
        loc="right",
        frameon=False,
    )
    return fig, axes


def prepare_data(raw_data: pl.DataFrame) -> pl.DataFrame:
    return (
        raw_data.remove(  # this capillary has unrealistic values
            (pl.col("mouse") == 20230511)
            & (pl.col("vessel_type") == "c")
            & (pl.col("vessel") == 3.0)
        )
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
        idata_file = IDATA_DIR / f"{ycol}.nc"
        formula = bmb.Formula(formula.format(y=ycol + "_stnd"))
        model = bmb.Model(
            formula, data=prepared_data.to_pandas(), priors=PRIORS
        )
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
        f, axes = plt.subplots(2, 1, figsize=(10, 6))
        f, axes = plot_ppc(f, axes, idata, prepared_data, model, ycol + "_stnd")
        f.savefig(PLOT_DIR / f"ppc_{ycol}.svg", bbox_inches="tight")


if __name__ == "__main__":
    main()
