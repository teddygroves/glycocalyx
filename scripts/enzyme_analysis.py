"""Script that runs the enzyme analysis.

We want to statistically test for the effect of enzymes on the glycocalyx thickness and some other parameters.
"""

from pathlib import Path

import arviz as az
import bambi as bmb
import numpy as np
import polars as pl
from matplotlib import pyplot as plt

from glycocalyx.plotting import forestplot

ROOT = Path(__file__).parent.parent
DATA_FILE = ROOT / "data" / "raw" / "gcx_thickness_140825.csv"
OUT_DIR = ROOT / "data" / "results" / "enzyme"
PLOT_DIR = ROOT / "plots"
FORMULA_ETA = "log({ycol}) ~ 0 + treatment + (1|mouse)"
FORMULA_SIGMA = "sigma ~ scale(log({ycol}_std))"
YCOLS = ["w", "dx", "s", "lambda", "Ig", "Ie", "Ip", "I0"]
SEED = 1234


def plot_ppc(ax, idata, data, ycol, model):
    for (mouse,), subdf in data.with_row_index().group_by("mouse"):
        xcol = f"{ycol}_std"
        x = subdf[xcol]
        y = subdf[ycol]
        ix = subdf["index"]
        sct = ax.scatter(x, y)
        qlow, qhigh = np.exp(
            idata.posterior_predictive[f"log({ycol})"]
            .quantile([0.05, 0.95], dim=["chain", "draw"])
            .to_numpy()[:, ix]
        )
        lines = ax.vlines(
            x,
            qlow,
            qhigh,
            zorder=-1,
            color="tab:blue",
            alpha=0.6,
        )
    ax.semilogy()
    ax.set(xlabel=xcol, ylabel=ycol)
    ax.legend(
        [sct, lines],
        [
            "Observation (color indicates mouse)",
            "5%-95% posterior predictive interval",
        ],
        frameon=False,
    )
    return ax


def main():
    data = pl.read_csv(DATA_FILE)
    ts = {}
    for ycol in YCOLS:
        print(f"\n******\tAnalysing column {ycol}...\t*******")
        idata_file = OUT_DIR / f"{ycol}.nc"
        log_ycol_mean = np.log(data[ycol]).mean()
        priors = {
            "mu": {
                "treatment": bmb.Prior("Normal", mu=log_ycol_mean, sigma=0.5)
            },
            "sigma": {
                # HalfNormal is intentional, this effect must be positive!
                f"scale(log({ycol}_std))": bmb.Prior("HalfNormal", sigma=2)
            },
        }
        formula = bmb.Formula(
            FORMULA_ETA.format(ycol=ycol),
            FORMULA_SIGMA.format(ycol=ycol),
        )
        model = bmb.Model(formula, data.to_pandas(), priors=priors)
        if not idata_file.exists():
            idata = model.fit(target_accept=0.999, seed=SEED)
            model.predict(idata, data=data, kind="response", inplace=True)
            idata.observed_data[f"{ycol}_std"] = data[f"{ycol}_std"]
            idata.to_netcdf(idata_file)
        else:
            idata = az.from_netcdf(idata_file)
        effect = idata.posterior["treatment"].sel(
            treatment_dim="Enzyme"
        ) - idata.posterior["treatment"].sel(treatment_dim="Saline")
        qlow, qhigh = effect.quantile([0.05, 0.95]).to_numpy()
        k = (effect > 0).mean().item()
        sp = max(k, 1 - k)
        name = f"{ycol} (SP = {round(sp, 2)})"
        ts[name] = effect
        f, ax = plt.subplots(figsize=(10, 6))
        ax = plot_ppc(ax, idata, data, ycol, model)
        f.savefig(PLOT_DIR / "enzyme" / f"ppc_{ycol}.svg", bbox_inches="tight")
        print(az.summary(idata))
        print(f"5% {ycol} effect quantile: {round(qlow, 2)}")
        print(f"95% {ycol} effect quantile: {round(qhigh, 2)}")

    f, ax = plt.subplots(figsize=(10, 6))
    ax = forestplot(ax, ts, xlabel="Effect difference (Enzyme - Saline)")
    ax.legend(frameon=False)
    f.savefig(PLOT_DIR / "enzyme" / "enzyme_effects.svg", bbox_inches="tight")


if __name__ == "__main__":
    main()
