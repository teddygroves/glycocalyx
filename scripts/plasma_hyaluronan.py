from pathlib import Path

import arviz as az
import bambi as bmb
import polars as pl
from matplotlib import pyplot as plt
import numpy as np

from glycocalyx.plotting import forestplot

ROOT = Path(__file__).parent.parent
RAW_DATA_FILE = ROOT / "data" / "raw" / "plasma_hyaluronan.csv"
PREPARED_DATA_FILE = ROOT / "data" / "prepared" / "plasma_hyaluronan.csv"
IDATA_DIR = ROOT / "data" / "results" / "plasma_hyaluronan"
PLOT_DIR = ROOT / "plots" / "plasma_hyaluronan"
FORMULA_MU = "log(concentration) ~ treatment + (1|mouse)"
FORMULA_SIGMA = "sigma ~ treatment"
SEED = 1234


def plot_ppc(ax, idata, data, model):
    treatment_to_x = {"Saline": 0.4, "Enzymes": 0.6}
    plot_df = data.with_columns(
        x_mean=pl.col("treatment").map_elements(
            treatment_to_x.get, return_dtype=pl.Float32
        ),
        jitter=np.random.normal(loc=0, scale=0.01, size=len(data)),
    ).with_columns(x=pl.col("x_mean") + pl.col("jitter"))
    for (mouse,), subdf in plot_df.with_row_index().group_by("mouse"):
        x = subdf["x"]
        y = subdf["concentration"]
        ix = subdf["index"]
        sct = ax.scatter(x, y)
        qlow, qhigh = np.exp(
            idata.posterior_predictive["log(concentration)"]
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
    ax.semilogy()
    ax.set(xlabel="Treatment", ylabel="Concentration")
    ax.set_xticks(list(treatment_to_x.values()), list(treatment_to_x.keys()))
    ax.set_xlim(0.3, 0.7)
    ax.legend(
        [sct, lines],
        [
            "Observation (color indicates mouse)",
            "2.5%-97.5% posterior predictive interval",
        ],
        frameon=False,
    )
    return ax


def main():
    data = pl.read_csv(RAW_DATA_FILE)
    ts = {}
    formula = bmb.Formula(FORMULA_MU, FORMULA_SIGMA)
    priors = {
        # Unlikely that any mouse is more than 0.6 different from
        # the others on log scale
        "1|mouse": bmb.Prior(
            "Normal",
            mu=0.0,
            sigma=bmb.Prior("HalfNormal", sigma=0.3),
        )
    }
    model = bmb.Model(formula, data=data.to_pandas(), priors=priors)
    idata_file = IDATA_DIR / "idata.nc"
    if not idata_file.exists():
        idata: az.InferenceData = model.fit(target_accept=0.99, seed=SEED)
        model.predict(idata, data=data, kind="response", inplace=True)
        print(az.summary(idata))
        idata.to_netcdf(str(IDATA_DIR / "idata.nc"))
    else:
        idata = az.from_netcdf(idata_file)
    t = -idata.posterior["treatment"]
    k = (t > 0).mean().to_numpy().item()
    sp = max(k, 1 - k)
    ts[f"Enzyme effect (SP={round(sp, 2)})"] = t
    f, ax = plt.subplots(figsize=(10, 6))
    ax = forestplot(ax, ts, xlabel="Enzyme effect")
    ax.legend(frameon=False)
    f.savefig(PLOT_DIR / "effects.svg", bbox_inches="tight")
    f, ax = plt.subplots(figsize=(10, 6))
    ax = plot_ppc(ax, idata, data, model)
    f.savefig(PLOT_DIR / "ppc.svg", bbox_inches="tight")


if __name__ == "__main__":
    main()
