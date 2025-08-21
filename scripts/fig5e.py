from pathlib import Path

import arviz as az
import bambi as bmb
import polars as pl
from matplotlib import pyplot as plt

from glycocalyx.plotting import forestplot

ROOT = Path(__file__).parent.parent
RAW_DATA_FILE = ROOT / "data" / "raw" / "plasma_hyaluronan.csv"
PREPARED_DATA_FILE = ROOT / "data" / "prepared" / "fig5e.csv"
IDATA_DIR = ROOT / "data" / "results" / "fig5e"
PLOT_DIR = ROOT / "plots" / "fig5e"
FORMULA_MU = "log(concentration) ~ treatment + (1|mouse)"
FORMULA_SIGMA = "sigma ~ treatment"
SEED = 1234


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
    idata: az.InferenceData = model.fit(target_accept=0.99, seed=SEED)
    model.predict(idata, data=data, kind="response", inplace=True)
    t = -idata.posterior["treatment"]
    k = (t > 0).mean().to_numpy().item()
    sp = max(k, 1 - k)
    ts[f"Enzyme effect (SP={round(sp, 2)})"] = t
    print(az.summary(idata))
    idata.to_netcdf(str(IDATA_DIR / "idata.nc"))
    f, ax = plt.subplots(figsize=(10, 6))
    ax = forestplot(ax, ts, xlabel="Enzyme effect")
    ax.legend(frameon=False)
    f.savefig(PLOT_DIR / "effects.svg", bbox_inches="tight")


if __name__ == "__main__":
    main()
