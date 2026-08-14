from pathlib import Path

import arviz as az
import bambi as bmb
import matplotlib.pyplot as plt
import polars as pl

from glycocalyx.plotting import forestplot

ROOT = Path(__file__).parent.parent
PLOT_DIR = ROOT / "plots"
RAW_DATA_FILE = ROOT / "data" / "raw" / "blood_cell_gcx.csv"
FORMULA_INTENSITY = "intensity ~ 1 + treatment + mouse_id + (1|capillary_id)"
FORMULA_SIGMA = "sigma ~ capillary_id"


def hierarchical_prior(sd: float):
    return bmb.Prior(
        "Normal",
        mu=0.0,
        sigma=bmb.Prior("HalfNormal", sigma=sd),
    )


def bcgcx_forestplot(mcmc):
    f, ax = plt.subplots()
    effect = mcmc.posterior["treatment"]
    ts = {"treatment_effect": effect}
    ax = forestplot(ax, ts, xlabel="Treatment effect")
    return f, ax


def plot_result(
    msts: pl.DataFrame,
    mcmc: az.InferenceData,
):
    qs = (0.025, 0.975)
    lows, highs = (
        mcmc.posterior_predictive["intensity"]  # type:ignore
        .quantile(q, dim=["chain", "draw"])
        .to_numpy()
        for q in qs
    )
    plot_df = msts.with_columns(qlow=lows, qhigh=highs)
    f, axes = plt.subplots(1, 2, figsize=[15, 5])
    for ax, ((mouse_id,), subdf) in zip(
        axes,
        plot_df.group_by("mouse_id", maintain_order=True),
    ):
        for (treatment,), subsubdf in subdf.group_by(
            "treatment",
            maintain_order=True,
        ):
            ax.scatter(
                subsubdf["clock_time"],
                subsubdf["intensity"],
                label=treatment,
            )
            ax.vlines(
                subsubdf["clock_time"],
                subsubdf["qlow"],
                subsubdf["qhigh"],
                color="gainsboro",
                zorder=-1,
            )
        ax.legend(title="Treatment")
        ax.set_xticks([])
        ax.set(title=f"Mouse {mouse_id}")
    return f, axes


def main():
    msts = (
        pl.read_csv(RAW_DATA_FILE)
        .with_columns(
            ct=pl.col("clock_time")
            .str.to_time("%R:%S.%3f")
            .cast(pl.Duration("ms")),
            mouse_id=pl.concat_str(pl.lit("M"), pl.col("mouse_id")),
            capillary_id=pl.concat_str(pl.lit("C"), pl.col("capillary_id")),
        )
        .with_columns(
            capillary_id=pl.concat_str(
                pl.col("mouse_id"), pl.col("capillary_id")
            ),
        )
    )
    treatment_time_df = (
        msts.filter(treatment="neuraminidase")
        .group_by("mouse_id")
        .agg(pl.col("clock_time").min())
    )
    treatment_times = dict(
        zip(
            treatment_time_df["mouse_id"],
            treatment_time_df["clock_time"],
        )
    )
    means = msts.group_by(
        "mouse_id",
        "capillary_id",
        "blood_cell_id",
        maintain_order=True,
    ).agg(
        pl.col("intensity").mean(),
        pl.col("clock_time").min(),
        pl.col("time").min(),
        pl.col("treatment").first(),
    )
    priors = {
        "Intercept": bmb.Prior("Normal", sigma=30.0),
        "treatment": bmb.Prior("Normal", sigma=30.0),
        "1|capillary_id": hierarchical_prior(sd=10.0),
        "mouse_id": bmb.Prior("Normal", sigma=10.0),
        "sigma_Intercept": bmb.Prior("Normal", sigma=2.0),
        "sigma_capillary_id": bmb.Prior("Normal", sigma=1.0),
    }
    model = bmb.Model(
        formula=bmb.Formula(FORMULA_INTENSITY, FORMULA_SIGMA),
        family="t",
        data=means.to_pandas(),
        priors=priors,
    )
    mcmc = model.fit()
    mcmc = model.predict(mcmc, inplace=False, kind="response")
    f, _ = plot_result(means, mcmc)
    f.savefig(PLOT_DIR / "blood_cell_gcx_ppc.svg")
    f, _ = bcgcx_forestplot(mcmc)
    f.savefig(PLOT_DIR / "blood_cell_gcx.svg")


if __name__ == "__main__":
    main()
