from pathlib import Path
from typing import Iterable

import arviz as az
import numpy as np
import polars as pl
from matplotlib import pyplot as plt


ROOT = Path(__file__).parent.parent.parent
DATA_FILE = ROOT / "data" / "prepared" / "measurements_grouped.csv"
IDATA_FILE = ROOT / "data" / "idata_grouped.nc"
PLOT_DIR = ROOT / "plots"


def histogram(msts: pl.DataFrame, groupcol: str = "mouse"):
    f, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, col in zip(axes, ["ln_y_std", "ln_y_norm_std"]):
        min = float(msts[col].cast(float).min())  # pyright: ignore[reportArgumentType]
        max = float(msts[col].cast(float).max())  # pyright: ignore[reportArgumentType]
        bins = np.linspace(min, max, 50)
        for vt, subdf in msts.group_by(groupcol):
            _ = ax.hist(
                subdf[col],
                bins=bins,
                label=vt,
                alpha=0.8,
            )
        ax.legend(frameon=False, title=groupcol)
    return f, axes


def plot_residuals(
    msts: pl.DataFrame,
    idata: az.InferenceData,
    groups: Iterable[str | Iterable[str]],
):
    resids = pl.from_pandas(
        (
            idata.posterior_predictive["ln_y_std"]  # pyright: ignore[reportAttributeAccessIssue]
            - idata.observed_data["ln_y_std"]  # pyright: ignore[reportAttributeAccessIssue]
        )
        .rename("resid")
        .to_dataframe()
        .reset_index()
    ).join(msts.with_row_index(), left_on="__obs__", right_on="index")
    bins = np.linspace(resids["resid"].min(), resids["resid"].max(), 50)
    figs = []
    for group in groups:
        f, ax = plt.subplots(figsize=[8, 5])
        for group_name, subdf in resids.group_by(group):
            ax.hist(
                subdf["resid"],
                density=True,
                bins=bins,
                alpha=0.7,
                label=str(group_name),
            )
        title = group if isinstance(group, str) else ", ".join(group)
        ax.set_title(title)
        ax.legend(frameon=False)
        figs.append((f, ax))
    return figs


def forestplot(ax, ts, xlabel="Distribution of posterior samples"):
    az.plot_forest(ts, ax=ax, combined=True, textsize=12, linewidth=3)
    ax.axvline(0.0, linestyle="--", color="black")
    xlow, xhigh = ax.get_xlim()
    xbiggest = max(abs(xlow), abs(xhigh))
    ax.set_xlim(-xbiggest, xbiggest)
    ax.set(title="", xlabel=xlabel)
    return ax


def resid_scatter(msts, idata, qlow=0.01, qhigh=0.99):
    dim = ["chain", "draw"]
    points = msts.with_columns(
        qlow=idata.posterior_predictive["yrep"].quantile(qlow, dim=dim).values,
        qhigh=idata.posterior_predictive["yrep"]
        .quantile(qhigh, dim=dim)
        .values,
    )
    f, ax = plt.subplots()
    ax.scatter(
        points["size"],
        points["ln_y_std_mean"],
        color="black",
        label="Average observed value in ROI",
    )
    ax.vlines(
        points["size"],
        points["qlow"],
        points["qhigh"],
        zorder=-1,
        label="Posterior predictive distribution",
    )
    ax.legend(frameon=False)
    ax.set(
        xlabel="Number of measurements",
        ylabel="Igcx (ln scale, standardised)",
    )
    return f, ax


def main():
    # config
    pl.Config.set_tbl_rows(100)

    # initial data exploration
    msts = pl.read_csv(DATA_FILE)
    # f, axes = histogram(msts, groupcol="vessel_type")
    # look at model results...
    idata = az.from_netcdf(IDATA_FILE)
    # Posterior predictive check
    f, ax = resid_scatter(msts, idata)
    f.savefig(PLOT_DIR / "resid_scatter.png", bbox_inches="tight", dpi=300)
    # vessel type figure
    lectins = ["wga", "lea"]
    vt_comps = [
        ("pa", "pea"),
        ("pa", "bp_a"),
        ("pea", "bp_a"),
        ("pv", "av"),
        ("pv", "bp_v"),
        ("av", "bp_v"),
    ]
    arterioles = ["pa", "pea", "bp_a"]
    venules = ["pv", "av", "bp_v"]
    ts = {}
    for lt in lectins:
        for vta, vtb in vt_comps:
            name = f"Effect difference: {vta} - {vtb}, lectin {lt}"
            ts[name] = (
                idata.posterior["a_vessel_type"].sel(vessel_type=vta)
                + idata.posterior["a_lectin_vessel_type"].sel(
                    lectin_vessel_type=f"{lt}:{vta}"
                )
                - idata.posterior["a_vessel_type"].sel(vessel_type=vtb)
                - idata.posterior["a_lectin_vessel_type"].sel(
                    lectin_vessel_type=f"{lt}:{vtb}"
                )
            )
        arteriole_effect, venule_effect = (
            (
                idata.posterior["a_vessel_type"].sel(vessel_type=vtypelist)
                + idata.posterior["a_lectin_vessel_type"].sel(
                    lectin_vessel_type=[f"{lt}:{vt}" for vt in vtypelist]
                )
            ).mean(dim=["vessel_type", "lectin_vessel_type"])
            for vtypelist in [arterioles, venules]
        )
        ts[f"Average effect difference: arterioles - venules, lectin {lt}"] = (
            arteriole_effect - venule_effect
        )
    f, ax = plt.subplots(figsize=[15, 8])
    ax = forestplot(ax, ts)
    f.savefig(PLOT_DIR / f"vessel_effects.png", bbox_inches="tight", dpi=300)
    # lectin figure
    ts = {}
    for vt in idata.posterior.coords["vessel_type"].values:
        name = f"Lectin effect difference: wga - lea, vessel type {vt}"
        wga = idata.posterior["a_lectin"].sel(lectin="wga") + idata.posterior[
            "a_lectin_vessel_type"
        ].sel(lectin_vessel_type=f"wga:{vt}")
        lea = idata.posterior["a_lectin"].sel(lectin="lea") + idata.posterior[
            "a_lectin_vessel_type"
        ].sel(lectin_vessel_type=f"lea:{vt}")
        ts[name] = wga - lea
    for name, vtypelist in [("arterioles", arterioles), ("venules", venules)]:
        ts[f"Average lectin effect difference: wga - lea, {name}"] = (
            idata.posterior["a_lectin"].sel(lectin="wga")
            - idata.posterior["a_lectin"].sel(lectin="lea")
            + idata.posterior["a_lectin_vessel_type"]
            .sel(lectin_vessel_type=[f"wga:{vt}" for vt in vtypelist])
            .mean(dim=["lectin_vessel_type"])
            - idata.posterior["a_lectin_vessel_type"]
            .sel(lectin_vessel_type=[f"lea:{vt}" for vt in vtypelist])
            .mean(dim=["lectin_vessel_type"])
        )
    f, ax = plt.subplots(figsize=[15, 8])
    ax = forestplot(ax, ts)
    ax.set_xlim(-2.8, -1.8)
    f.savefig(PLOT_DIR / "lectin_effects.png", bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    main()
