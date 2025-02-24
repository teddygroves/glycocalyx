from pathlib import Path

import arviz as az
import polars as pl
from matplotlib import pyplot as plt


ROOT = Path(__file__).parent.parent.parent
DATA_FILE = ROOT / "data" / "prepared" / "contours.csv"
IDATA_FILE = ROOT / "data" / "idata.nc"
PLOT_DIR = ROOT / "plots"


def forestplot(ax, ts, xlabel="Distribution of posterior samples"):
    az.plot_forest(ts, ax=ax, combined=True, textsize=12, linewidth=3)
    ax.axvline(0.0, linestyle="--", color="black")
    xlow, xhigh = ax.get_xlim()
    xbiggest = max(abs(xlow), abs(xhigh))
    ax.set_xlim(-xbiggest, xbiggest)
    ax.set(title="", xlabel=xlabel)
    return ax


def resid_scatter(msts, idata, qlow=0.01, qhigh=0.99, colorcol="lectin"):
    dim = ["chain", "draw"]
    points = msts.with_columns(
        qlow=idata.posterior_predictive["yrep"].quantile(qlow, dim=dim).values,
        qhigh=idata.posterior_predictive["yrep"]
        .quantile(qhigh, dim=dim)
        .values,
    )
    f, ax = plt.subplots()
    for (groupname,), subdf in points.group_by(colorcol):
        ax.scatter(
            subdf["size"],
            subdf["ln_y_norm_mean_std"],
            label=groupname,
            alpha=0.8,
        )
    ax.vlines(
        points["size"],
        points["qlow"],
        points["qhigh"],
        zorder=-1,
        label="Posterior predictive distribution",
        color="gray",
    )
    ax.legend(frameon=False)
    ax.set(
        xlabel="Number of measurements",
        ylabel="Igcx relative to PA (ln scale, standardised)",
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
    for colorcol in [
        "mouse",
        "lectin",
        "vessel_type",
        "contour_in_vessel_type",
    ]:
        f, ax = resid_scatter(msts, idata, colorcol=colorcol)
        f.savefig(
            PLOT_DIR / f"resid_scatter_{colorcol}.svg",
            bbox_inches="tight",
            dpi=300,
        )
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
    f.savefig(PLOT_DIR / "vessel_effects.svg", bbox_inches="tight", dpi=300)
    # lectin figure
    ts = {}
    for vt in idata.posterior.coords["vessel_type"].values:
        name = f"Lectin effect difference: wga - lea, vessel type {vt}"
        wga = idata.posterior["a_lectin_vessel_type"].sel(
            lectin_vessel_type=f"wga:{vt}"
        )
        lea = idata.posterior["a_lectin_vessel_type"].sel(
            lectin_vessel_type=f"lea:{vt}"
        )
        ts[name] = wga - lea
    for name, vtypelist in [("arterioles", arterioles), ("venules", venules)]:
        ts[f"Average lectin effect difference: wga - lea, {name}"] = (
            idata.posterior["a_lectin_vessel_type"]
            .sel(lectin_vessel_type=[f"wga:{vt}" for vt in vtypelist])
            .mean(dim=["lectin_vessel_type"])
            - idata.posterior["a_lectin_vessel_type"]
            .sel(lectin_vessel_type=[f"lea:{vt}" for vt in vtypelist])
            .mean(dim=["lectin_vessel_type"])
        )
    f, ax = plt.subplots(figsize=[15, 8])
    ax = forestplot(ax, ts)
    f.savefig(PLOT_DIR / "lectin_effects.svg", bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    main()
