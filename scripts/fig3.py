from pathlib import Path

import arviz as az
import cmdstanpy
import numpy as np
import polars as pl
from matplotlib import pyplot as plt

from glycocalyx.plotting import forestplot, resid_scatter
from glycocalyx.util import get_ln_mean, standardise

ROOT = Path(__file__).parent.parent
RAW_DATA_FILE = ROOT / "data" / "raw" / "GlycocalyxPaper_DifferentLectins.csv"
PIXEL_FILE = ROOT / "data" / "prepared" / "pixels.csv"
CONTOUR_FILE = ROOT / "data" / "prepared" / "contours.csv"
STAN_FILE = ROOT / "src" / "glycocalyx" / "model.stan"
IDATA_FILE = ROOT / "data" / "results" / "fig3" / "idata.nc"
PLOT_DIR = ROOT / "plots" / "fig3"
GROUP_COLS = ["lectin", "mouse", "vessel_type", "contour_in_vessel_type"]
SEED = 1234


def jsonize(d: dict):
    def jsonize_value(val):
        if isinstance(val, int):
            return val
        elif isinstance(val, float):
            return val
        elif isinstance(val, pl.Series):
            return val.to_numpy()
        else:
            msg = f"Can't jsonize {val}"
            raise ValueError(msg)

    return {k: jsonize_value(v) for k, v in d.items()}


VesselType = pl.Enum(["pa", "pea", "bp_a", "bp_v", "pv", "av"])
LectinVesselType = pl.Enum(
    [
        "lea:pa",
        "wga:pa",
        "lea:pea",
        "wga:pea",
        "lea:bp_a",
        "wga:bp_a",
        "lea:bp_v",
        "wga:bp_v",
        "lea:pv",
        "wga:pv",
        "lea:av",
        "wga:av",
    ]
)


def prepare_data():
    pxl_raw = pl.read_csv(RAW_DATA_FILE, infer_schema_length=20000)
    pixel = (
        pxl_raw.with_columns(ln_y=np.log(pl.col("I_gcx")))
        .pipe(lambda df: df.with_columns(mouse_vt_ln_mean=get_ln_mean(df)))
        .with_columns(ln_y_norm=pl.col("ln_y") - pl.col("mouse_vt_ln_mean"))
    )
    contour_group = pixel.group_by(GROUP_COLS)
    contour = (
        contour_group.agg(pl.col("ln_y_norm").mean().alias("ln_y_norm_mean"))
        .with_columns(ln_y_norm_mean_std=standardise(pl.col("ln_y_norm_mean")))
        .with_columns(size=contour_group.len()["len"])
        .sort(*GROUP_COLS)
    )
    contour.write_csv(CONTOUR_FILE)
    pixel.write_csv(PIXEL_FILE)


def fit():
    msts = (
        pl.read_csv(CONTOUR_FILE)
        .with_columns(
            lectin=pl.col("lectin").cast(pl.Categorical),
            mouse=pl.col("mouse").cast(pl.Categorical),
            vessel_type=pl.col("vessel_type").cast(VesselType),
            lectin_vessel_type=pl.concat_str(
                [pl.col("lectin"), pl.col("vessel_type")], separator=":"
            ).cast(LectinVesselType),
        )
        .with_row_index()
    )
    model = cmdstanpy.CmdStanModel(stan_file=STAN_FILE)
    data = jsonize(
        {
            "N": len(msts),
            "N_lectin": len(msts["lectin"].cat.get_categories()),
            "N_mouse": len(msts["mouse"].cat.get_categories()),
            "N_vessel_type": len(msts["vessel_type"].cat.get_categories()),
            "N_lectin_vessel_type": len(
                msts["lectin_vessel_type"].cat.get_categories()
            ),
            "size": msts["size"],
            "y": msts["ln_y_norm_mean_std"],
            "lectin": msts["lectin"].to_physical() + 1,
            "mouse": msts["mouse"].to_physical() + 1,
            "vessel_type": msts["vessel_type"].to_physical() + 1,
            "lectin_vessel_type": msts["lectin_vessel_type"].to_physical() + 1,
        }
    )
    coords = {
        "lectin": msts["lectin"].cat.get_categories(),
        "mouse": msts["mouse"].cat.get_categories(),
        "vessel_type": msts["vessel_type"].cat.get_categories(),
        "lectin_vessel_type": msts["lectin_vessel_type"].cat.get_categories(),
        "observation": msts["index"],
    }
    dims = {
        "a_lectin": ["lectin"],
        "a_mouse": ["mouse"],
        "a_vessel_type": ["vessel_type"],
        "a_lectin_vessel_type": ["lectin_vessel_type"],
        "yrep": ["observation"],
        "y": ["observation"],
    }
    mcmc = model.sample(
        data=data,
        chains=4,
        iter_warmup=1000,
        iter_sampling=1000,
        adapt_delta=0.999,
        max_treedepth=11,
        seed=SEED,
    )
    idata = az.from_cmdstanpy(
        mcmc,
        posterior_predictive="yrep",
        observed_data=data,
        coords=coords,
        dims=dims,
    )
    print(az.summary(idata, var_names=["~yrep", "~free"], filter_vars="regex"))
    idata.to_netcdf(str(IDATA_FILE))


def analyse():
    msts = pl.read_csv(CONTOUR_FILE)
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
            t = (
                idata.posterior["a_vessel_type"].sel(vessel_type=vta)
                + idata.posterior["a_lectin_vessel_type"].sel(
                    lectin_vessel_type=f"{lt}:{vta}"
                )
                - idata.posterior["a_vessel_type"].sel(vessel_type=vtb)
                - idata.posterior["a_lectin_vessel_type"].sel(
                    lectin_vessel_type=f"{lt}:{vtb}"
                )
            )
            k = (t > 0).mean().item()
            sp = max(k, 1 - k)
            name = (
                f"Effect difference: {vta} - {vtb}, lectin {lt}"
                f" (SP={round(sp, 2)})"
            )
            ts[name] = t
        arteriole_effect, venule_effect = (
            (
                idata.posterior["a_vessel_type"].sel(vessel_type=vtypelist)
                + idata.posterior["a_lectin_vessel_type"].sel(
                    lectin_vessel_type=[f"{lt}:{vt}" for vt in vtypelist]
                )
            ).mean(dim=["vessel_type", "lectin_vessel_type"])
            for vtypelist in [arterioles, venules]
        )
        t = arteriole_effect - venule_effect
        k = (t > 0).mean().item()
        sp = max(k, 1 - k)
        name = (
            f"Average effect difference: arterioles - venules, lectin {lt}"
            f" (SP={round(sp, 2)})"
        )
        ts[name] = t
    f, ax = plt.subplots(figsize=[15, 8])
    ax = forestplot(ax, ts)
    f.savefig(PLOT_DIR / "vessel_effects.svg", bbox_inches="tight", dpi=300)
    # lectin figure
    ts = {}
    for vt in idata.posterior.coords["vessel_type"].values:
        wga = idata.posterior["a_lectin_vessel_type"].sel(
            lectin_vessel_type=f"wga:{vt}"
        )
        lea = idata.posterior["a_lectin_vessel_type"].sel(
            lectin_vessel_type=f"lea:{vt}"
        )
        t = wga - lea
        k = (t > 0).mean().item()
        sp = max(k, 1 - k)
        name = (
            "Average lectin effect difference: wga - lea, "
            f"vessel type {vt} (SP={round(sp, 2)})"
        )
        ts[name] = t
    for name, vtypelist in [("arterioles", arterioles), ("venules", venules)]:
        t = idata.posterior["a_lectin_vessel_type"].sel(
            lectin_vessel_type=[f"wga:{vt}" for vt in vtypelist]
        ).mean(dim=["lectin_vessel_type"]) - idata.posterior[
            "a_lectin_vessel_type"
        ].sel(lectin_vessel_type=[f"lea:{vt}" for vt in vtypelist]).mean(
            dim=["lectin_vessel_type"]
        )
        k = (t > 0).mean().item()
        sp = max(k, 1 - k)
        name = (
            "Average lectin effect difference: wga - lea, "
            f"{name} (SP={round(sp, 2)})"
        )
        ts[name] = t
    f, ax = plt.subplots(figsize=[15, 8])
    ax = forestplot(ax, ts)
    f.savefig(PLOT_DIR / "lectin_effects.svg", bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    if not (CONTOUR_FILE.exists() and PIXEL_FILE.exists()):
        prepare_data()
    if not IDATA_FILE.exists():
        fit()
    analyse()
