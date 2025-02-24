from pathlib import Path

import cmdstanpy
import arviz as az
import polars as pl

ROOT = Path(__file__).parent.parent.parent
DATA_FILE = ROOT / "data" / "prepared" / "contours.csv"
STAN_FILE = ROOT / "src" / "glycocalyx" / "model.stan"
OUTPUT_FILE = ROOT / "data" / "idata.nc"


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


def main():
    msts = (
        pl.read_csv(DATA_FILE)
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
    )
    idata = az.from_cmdstanpy(
        mcmc,
        posterior_predictive="yrep",
        observed_data=data,
        coords=coords,
        dims=dims,
    )
    print(az.summary(idata, var_names=["~yrep", "~free"], filter_vars="regex"))
    idata.to_netcdf(str(OUTPUT_FILE))


if __name__ == "__main__":
    main()
