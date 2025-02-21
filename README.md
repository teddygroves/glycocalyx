# Glycocalyx analysis

This repository contains code analysing measurements of glycocalyx density in mice.

To run the analysis, first install [cmdstanpy](https://mc-stan.org/cmdstanpy/index.html) and [uv](https://docs.astral.sh/uv/). Then run the following commands from the project root:

```sh
> uv run src/glycocalyx/data_preparation.py
> uv run src/glycocalyx/fit_cmdstanpy.py
> uv run src/glycocalyx/investigate.py
```

The shell script `run_analysis.sh` runs all these commands in one go:

```sh
> bash run_analysis.sh
```
