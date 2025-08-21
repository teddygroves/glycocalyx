# Glycocalyx analysis

This repository contains code analysing measurements of glycocalyx density in mice.

To run the analysis, first install [cmdstanpy](https://mc-stan.org/cmdstanpy/index.html) and [uv](https://docs.astral.sh/uv/). Then run the following commands from the project root:

```sh
> uv run scripts/fig5e.py
> uv run scripts/fig3.py
> uv run scripts/fig6.py
> uv run scripts/igcx_analysis.py
> uv run scripts/enzyme_analysis.py
```

The shell script `run_analysis.sh` runs all these commands in one go:

```sh
> bash run_analysis.sh
```

## Analyses

We constructed five models:

- (figure 5e) Model of plasma hyaluronic acid level measurements. We used a distributional model to account for treatment-dependent heteroskedasticity. For the expected measurement value we used a linear regression on logarithmic scale, with non-random effects for treatment (Enzyme or Saline) and random effects per mouse. For the expected measurement error we used a linaer regression on logarithmic scale with a non-random treatment effect.
- (figure 3) Model of glycocalyx intensity measurements under dual-lectin mapping. We used a linear regression on log scale, with random intercepts per vessel type and per lectin/vessel type interaction class.
- (figure 6) Models of FRAP parameters. We fit separate linear regression models to each set of parameters after transforming the parameter values to unconstrained space. Each model had non-random treatment effects and random effect for mouse and ROI. 
- (igcx analysis) Model of $I_{max}$ measurements. Linear regression with non-random effects for treatment, vessel type and treatment:vessel type interaction, as well as random effects for mouse and vessel. 
- (enzyme analysis) Model of FRAP parameters under Enzyme and Saline treatments. We used a distributional model due to availability of information about the likely accuracy of the measurements. For the expected measurement value we used a linear regression on log scale with non-random treatment effects and random effects per mouse. For the expected measurement error we used a linear regression on log scale with a non-random, positive-constrained effect for the measurement error predictor.

Each model was fitted by adaptive Hamiltonian Monte Carlo to obtain posterior samples.

