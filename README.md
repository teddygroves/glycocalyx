# Statistical analysis of glycocalyx heterogeneity

This repository contains statistical analysis of data produced by in vivo two-photon microscopy of fluorescently-labeled glycocalyx in mice

To run the analysis, first install [cmdstanpy](https://mc-stan.org/cmdstanpy/index.html) and [uv](https://docs.astral.sh/uv/). Then run the following commands from the project root:

```sh
> uv run scripts/plasma_hyaluronan.py
> uv run scripts/rel_gcx_intensity.py
> uv run scripts/gcx_frap.py
> uv run scripts/abs_gcx_intenxity.py
> uv run scripts/gcx_thickness.py
```

The shell script `run_analysis.sh` runs all these commands in one go:

```sh
> bash run_analysis.sh
```

## Analyses

We constructed five models (see more details in Supplementary Note: Statistical models): 

- Model of relative glycocalyx fluorescence intensity (glycocalyx maps; Fig. 3) from the data obtained with dual-lectin labelling. We used a linear regression on log scale, with random intercepts per vessel type and per lectin/vessel type interaction class. 

- Model of absolute glycocalyx fluorescence intensity (fast imaging protocol; Fig.5d). Linear regression with non-random effects for treatment, vessel type and treatment:vessel type interaction class (i.e. the combination of treatment and vessel type), as well as random effects for mouse and vessel. 

- Model of plasma hyaluronan concentration (Fig. 5e). We used a distributional model to account for treatment-dependent heteroskedasticity. For the expected measurement value we used a linear regression on logarithmic scale, with non-random effects for treatment (Enzyme or Saline) and random effects per mouse. For the expected measurement error we used a linear regression on logarithmic scale with a non-random treatment effect.  

- Models of FRAP parameters (Fig. 6). We fit separate linear regression models to each set of parameters after transforming the parameter values to unconstrained space. Each model had non-random treatment effects and random effects for mouse and ROI.  

- Model of parameters obtained by fitting glycocalyx line-profiles of fluorescence (Fig.7). We used a distributional model due to availability of information about the likely accuracy of the measurements. For the expected measurement value we used a linear regression on log scale with non-random treatment effects and random effects per mouse. For the expected measurement error we used a linear regression on log scale with a non-random, positive-constrained effect for the measurement error predictor. 

Each model was fitted by adaptive Hamiltonian Monte Carlo to obtain posterior samples and evaluated based on qualitative fit to the observed data under posterior predictive checking. 
