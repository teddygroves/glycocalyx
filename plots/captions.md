# Figure captions

## 1) plots/fig3/resid_scatter_vessel_type.svg
Posterior predictive check for our model of glycocalyx maps. For each contour in our dataset, the plot shows the mean observed measurement (coloured dot) and our model's posterior predictive distribution for this quantity (great vertical line). Dot colours indicate vessel types and x axis position indicates number of measurements of the contour in our dataset. The graph shows general agreement between our model and the observed data, with no particular trend to make better or worse predictions depending on the number of measurements or the vessel type.

## 2) plots/fig5e/ppc.svg
Posterior predictive check for our model of Plasma hyaluronan concentration measurements. For each measurement, the coloured dot indicates the observed value and the vertical line summarises our model's posterior predictive distribution. Dot colours indicate mice. The dot-line pairs are randomly jittered along the x axis to separate them. Note that the measurements under the enzyme treatment are more dispersed, motivating the use of a distributional model, and that the model was able to capture this heteroskedasticity, as shown by the corresponding wider posterior intervals. 

## 3) plots/igcx/ppc_ln_i_max.svg

Posterior predictive check for our model of glycocalyx fluorescence intensities (see figure 5). For each measurement, the coloured dot indicates the observed value and the vertical line summarises our model's posterior predictive distribution. This graph shows general agreement between the model and observations, with the exception of the mouse with brown dots, some of whose measurements were higher than our model considered plausible. Since there were comparatively few such anomalous measurements we did not update our model to accomodate them.


## 4) All ppc plots in plots/fig6 as a multi-panel figure
Posterior predictive checks for our models of glycocalyx fluorescence recovery parameters (see figure 6). For each measurement, the coloured dot indicates the observed value and the vertical line summarises our model's posterior predictive distribution. The dot-line pairs are randomly jittered along the x axis to separate them. The graphs show that our models tended to agree with the data, with a slight tendency towards under-fitting, as shown by the fact that nearly all coloured dots lie well inside their corresponding intervals. While this suggests that we could have used narrower priors, we did not do so due to the lack of strong information about the target parameters.

## 5) All ppc_*_treatment plots in plots/enzyme
Posterior predictive checks for our models of WGA-AF-labeled glycocalyx thickness measurements (see figure 7). For each measurement, the coloured dot indicates the observed value and the vertical line summarises our model's posterior predictive distribution. The dot-line pairs are randomly jittered along the x axis to separate them. The graphs show good overall agreement between the model and data, including capturing distributional information, as shown by the model's appropriately larger and smaller intervals. As with the figure 6 plots, there is a slight tendency towards under-fitting: again we did not use narrower prior distributions as we could not justify this based on the available pre-experimental information.


