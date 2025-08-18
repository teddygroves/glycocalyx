import numpy as np
from matplotlib import pyplot as plt


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


def forestplot(ax, ts, xlabel="Test statistic", qlow=0.025, qhigh=0.975):
    ylimlow, ylimhigh = ax.get_ylim()
    ytickys = np.linspace(ylimlow, ylimhigh, len(ts) + 2)
    ys = ytickys[1:-1]
    xlows = [np.quantile(t, qlow) for t in ts.values()]
    xhighs = [np.quantile(t, qhigh) for t in ts.values()]
    xmeans = [np.mean(t) for t in ts.values()]
    xbiggest = max(np.abs(xlows + xhighs)) + 0.1
    ax.set_xlim(-xbiggest, xbiggest)
    line_label = f"{qlow * 100}%-{qhigh * 100}% interquantile range"
    for i, (y, xlow, xhigh, xmean) in enumerate(zip(ys, xlows, xhighs, xmeans)):
        line = ax.hlines(
            y=y,
            xmin=xlow,
            xmax=xhigh,
            linewidth=2,
            label=line_label if i == 0 else "",
        )
        ax.plot(
            xmean,
            y,
            marker="o",
            color=line.get_colors()[0],
            label="posterior mean" if i == 0 else "",
        )
    ax.set_yticks(ytickys, [""] + list(ts.keys()) + [""])
    # az.plot_forest(ts, ax=ax, combined=True, textsize=12, linewidth=3, hdi_prob=0.95);
    ax.axvline(0.0, linestyle="--", color="black")
    ax.set(title="", xlabel=xlabel)
    ax.tick_params(axis="y", which="both", left=False, right=False)
    return ax
