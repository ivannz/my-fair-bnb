import numpy as np
from matplotlib import pyplot as plt

from matplotlib.axes import Axes
from numpy import ndarray


def pp_curve(*, x: ndarray, y: ndarray, num: int = None) -> tuple[ndarray, ndarray]:
    """Build threshold-parameterized pipi curve."""
    # sort each sample for fast O(\log n) eCDF queries by `searchsorted`
    x, y = np.sort(x), np.sort(y)

    # pool sorted samples to get thresholds
    xy = np.concatenate((x, y))
    if num is None:
        # finest detail thresholds: sort the pooled samples (sorted
        #  arrays can be merged in O(n), but it turns out numpy does
        #  not have the procedure)
        xy.sort()

    else:
        # coarsen by finding threshold grid in the pooled sample, that
        #  is equispaced after being transformed by the empirical cdf.
        xy = np.quantile(xy, np.linspace(0, 1, num=num), method="linear")

    # add +ve/-ve inf end points to the parameter value sequence
    xy = np.r_[-np.inf, xy, +np.inf]

    # we build the pp-curve the same way as we build the ROC curve:
    #  by parameterizing with the a monotonic threshold sequence
    #    pp: v \mapsto (\hat{F}_x(v), \hat{F}_y(v))
    #  where \hat{F}_S(v) = \frac1{n_S} \sum_j 1_{S_j \leq v}
    p = np.searchsorted(x, xy) / len(x)
    q = np.searchsorted(y, xy) / len(y)

    return p, q


def plot_series(ax: Axes = None, n_last: int = 25, **series) -> None:
    """Plot the series, adding a marker to indicate the average value"""
    ax = plt.gca() if ax is None else ax

    els = {}
    for name, x in series.items():
        (el,) = ax.plot(x, label=name)
        # add the average estimate tick to the right-hand side
        # XXX throws a warning on all-nan slices
        avg, col = np.nanmean(x[-n_last:]), el.get_color()
        ax.axhline(
            avg,
            0.975,
            c=col,
            alpha=0.25,
            zorder=-10,
        )
        ax.annotate(
            f"{avg:.2g}",
            c=col,
            fontsize="xx-small",
            xy=(1.005, avg),
            xytext=(0.0, -2.0),
            xycoords=("axes fraction", "data"),
            textcoords="offset points",
            zorder=-10,
        )
        els[name] = el

    ax.legend(els.values(), series, loc="best", fontsize="x-small", ncol=3)

    return els
