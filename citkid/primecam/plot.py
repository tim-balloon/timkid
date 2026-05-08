import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick

def plot_ares_opt(a_nls, fcal_indices):
    """
    Plots the current status of the power optimization procedure

    Parameters:
    a_nls (list): each value is an array of the values of the nonlinearity
        parameter for each resonator for the given iteration
    fcal_indices (array-like): indices that are calibration tones

    Returns:
    fig_hist (pyplot.fig): histogram of nonlinearity parameters for each
        iteration
    fig_opt (pyplot.fig): plot of the percent of the array that is optimized
        versus iteration number
    """
    fig_hist, ax_hist = plt.subplots(figsize = (6, 4), dpi = 200, layout = 'tight')
    ax_hist.set(xlim = (0,1.5), xlabel = 'Nonlinearity parameter', ylabel = 'Number of KIDs')
    bins = np.linspace(0, 1.5, 20)
    for index, a_nl in enumerate(a_nls):
        [a_nl[i] for i in range(len(a_nl)) if i not in fcal_indices]
        color = plt.cm.viridis(index / len(a_nls))
        if index == len(a_nls):
            alpha = 0.8 
        else:
            alpha = 0.5
        ax_hist.hist(a_nl, bins, label = index, edgecolor = color, color = color, alpha = alpha)
    ax_hist.legend(title = 'Iteration', framealpha = 1)

    fig_opt, ax_opt = plt.subplots(figsize = (6, 4), dpi = 200, layout = 'tight')
    ax_opt.set(xlabel = 'Iteration', ylabel = 'Percentage of array optimized')
    ax_opt.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax_opt.grid()
    indices, percents = [], []
    for index, a_nl in enumerate(a_nls):
        a_nl = [a_nl[i] for i in range(len(a_nl)) if i not in fcal_indices]
        color = plt.cm.viridis(index / len(a_nls))
        a_opt = [a for a in a_nl if a <= 0.6 and a >= 0.4]
        percent = len(a_opt) / (len(a_nl)) * 100
        ax_opt.scatter(index, percent, marker = 's', color = color)
        indices.append(index)
        percents.append(percent)
    ax_opt.plot(indices, percents, '--k')
    return fig_hist, fig_opt

def save_christmas_plot(idx0, state_history, fit_idx, plot_directory):
    """
    Save the iteration-by-resonator bifurcation state plot.

    Parameters
    ----------
    idx0 : int
        Current iteration index.
    state_history : np.ndarray
        Boolean array storing bifurcation state by resonator and iteration.
    fit_idx : sequence of int
        Resonator indices included in the optimization.
    plot_directory : str or path-like
        Directory where the figure should be written.
    """
    iter_range = np.array(range(idx0+1), dtype=int)
    fig, ax = plt.subplots(figsize=(12, 6))
    for i in range(len(fit_idx)):
        ax.scatter(np.array([i]*(idx0+1))[ state_history[i, :idx0+1]], iter_range[ state_history[i, :idx0+1]], c='r', marker='o')
        ax.scatter(np.array([i]*(idx0+1))[~state_history[i, :idx0+1]], iter_range[~state_history[i, :idx0+1]], c='g', marker='x')
    ax.set_xlabel('KID Index')
    ax.set_ylabel('Iteration')

    plt.savefig(plot_directory+'/christmas.png', dpi=80, bbox_inches='tight')
    plt.close()

def save_ares_history_plot(idx0, ares_history, fit_idx, plot_directory):
    """
    Save the resonator drive history accumulated so far.

    Parameters
    ----------
    idx0 : int
        Current iteration index.
    ares_history : np.ndarray
        Drive amplitude history indexed by resonator and iteration.
    fit_idx : sequence of int
        Resonator indices included in the optimization.
    plot_directory : str or path-like
        Directory where the figure should be written.
    """
    iter_range = np.array(range(idx0+1), dtype=int)
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, kid_idx in enumerate(fit_idx):
        ax.plot(iter_range, ares_history[kid_idx, iter_range], marker='o', alpha=0.5)
    ax.set_yscale('log')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('ares')
    fig.savefig(plot_directory+'/ares_history.png', dpi=80, bbox_inches='tight')
    plt.close()