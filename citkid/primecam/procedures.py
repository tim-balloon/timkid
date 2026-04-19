import numpy as np
from tqdm.auto import tqdm
from .update_ares import get_rfsoc_power, update_ares_pscale, update_ares_addonly
from .update_fres import update_fres
from .data_io import import_iq_noise
from .analysis import fit_iq
from .plot import plot_ares_opt, save_ares_history_plot, save_christmas_plot
from ..util import save_fig
import os

def take_iq_noise(rfsoc, fres, ares, qres, fcal_indices, file_suffix,
                  noise_time = 200, fine_bw = 0.2, rough_bw = 0.2,
                  take_rough_sweep = False, fres_update_method = 'distance',
                  npoints_rough = 300, npoints_gain = 100, npoints_fine = 600,
                  nnoise_timestreams = 1, N_accums = 5, pres = None):
    """
    Takes IQ sweeps and noise. The LO frequency must already be set.

    Parameters:
    rfsoc (citkid.primecam.instrument.RFSOC): RFSOC instance
    fres (np.array): array of center frequencies in Hz
    ares (np.array): array of amplitudes in RFSoC units
    qres (np.array): array of resonators Qs for cutting data. Resonances should
        span fres / qres
    fcal_indices (np.array): indices into fres of calibration tones
    file_suffix (str): suffix for file names
    noise_time (float or None): noise timestream length in seconds, or None to
        bypass noise acquisition
    fine_bw (float): fine sweep bandwidth in MHz. Gain bandwidth is 10 X fine
        bandwidth
    rough_bw (float): rough sweep bandwidth in MHz
    take_rough_sweep (bool): if True, first takes a rough sweep and optimizes
        the tone frequencies
    fres_update_method (str): method for updating the tone frequencies. 'mins21'
        for the minimum of |S21|, 'distance' for the point of furthest distance
        in IQ space from the off-resonance point, or 'spacing' for the point
        of largest spacing in IQ space
    nnoise_timestreams (int): number of noise timestreams to take sequentially
    N_accums (int): number of accumulations for the target sweeps
    """
    fres, ares, qres = np.array(fres), np.array(ares), np.array(qres)
    if file_suffix != '':
        file_suffix = '_' + file_suffix
    if take_rough_sweep:
        np.save(rfsoc.out_directory + f'fres_initial{file_suffix}.npy', fres)
    np.save(rfsoc.out_directory + f'ares{file_suffix}.npy', ares)
    np.save(rfsoc.out_directory + f'qres{file_suffix}.npy', qres)
    np.save(rfsoc.out_directory + f'fcal_indices{file_suffix}.npy',
            fcal_indices)
    # write initial target comb
    rfsoc.write_targ_comb_from_custom(fres, ares, pres = pres)
    # rough sweep
    if take_rough_sweep:
        filename = f's21_rough{file_suffix}.npy'
        npoints = npoints_rough
        bw = rough_bw
        rfsoc.target_sweep(filename, npoints = npoints, bandwidth = bw, N_accums = N_accums)
        f, i, q = np.load(rfsoc.out_directory + filename)
        z = i + 1j * q
        fres = update_fres(f, z, npoints = npoints,
                               fcal_indices = fcal_indices,
                               method = fres_update_method)
        rfsoc.write_targ_comb_from_custom(fres, ares, pres = pres)
    np.save(rfsoc.out_directory + f'fres{file_suffix}.npy', fres)

    # Gain Sweep
    filename = f's21_gain{file_suffix}.npy'
    npoints = npoints_gain
    bw = 10 * fine_bw
    rfsoc.target_sweep(filename, npoints = npoints, bandwidth = bw, N_accums = N_accums)

    # Fine Sweep
    filename = f's21_fine{file_suffix}.npy'
    npoints = npoints_fine
    bw = fine_bw
    rfsoc.target_sweep(filename,  npoints = npoints, bandwidth = bw, N_accums = N_accums)

    # Noise
    if noise_time is not None:
        if nnoise_timestreams == 1:
            filename = f'noise{file_suffix}_00.npy'
            rfsoc.capture_save_noise(noise_time, filename)
        else:
            for nindex in range(nnoise_timestreams):
                filename = f'noise{file_suffix}_{nindex:02d}.npy'
                rfsoc.capture_save_noise(noise_time, filename)

def optimize_ares(rfsoc, fres, ares, qres, fcal_indices, max_dbm = -50,
                  a_target = 0.5, n_iterations = 10, n_addonly = 3,
                  fine_bw = 0.2, fres_update_method = 'distance',
                  npoints_gain = 50, npoints_fine = 400, plot_directory = None,
                  verbose = False, N_accums = 5, cut_other_resonators = True):
    """
    Optimize tone powers using by iteratively fitting IQ loops and using a_nl
    of each fit to scale each tone power

    Parameters:
    rfsoc (citkid.primecam.instrument.RFSOC): RFSOC instance
    fres (np.array): array of center frequencies in Hz
    ares (np.array): array of amplitudes in RFSoC units
    qres (np.array): array of resonators Qs for cutting data. Resonances should
        span fres / qres
    fcal_indices (np.array): calibration tone indices
    max_dbm (float): maximum power per tone in dBm
    a_target (float): target value for a_nl. Must be in range (0, 0.77]
    n_iterations (int): total number of iterations
    n_addonly (int): number of iterations at the end to optimize using
        update_ares_addonly. Iterations before these use update_ares_pscale
    fine_bw (float): fine sweep bandwidth in MHz. See take_iq_noise
    fres_update_method (str): method for updating frequencies. See update_fres
    npoints_gain (int): number of points in the gain sweep
    npoints_fine (int): number of points in the fine sweep
    plot_directory (str or None): path to save histograms as the optimization is
        running. If None, doesn't save plots
    verbose (bool): if True, displays a progress bar of the iteration number
    N_accums (int): number of accumulations for the target sweeps
    """
    if plot_directory is not None:
        os.makedirs(plot_directory, exist_ok = True)
    fres, ares, qres = np.array(fres), np.array(ares), np.array(qres)
    pbar0 = list(range(n_iterations))
    if verbose:
        pbar0 = tqdm(pbar0, leave = False)
    fit_idx = [i for i in range(len(fres)) if i not in fcal_indices]
    a_max = get_rfsoc_power(max_dbm, np.mean(fres))
    a_nls = []
    for idx0 in pbar0:
        if verbose:
            pbar0.set_description('sweeping')
        file_suffix = f'{idx0:02d}'
        take_iq_noise(rfsoc, fres, ares, qres, fcal_indices, file_suffix,
                      noise_time = None, fine_bw = fine_bw,
                      take_rough_sweep = False, npoints_gain = npoints_gain,
                      npoints_fine = npoints_fine, N_accums = N_accums)
        # Fit IQ loops
        if verbose:
            pbar0.set_description('fitting')
        data =\
        fit_iq(rfsoc.out_directory, None, file_suffix, 0, 0, 0, 0, 0, plotq = False, verbose = False)
        a_nl = np.array(data.sort_values('dataIndex').iq_a, dtype = float)
        if len(a_nls):
            a_nl[a_nl == np.nan] = a_nls[-1][a_nl == np.nan]
        else:
            a_nl[a_nl == np.nan] = 2
        a_nls.append(a_nl)
        np.save(rfsoc.out_directory + f'a_nl_{file_suffix}.npy', a_nl)
        if plot_directory is not None:
            fig_hist, fig_opt = plot_ares_opt(a_nls, fcal_indices)
            save_fig(fig_hist, 'ares_hist', plot_directory)
            save_fig(fig_opt, 'ares_opt', plot_directory)
        # Update ares
        if idx0 <= n_addonly:
            ares[fit_idx] = update_ares_pscale(fres[fit_idx], ares[fit_idx],
                                           a_nl[fit_idx], a_target = a_target,
                                           a_max = a_max, dbm_change_high = 2,
                                           dbm_change_low = 2)
        else:
            ares[fit_idx] = update_ares_addonly(fres[fit_idx], ares[fit_idx],
                                                a_nl[fit_idx],
                                                a_target = a_target,
                                                a_max = a_max,
                                                dbm_change_high = 1,
                                                dbm_change_low = 1)
        # update fres
        f, i, q = np.load(rfsoc.out_directory + f's21_fine_{file_suffix}.npy')
        fres = update_fres(f, i + 1j * q, len(f) // len(fres),
                           fcal_indices = fcal_indices, method = fres_update_method,
                        cut_other_resonators=cut_other_resonators, fres = fres, qres = qres)
        # for the last iteration, save the updated ares list
        if idx0 == len(fres) - 1:
            np.save(rfsoc.out_directory + f'ares_{idx0 + 1:02d}', ares)

def optimize_ares_siq(rfsoc, fres, ares_min, ares_max, ares_cal, qres, fcal_indices, threshold,
                  n_iterations=10, fine_bw=0.2, fres_update_method='distance',
                  npoints_gain=50, npoints_fine=400, plot_directory=None,
                  verbose=False, N_accums=5, cut_other_resonators=True):
    """
    Optimize tone powers by looking at the gap in the IQ loops to determine bifurcation.

    The routine frequency-sweeps the current set of resonators, classifies each
    resonator as bifurcated or not, and updates ares by a
    binary-search procedure over multiple iterations. Diagnostic plots and
    updated fine-sweep frequencies can optionally be produced at each step.

    Parameters
    ----------
    rfsoc: citkid.primecam.instrument.RFSOC RFSOC instance
    fres : array-like
        Initial resonator center frequencies [Hz].
    ares_min : float
        Lower limit of the amplitude in RFSoC units.
    ares_max : float
        Upper limit of the amplitude in RFSoC units.
    ares_cal : float
        Fixed drive amplitude assigned to calibration tones in RFSoC units.
    qres : array-like
        Resonator quality-factor estimates. Should be the same size as fres.
    fcal_indices : sequence of int
        Indices of calibration tones that should not be optimized.
    threshold : float
        Threshold used by ``_is_bifurcated`` to classify each sweep.
    n_iterations : int, optional
        Number of optimization iterations to perform. Default value 10.
    fine_bw : float, optional
        Fine sweep bandwidth [MHz].
    fres_update_method : str or None, optional
        Method for updating frequencies. See update_fres.
    npoints_gain : int, optional
        number of points in the gain sweep
    npoints_fine : int, optional
        number of points in the fine sweep
    plot_directory : str or path-like, optional
        Output directory for diagnostic figures. If omitted, no plots are saved.
    verbose : bool, optional
        If ``True``, display a progress bar during the optimization.
    N_accums : int, optional
        number of accumulations for the target sweeps
    cut_other_resonators : bool, optional
        Forwarded to ``update_fres`` when frequency updates are enabled.

    Returns
    -------
    None
        Results are currently communicated through saved files and in-place
        workflow side effects rather than a structured return value.
    """
    import os
    from tqdm.auto import tqdm
    
    if (plot_directory is not None) and (not os.path.exists(plot_directory)):
        os.makedirs(plot_directory)

    fres, qres = np.array(fres), np.array(qres)
    pbar0 = list(range(n_iterations))
    if verbose:
        pbar0 = tqdm(pbar0, leave = False)
    fit_idx = [i for i in range(len(fres)) if i not in fcal_indices]

    ares = np.full_like(fres, fill_value=ares_min)
    
    ares[fcal_indices] = ares_cal
    
    theta_list = np.zeros_like(fres, dtype=float)
    
    ares_0_list = ares.copy()
    ares_1_list = np.full_like(fres, fill_value=ares_max)
    ares_1_list[fcal_indices] = ares_cal
    ares_history = np.ones(shape=(len(fres), n_iterations), dtype=float)
    state_history = np.zeros(shape=(len(fres), n_iterations), dtype=bool)       # True if bifurcated
    
    # Do a binary search in ares to set resonators just below bifurcation.
    for idx0 in pbar0:
        if verbose:
            pbar0.set_description('sweeping')
        file_suffix = f'{idx0:02d}'
        
        take_iq_noise(rfsoc, fres, ares, qres, fcal_indices, file_suffix,
                  noise_time=None, fine_bw=fine_bw,
                  take_rough_sweep=False, npoints_gain=npoints_gain,
                  npoints_fine=npoints_fine, N_accums=N_accums)
        np.save(rfsoc.out_directory + f'ares_{file_suffix}', ares)
        
        # Determining bifurcation
        if verbose:
            pbar0.set_description('bifurcated?')

        if idx0 == 0:
            f, I, Q = np.load(rfsoc.out_directory + f's21_fine_{file_suffix}.npy')

            N_tone = len(fres)
            N_per_tone = len(f) // N_tone

        _update_ares_for_iteration(
            idx0,
            rfsoc.out_directory + f's21_fine_{file_suffix}.npy',
            N_tone,
            N_per_tone,
            fcal_indices,
            ares,
            ares_max,
            ares_0_list,
            ares_1_list,
            theta_list,
            ares_history,
            state_history,
            threshold,
        )

        if plot_directory is not None:
            # christmas plot
            save_christmas_plot(idx0, state_history, fit_idx, plot_directory)
            # ares plot
            save_ares_history_plot(idx0, ares_history, fit_idx, plot_directory)

        # update fres
        if (fres_update_method != 'None') and (fres_update_method != None):
            f, i, q = np.load(rfsoc.out_directory + f's21_fine_{file_suffix}.npy')
            fres = update_fres(f, i + 1j * q, len(f) // len(fres),
                        fcal_indices=fcal_indices, method=fres_update_method,
                        cut_other_resonators=cut_other_resonators, fres=fres, qres=qres)
            
            if verbose:
                print("fres updated.")

    return

################################################################################
######################### Utility functions ####################################
################################################################################
def make_cal_tones(fres, ares, qres, max_n_tones = 1000,
                   resonator_indices = None,
                   new_resonator_indices_start = None):
    '''
    Adds calibration tones to the given resonator list. Fills in largest spaces
    between resonators, up to max_n_tones. If resonator_indices is provides,
    also creates a new list of resonator indices where the new calibration
    tones are labelled by sequential indices starting at
    new_resonator_indices_start if provided, or the length of the array if not

    Parameters:
    fres, ares, qres (np.array): frequency, amplitude, and Q arrays
    max_n_tones (int): maximum number of tones
    resonator_indices (array-like or None): resonator indices corresponding to
        fres
    new_resonator_indices_start (int or None): first index of the new resonator
        index labels, or None to start from the end of the array

    Returns:
    fres, ares, qres (np.array): frequency, amplitude, and Q arrays with
        calibration tones added
    fcal_indices (np.array): calibration tone indices
    new_resonator_indices (np.array): new resonator index list with the new
        calibration tones labelled sequentially starting at
        new_resonator_indices_start if provided, or the length of the array if
        not
    '''
    fres = np.asarray(fres, dtype = float)
    ares = np.asarray(ares, dtype = float)
    qres = np.asarray(qres, dtype = float)
    ix = np.argsort(fres)
    fres, ares, qres = fres[ix], ares[ix], qres[ix]

    if resonator_indices is not None and len(resonator_indices) != len(fres):
        raise ValueError('resonator_indices must be the same length as fres')
    if resonator_indices is None:
        resonator_indices = np.asarray(range(len(fres)))
    new_resonator_indices = np.asarray(resonator_indices, dtype = int)
    if new_resonator_indices_start is None:
        new_resonator_indices_start = max(resonator_indices) + 1

    ix = np.flip(np.argsort(np.diff(fres)))[:max_n_tones - len(fres)]
    fcal = np.sort([np.mean(fres[i:i+2]) for i in ix])
    fcal_indices = np.searchsorted(fres, fcal)
    fcal_indices += np.asarray(range(len(fcal_indices)), dtype = int)
    for fcal_index, fres_index in enumerate(fcal_indices):
        fres = np.insert(fres, fres_index, fcal[fcal_index])
        ares = np.insert(ares, fres_index, 260)
        qres = np.insert(qres, fres_index, np.inf)
        new_index = new_resonator_indices_start + fcal_index
        new_resonator_indices = np.insert(new_resonator_indices, fres_index, new_index)
    return fres, ares, qres, fcal_indices, new_resonator_indices

def _fit_circle_geometric(x, y):
    """
    Fit a circle to planar data using nonlinear least squares.

    Parameters
    ----------
    x, y : array-like
        Coordinates of the points to fit.

    Returns
    -------
    np.ndarray
        Best-fit ``[x_center, y_center, radius]``.
    """
    from scipy.optimize import least_squares
    x = np.asarray(x)
    y = np.asarray(y)

    def residuals(params):
        xc, yc, r = params
        return np.sqrt((x - xc)**2 + (y - yc)**2) - r

    # Initial guess
    x0 = [np.mean(x), np.mean(y), np.std(x)]

    result = least_squares(residuals, x0)
    return result.x

def _get_theta(f, I, Q):
    """
    Estimate the rotation angle needed to align an IQ loop.

    The input sweep is amplitude-normalized, phase-normalized, and corrected
    for an approximately linear phase slope before fitting the loop geometry.

    Parameters
    ----------
    f : array-like
        Fine sweep frequencies for one resonator.
    I, Q : array-like
        In-phase and quadrature sweep samples.

    Returns
    -------
    float
        Angle, in radians, used to rotate the loop into a standard frame.
    """
    S21 = I+1j*Q

    mag_lin = np.abs(S21)   # magnitude of S21
    phi = np.angle(S21)     # phase of S21

    # normalize to same baseline amplitude
    S21 /= np.mean(mag_lin[:20])

    # normalize to same baseline phase
    S21 *= np.exp(-1j * np.median(phi[:20]))
    
    # remove linear phase
    phi = np.angle(S21)
    phi_1 = phi[10]
    phi_2 = phi[-10]
    slope = (phi_2-phi_1)/(f[-10]-f[10])            # fit to line
    S21 *= np.exp(-1j * (slope*(f-f[10])+phi_1))
    
    I, Q = np.real(S21), np.imag(S21)
    phi = np.angle(S21)

    I_c, Q_c, _ = _fit_circle_geometric(I, Q)
    I_off = np.median(np.concat((I[:10], I[-10:])))
    Q_off = np.median(np.concat((Q[:10], Q[-10:])))
    theta = np.atan2(Q_off-Q_c, I_off-I_c)

    return theta

def _transform_to_unit_circle(f, I, Q, theta):
    """
    Normalize and rotate an IQ loop onto an approximate unit circle.

    The transformed loop is shifted so the off-resonance point lies near
    ``(1, 0)`` and the fitted circle is centered at the origin with unit radius.

    Parameters
    ----------
    f : array-like
        Fine sweep frequencies for one resonator.
    I, Q : array-like
        In-phase and quadrature sweep samples.
    theta : float
        Rotation angle returned by ``_get_theta``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Transformed I and Q coordinates on the normalized unit circle..
    """
    S21 = I+1j*Q

    mag_lin = np.abs(S21)   # magnitude of S21
    phi = np.angle(S21)     # phase of S21

    # normalize to same baseline amplitude
    S21 /= np.mean(mag_lin[:20])

    # normalize to same baseline phase
    S21 *= np.exp(-1j * np.median(phi[:20]))
    
    # remove linear phase
    phi = np.angle(S21)
    phi_1 = phi[10]
    phi_2 = phi[-10]
    slope = (phi_2-phi_1)/(f[-10]-f[10])            # fit to line
    S21 *= np.exp(-1j * (slope*(f-f[10])+phi_1))
    
    I, Q = np.real(S21), np.imag(S21)
    phi = np.angle(S21)

    # rotate the circle around (1, 0) so that center is on real axis and off-resonance is at (1,0)
    I_off = np.median(np.concat((I[:10], I[-10:])))
    Q_off = np.median(np.concat((Q[:10], Q[-10:])))
    I -= I_off
    Q -= Q_off
    
    S21 = I + 1j*Q
    S21 *= np.exp(-1j*theta)    # rotate so that on-resonance is on real axis
    I = np.real(S21) + 1        # shift off-resonance back to (1, 0)
    Q = np.imag(S21)
    S21 = I + 1j*Q

    # using algebra to find the center
    I1 = 1
    Q1 = 0
    I2 = I[np.argmin(S21)]
    Q2 = Q[np.argmin(S21)]
    Ic = 0.5*(I1+I2) + 0.5*((Q2-Q1)**2)/(I2-I1) # center of the circle
    Qc = 0                                      # center of the circle
    radius = I1-Ic

    # shift the center to (0, 0)
    I -= Ic
    Q -= Qc
    # normalize the circle to unit circle
    I /= radius
    Q /= radius
    
    return I, Q

def _is_bifurcated(f, I, Q, theta, threshold, medfilter_size=51):
    """
    Detect whether a resonator sweep appears bifurcated.

    The decision is based on the frequency derivative of the normalized IQ loop
    relative to a median-filtered baseline.

    Parameters
    ----------
    f : array-like
        Fine sweep frequencies for one resonator.
    I, Q : array-like
        In-phase and quadrature sweep samples.
    theta : float
        Rotation angle used to normalize the loop geometry.
    threshold : float
        Detection threshold applied to the filtered derivative ratio.
    medfilter_size : int, optional
        Median filter window used to estimate the baseline derivative scale.

    Returns
    -------
    bool
        ``True`` if the resonator is classified as bifurcated.
    """
    from scipy.ndimage import median_filter

    I, Q = _transform_to_unit_circle(f, I, Q, theta)
    
    # find the distance between to points in the IQ circle
    df = f[1]-f[0]
    dI = np.diff(I)/df
    dQ = np.diff(Q)/df
    dIQ = np.sqrt(dI**2 + dQ**2)
    
    riq = dIQ/median_filter(dIQ, medfilter_size)

    return np.max(riq) > threshold


def _update_ares_for_iteration(idx0, file_path, N_tone, N_per_tone, fcal_indices,
                               ares, ares_max, ares_0_list, ares_1_list,
                               theta_list, ares_history, state_history, threshold):
    """
    Update resonator states and next-step drive amplitudes for one iteration.

    Parameters
    ----------
    idx0 : int
        Current iteration index.
    file_path : str or path-like
        Path to the saved fine sweep file for this iteration.
    N_tone : int
        Total number of tones in the sweep.
    N_per_tone : int
        Number of fine sweep samples stored for each tone.
    fcal_indices : sequence of int
        Indices of calibration tones that should be skipped.
    ares : np.ndarray
        Current resonator drive amplitudes. Updated in place.
    ares_max : float
        Maximum allowed resonator drive amplitude.
    ares_0_list, ares_1_list : np.ndarray
        Lower and upper amplitude bounds used by the search.
    theta_list : np.ndarray
        Cached rotation angles for each resonator.
    ares_history : np.ndarray
        History array populated in place with tested amplitudes.
    state_history : np.ndarray
        History array populated in place with bifurcation states.
    threshold : float
        Bifurcation detection threshold.
    """
    f, I, Q = np.load(file_path)
    for res_idx in range(N_tone):
        if res_idx in fcal_indices:
            continue

        i_beg = N_per_tone*res_idx
        i_end = N_per_tone*(res_idx+1)
        f_fine = f[i_beg:i_end]
        I_fine = I[i_beg:i_end]
        Q_fine = Q[i_beg:i_end]

        ares_history[res_idx, idx0] = ares[res_idx]

        if idx0 == 0:
            theta_list[res_idx] = _get_theta(f_fine, I_fine, Q_fine)

        state = _is_bifurcated(f_fine, I_fine, Q_fine, theta_list[res_idx], threshold=threshold)
        state_history[res_idx, idx0] = state

        if idx0 == 0:
            ares[res_idx] = ares_max
        elif idx0 == 1:
            ares[res_idx] = np.sqrt(ares_0_list[res_idx] * ares_1_list[res_idx])
        else:
            if state:
                ares_1_list[res_idx] = ares[res_idx]
            else:
                ares_0_list[res_idx] = ares[res_idx]
            ares[res_idx] = np.sqrt(ares_0_list[res_idx] * ares_1_list[res_idx])