#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import numpy as np
from tqdm import trange
from scipy.interpolate import interp1d

from DiffusionSolver import diffusion_solver
from TamocWrapper import setup_ambient_profile, run_tamoc_sbm, parse_tamoc_sbm_results

############################################################
#### Helper function for running an Eulerian simulation ####
############################################################

def run_eulerian(depth, deposited, direct, z, K, halflife, kw, Tmax, K_times=None, kw_times=None, dt=3600, t0=0, loop=True, threshold=1e-7):
    '''
    Runs a simulation with the diffusion-reaction model, using
    results from Tamoc as initial conditions.

    Parameters
    ----------
    depth : ndarray
        1D array of depths, output from Tamoc. Does not have to
        be equidistant, as this will not be used in the solver.
    deposited : ndarray
        1D array of initial dissolved fraction at those depths. Will
        be interpolated onto the equidistant grid used in the solver.
    zc : ndarray
        1D array with z grid to be used by Eulerian solver.
    K : ndarray
        1D array of diffusivity values, at positions given by zc.
        or
        2D array of diffusivity values, at positions given by zc and times given by K_times.
    halflife : float
        The biodegradation half-life, in seconds.
    kw : float
        The mass transfer coefficient at the surface, in m/s.
    Tmax : float
        Duration of simulation, in seconds.
    K_times : ndarray
        1D array of times for when K values are valid, given in seconds since simulation start.
    dt : float
        Timestep, in seconds.

    Returns
    -------
    C : ndarray
        2D array giving concentrations as function of position and time.
    evap : ndarray
        1D array giving evaporated amount as a function of time.
    biod : ndarray
        1D array giving biodegraded amount as a function of time.
    tc : ndarray
        1D array defining the times used in the model.
    '''

    # Check for consistency of input
    if K_times is not None:
        assert len(K.shape) == 2
    else:
        assert len(K.shape) == 1
    if kw_times is not None:
        assert len(kw) == len(kw_times)


    # Interpolate the results of the Tamoc simulation
    # onto the grid for the Eulerian solver.
    C0 = interp1d(depth[:], deposited[:], fill_value='extrapolate', bounds_error = False, kind='linear')(z)
    # Normalise the initial concentration to be equal to dissolved fraction
    dz = z[1] - z[0] # Assuming fixed cell size
    C0 = (1-direct) * C0 / np.sum(C0*dz)

    # Calculate biodegradation rate from half-life
    lifetime = halflife/np.log(2)
    Q = 1/lifetime

    # Run simulation
    C, evap, biod = diffusion_solver(z, C0, K, Tmax, dt, kw = kw, Q = Q, K_times=K_times, kw_times=kw_times, t0=t0, loop=loop, threshold=threshold)

    # Create list of timesteps, for convenience in plotting, etc.
    tc = np.linspace(t0, Tmax, C.shape[0])

    return C, evap, biod, tc


##########################################################
#### Functions for running an ensemble of simulations ####
##########################################################


def run_one_year(z0, d0, t0, halflife, kw, zc_input, T_input, S_input, zf_input, K_input, K_times, kw_times=None, dt=3600, Nz=None, additional_output=False):
    '''
    Function to run a simulation for one year, starting with a
    Tamoc simulation, and then one year of diffusion-reaction.

    Parameters
    ----------
    case : string
        Name of the case, must match .nc file with ambient data.
    z0 : float
        Bubble release depth, in meters.
    d0 : float
        Initial bubble diameter, at release depth, in meters.
    halflife : float
        Biodegradation half-life, in seconds.
    kw : float
        Mass transfer coefficient, in m/s.
    startday : int
        Day of the year (number of days after January 1) at which to start simulation.
    datafolder : string
        Path to the .nc files with ambient data.
    Nz : int
        Number of points in z-axis discretisation for the Eulerian solver.
    dt : float
        Timestep to use in the Eulerian solver.

    Returns
    -------
    direct : ndarray
        1D array containing fraction of released methane that was
        transported directly to atmosphere with bubble. Constant in time,
        but provided as a vector for plotting convenience later.
    dissolved : ndarray
        1D array containing fraction of released methane that remains
        dissolved, as a function of time.
    biodegraded : ndarray
        1D array containing fraction of released methane that was biodegraded,
        as a function of time.
    evaporated : ndarray
        1D array containing fraction of released methane that escaped to the
        atmosphere via mass transfer at the surface, as a function of time.
    timestamps : ndarray
        1D array containing time coordinates for the four other vectors.
    '''

    # Set up grid and so on
    if Nz is None:
        # use grid from input
        zc = zc_input.copy()
        # Assuming fixed cell size
        dz = np.abs(zc[1:] - zc[:-1])
        assert (np.amax(dz) - np.amin(dz)) < (np.amin(dz) * 1e-3)
        dz = np.mean(dz)
    else:
        # Create grid for Eulerian diffusion-reaction solver
        zf, dz = np.linspace(0, z0, Nz+1, retstep=True)
        zc = zf[:-1] + dz/2
    # End of simulation, in seconds
    # Upping to twenty years
    Tmax = t0 + 20 * 365 * (24*3600)
    # Number of timesteps
    Nt = int(Tmax / dt)
    # Biodegradation rate, converted from half-life
    Q = np.log(2) / halflife

    # Get ambient profile for Tamoc
    it = np.searchsorted(K_times, t0)
    profile = setup_ambient_profile(zc_input, T_input[it,:], S_input[it,:])
    # Hydrate formation time. inf => clean bubble
    t_hyd = np.inf
    # Run Tamoc to get initial conditions
    bub, sbm = run_tamoc_sbm(z0, d0, profile, t_hyd=t_hyd)
    depth, remaining, deposited, direct = parse_tamoc_sbm_results(sbm, z0)

    t = t0 # Variable to keep track of time
    C, evaporated, biodegraded, timestamps = run_eulerian(depth, deposited, direct, zc, K_input, halflife, kw, Tmax, kw_times=kw_times, dt=dt, t0=t0, K_times=K_times, loop=True)

    # Integrate C to get dissolved fraction
    dissolved = np.sum(dz * C, axis=1)
    # Convert direct bubble transfer to an array for consistency with the others,
    # even though it is constant in time
    direct = direct * np.ones_like(timestamps)

    if additional_output:
        return direct, dissolved, biodegraded, evaporated, timestamps, C, deposited, depth
    else:
        return direct, dissolved, biodegraded, evaporated, timestamps


def run_ensemble(z0, d0, halflife, kw, Nruns, zc_input, T_input, S_input, zf_input, K_input, K_times, kw_times=None, Nz=None, dt=3600, progressbar=True):
    '''
    Function to run Nruns simulations, with evenly spaced startdays throughout a year.

    Parameters
    ----------
    case : string
        Name of the case, must match .nc file with ambient data.
    z0 : float
        Bubble release depth, in meters.
    d0 : float
        Initial bubble diameter, at release depth, in meters.
    halflife : float
        Biodegradation half-life, in seconds.
    kw : float
        Mass transfer coefficient, in m/s.
    Nruns : int
        Number of simulations to run (will be evenly spaced throughout a year).
    datafolder : string
        Path to the .nc files with ambient data.
    Nz : int
        Number of points in z-axis discretisation for the Eulerian solver.
    dt : float
        Timestep to use in the Eulerian solver.

    Returns
    -------
    direct : ndarray
        1D array containing fraction of released methane that was
        transported directly to atmosphere with bubble. Constant in time,
        but provided as a vector for plotting convenience later.
    dissolved : ndarray
        1D array containing fraction of released methane that remains
        dissolved, as a function of time.
    biodegraded : ndarray
        1D array containing fraction of released methane that was biodegraded,
        as a function of time.
    evaporated : ndarray
        1D array containing fraction of released methane that escaped to the
        atmosphere via mass transfer at the surface, as a function of time.
    tc : ndarray
        1D array containing time coordinates for the four other vectors.
    '''

    # Some checks on inputs
    if isinstance(kw, np.ndarray):
        assert len(kw) == len(kw_times)

    # Use progress bar if indicated.
    if progressbar:
        iterator = trange
    else:
        iterator = range

    # Lists to store results
    direct_list = []
    dissolved_list = []
    biodegraded_list = []
    evaporated_list = []
    tc_list = []
    # Loop over runs
    for i in iterator(Nruns):
        t0 = i * (3600*24*365/Nruns)
        direct, dissolved, biodegraded, evaporated, tc = run_one_year(z0, d0, t0, halflife, kw, zc_input, T_input, S_input, zf_input, K_input, K_times, dt=dt, Nz=Nz, kw_times=kw_times)
        direct_list.append(direct)
        dissolved_list.append(dissolved)
        biodegraded_list.append(biodegraded)
        evaporated_list.append(evaporated)
        tc_list.append(tc)
        # Timestamps will be the same for all simulations
        # (always starts at 0)

    # Convert to arrays and return
    direct = np.array(direct_list)
    dissolved = np.array(dissolved_list)
    biodegraded = np.array(biodegraded_list)
    evaporated = np.array(evaporated_list)
    tc = np.array(tc_list)

    return direct, dissolved, biodegraded, evaporated, tc
