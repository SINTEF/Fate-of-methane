#!/usr/bin/env python
# -*- coding: utf-8 -*-



import numpy as np
from scipy.interpolate import interp1d

from DiffusionSolver import diffusion_solver

############################################################
#### Helper function for running an Eulerian simulation ####
############################################################

def run_eulerian(depth, deposited, direct, z, K, halflife, kw, Tmax, K_times = None, dt = 3600, FVM=False, reaction_term_flux=True):
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

    # Interpolate the results of the Tamoc simulation
    # onto the grid for the Eulerian solver.
    C0 = interp1d(depth[:], deposited[:], fill_value = 0.0, bounds_error = False, kind='cubic')(z)
    # Normalise the initial concentration to be equal to dissolved fraction
    dz = z[1] - z[0] # Assuming fixed cell size
    C0 = (1-direct) * C0 / np.sum(C0*dz)

    # Calculate biodegradation rate from half-life
    lifetime = halflife/np.log(2)
    Q = 1/lifetime

    # Run simulation
    C, evap, biod = diffusion_solver(z, C0, K, Tmax, dt, kw = kw, Q = Q, K_times = K_times)

    # Create list of timesteps, for convenience in plotting, etc.
    tc = np.linspace(0, Tmax, C.shape[0])

    return C, evap, biod, tc


##########################################################
#### Functions for running an ensemble of simulations ####
##########################################################


def run_one_year(case, z0, d0, halflife, kw, startday = 0, datafolder = os.path.join('..', 'data'), Nz = 1001, dt = 3600):
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
    # Creat grid for Eulerian diffusion-reaction solver
    zc = np.linspace(0, z0, Nz)
    # Simulation duration, in seconds
    Tmax = 365 * (24*3600)
    # Biodegradation rate, converted from half-life
    Q = np.log(2) / halflife

    # We're operating with a simplified year, of 365 days,
    # with 183 days of summer, starting on April 1,
    # and 182 days of winter, starting on October 1.
    # A simulation will start on the given day of the year,
    # run for the remainder of the current season, then switch
    # to the other season, and finally switch back and run
    # until 365 days is reached.
    startday = int(startday)
    assert (0 <= startday) and (startday < 365)
    if (startday < 90) or (273 < startday):
        seasons = ['winter', 'summer', 'winter']
        # Simulation times, in seconds, for the different seasons
        runtimes = (24*3600)*np.array([(90 - startday)%365, 183, (182 - (90 - startday)) % 365])
    else:
        seasons = ['summer', 'winter', 'summer']
        runtimes = (24*3600)*np.array([273 - startday, 182, 183 - (273 - startday)])
    assert sum(runtimes) == 365 * (24*3600)

    # Run Tamoc to get initial conditions
    depth, deposited = run_tamoc(case, seasons[0], z0, d0)
    # Take note of direct bubble transfer to surface
    direct = 1 - simps(deposited[::-1], x = depth[::-1])
    # Arrays with only one element, will be concatenated later
    dissolved = np.array([1.0 - direct])
    biodegraded = np.array([0.0])
    evaporated = np.array([0.0])
    timestamps = np.array([0.0])

    # Run diffusion-reaction simulations in order, until 365 days is reached,
    # providing the results of one simulation as input to the next.
    t = 0 # Variable to keep track of time
    for season, runtime in zip(seasons, runtimes):
        # Get the diffusivity profile
        K  = get_diffusivity(case, season, zc)
        if t < Tmax:
            t += runtime
            C, evap, biod, tc = run_eulerian(depth, deposited, zc, K, halflife, kw, runtime, dt = dt)
            # Concatenating arrays, skipping first element of next array,
            # which is equal to last element of previous array.
            dissolved = np.concatenate((dissolved, simps(C, x = zc, axis = 1)[1:]))
            # These three will start again at 0 each time, so add the last value
            # in the previous simulation.
            biodegraded = np.concatenate((biodegraded, biod[1:] + biodegraded[-1]))
            evaporated = np.concatenate((evaporated, evap[1:] + evaporated[-1]))
            timestamps = np.concatenate((timestamps, tc[1:] + timestamps[-1]))
            # Use last output as input next time around
            depth = zc
            deposited = C[-1,:]

    # Convert direct bubble transfer to an array for consistency with the others,
    # even though it is constant in time
    direct = direct * np.ones_like(timestamps)

    return direct, dissolved, biodegraded, evaporated, timestamps


def run_ensemble(case, z0, d0, halflife, kw, Nruns, datafolder = os.path.join('..', 'data'), Nz = 1001, dt = 3600, progressbar = True):
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
    # Loop over runs
    for i in iterator(Nruns):
        startday = int(365*i/Nruns)
        direct, dissolved, biodegraded, evaporated, tc = run_one_year(case, z0, d0, halflife, kw, startday = startday, Nz = Nz, dt = dt)
        direct_list.append(direct)
        dissolved_list.append(dissolved)
        biodegraded_list.append(biodegraded)
        evaporated_list.append(evaporated)
        # Timestamps will be the same for all simulations
        # (always starts at 0)

    # Convert to arrays and return
    direct = np.array(direct_list)
    dissolved = np.array(dissolved_list)
    biodegraded = np.array(biodegraded_list)
    evaporated = np.array(evaporated_list)

    return direct, dissolved, biodegraded, evaporated, tc
