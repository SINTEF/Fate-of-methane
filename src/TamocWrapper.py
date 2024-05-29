#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import xarray as xr
from contextlib import redirect_stdout

from tamoc import dbm
from tamoc import ambient
from tamoc import seawater
from tamoc import dispersed_phases
from tamoc import single_bubble_model



####################################################################
#### Functions to get ambient profile and mass transfer coefficient ####
####################################################################



def setup_ambient_profile(z, T, S, oxygen=None):
    '''
    Set up a profile representing ambient conditions in tamoc.

    Parameters
    ----------
    z : ndarray
        1D array of depth coordinates, going downwards from surface
    T : ndarray
        1D array of temperature values at z depths, in degrees Celsius
    S : ndarray
        1D array of salinity values at z depths, in PSU
    oxygen : ndarray, optional
        1D array of dissolved oxygen concentration at z depths, in kg/m^3
    '''

    # Create an xarray dataset and add data provided
    data = xr.Dataset()
    data.coords['z'] = z
    data.coords['z'].attrs = {'units' : 'm'}
    data['temperature'] = (('z'), T)
    data['temperature'].attrs = {'units' : 'deg C'}
    data['salinity'] = (('z'), S)
    data['salinity'].attrs = {'units' : 'psu'}

    # Compute pressure by integrating density
    # Note that compute pressure requires temperature in Kelvin
    T_kelvin, _ = ambient.convert_units(T, 'deg C')
    P = ambient.compute_pressure(z, T_kelvin, S, 0)
    data['pressure'] = (('z'), P)
    data['pressure'].attrs = {'units' : 'Pa'}

    # Add oxygen if provided, and create profile object
    if oxygen is None:
        profile = ambient.Profile(data)
    else:
        data['oxygen']  = (('z'), oxygen)
        data['oxygen'].attrs = {'units' : 'kg/m^3'}
        profile = ambient.Profile(data, chem_names=['oxygen'], chem_units=['kg/m^3'])

    # Add ambient dissolved gas concentrations
    # Calculated by assuming equilibrium with atmosphere
    # taking pressure at depth into account
    # Note that oxygen will not be overwritten if already present
    profile.add_computed_gas_concentrations()

    return profile


def get_kw(windspeed, variance, Sc = 660):
    '''
    Get mass transfer coefficient based on empirical expression
    from Najjar & Orr (1998).

    Parameters
    ----------
    windspeed : float
        Wind speed at 10 m above sea level, in m/s
    variance : float
        Variance of the wind speed, not entirely sure what
        to do about this one.
    Sc : float
        Schmidt number.

    Returns
    -------
    kw : float
        The mass transfer coefficient, in m/s.
    '''
    a = 0.336 # Empirical parameter, in crazy units of (cm/h)*(s^2/m^2)
    # Dividing by 360000 converts from cm/hour to m/second
    kw = (a/360000) *(windspeed**2 + variance)*np.sqrt(660/Sc)
    return kw


########################################################
#### Helper function for running a Tamoc simulation ####
########################################################

def run_tamoc_sbm(z0, d0, profile, silence = True, delta_t=0.1, t_hyd=None):
    '''
    Runs a Tamoc simulation with the single-bubble model,
    for a given case, season, initial depth and initial bubble size.
    The input data must be prepared ahead of time.

    Parameters
    ----------
    z0 : float
        Release depth, in meters.
    d0 : float
        Initial bubble diameter, in meters.
    profile : tamoc.ambient.Profile
        Profile object from tamoc describing ambient conditions.
        Can be created with setup_ambient() function defined above.
    silence : bool, optional
        Tamoc is hardcoded to print output when running simulations.
        If silence == True, then capture and hide this output.
    delta_t : float, optional
        Output interval from tamoc (passed to scipy integrator)
    return_radius : bool, optional
        If True, return the bubble radius development

    Returns
    -------
    depth : ndarray
        1D array with a list of depths, in meters.
    deposited : ndarray
        1D array with giving the fraction of the initial mass of the
        bubble that was dissolved into the water at each of the depths.
    direct : float
        Fraction of original methane that directly reached the atmosphere with the bubble
    '''


    # Initialize a single_bubble_model.Model object with the ambient data
    sbm = single_bubble_model.Model(profile)

    # Create a light gas bubble to track
    # Note that we include nitrogen and oxygen even though the bubble
    # is initially pure methane. This is because we need the simulation
    # to track nitrogen and oxygen that enters the bubble from the
    # ambient water.
    composition = ['methane', 'nitrogen', 'oxygen']
    bub = dbm.FluidParticle(composition, fp_type=0.)

    # Set the mole fractions of each component at release.
    # Order is the same as composition above.
    mol_frac = np.array([1.0, 0.0, 0.0])


    # Get the temperature at the release depth
    # (we assume bubble has ambient temperature)
    T0 = profile.get_values(z0, ['temperature'])

    # When this much mass remains, terminate the simulation.
    fdis = 1.e-12

    # Also, use the hydrate model from Jun et al. (2015) to set the
    # hydrate shell formation time, unless provided
    if t_hyd is None:
        # Get the pressure at the release depth.
        P = profile.get_values(z0, 'pressure')
        m = bub.masses_by_diameter(d0, T0, P, mol_frac)
        t_hyd = dispersed_phases.hydrate_formation_time(bub, z0, m, T0, profile)

    # Simulate the bubble rising through the water column.
    # Use context manager to prevent Tamoc from printing output if silence is True
    if silence:
        # Create a context manager that redirects stdout to /dev/null
        context_manager = redirect_stdout(open(os.devnull, 'w'))
    else:
        # Create a context manager that redirects stdout to stdout
        # (i.e., it does nothing)
        context_manager = redirect_stdout(sys.stdout)

    with context_manager:
        sbm.simulate(bub, z0, d0, mol_frac, T0, K_T=1, fdis=fdis,
                t_hyd=t_hyd, delta_t=delta_t)

    return bub, sbm


def parse_tamoc_sbm_results(sbm, z0):

    # The results of the simulation are stored as an array in sbm.y.
    # Depth at each step is stored as sbm.y[:,2].
    depth = sbm.y[:,2]
    # For some reason, the first two points are identical, so
    # change the first point to the actual release depth
    depth[0] = z0
    # Calculate remaining fraction in bubble at each step.
    fraction = sbm.y[:,3] / sbm.y[0,3]
    # For some reason, the two last points may also, in some cases,
    # be identical, and this will also lead to trouble below.
    # Deal with this here.
    if depth[-2] == depth[-1]:
        # Reduce the length of the depth array by 1, by lopping off the last point.
        depth = depth[:-1]
        # Reduce the length of fraction array by 1, but keep the last value.
        tmp = fraction[-1]
        fraction = fraction[:-1]
        fraction[-1] = tmp

    # Take difference to get deposited fraction at each step,
    # and divide by length of each step to get concentration.
    deposited = (fraction[1:] - fraction[:-1]) / (depth[1:] - depth[:-1])

    # Drop the last (upper) point in depth, to get same number
    # of points in both arrays.
    depth = depth[:-1]

    # Calculate the amount that was transported directly to atmosphere:
    if depth[-1] < 1:
        direct = sbm.y[-1,3]/sbm.y[0,3]
    else:
        direct = 0.0

    return depth, fraction, deposited, direct


def get_bubble_diameter(bub, sbm, profile, dirty_size=None):

    z = sbm.y[:,2]
    d = np.zeros_like(z)

    for i in range(len(z)):
        m = sbm.y[i,3:-1]
        # Get the ambient profile data
        Ta, Sa, P = profile.get_values(z[i], ['temperature', 'salinity', 'pressure'])
        d[i] = bub.diameter(m, Ta, P)

    if dirty_size is not None:
        t_dirty = sbm.t[np.argmin(np.abs(d - dirty_size))]
        return z, d, t_dirty
    else:
        return z, d
