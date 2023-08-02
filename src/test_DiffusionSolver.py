#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import pytest
import numpy as np
from scipy.stats import norm

from DiffusionSolver import diffusion_solver


def test_constant_diffusion():
    '''
    Test that the variance of the initial distribution increases
    linearly in time as expected.
    Constant diffusivity, zero biodegradation and zero mass transfer.
    '''
    # Number of points
    Nz = 1001
    # Max depth
    Zmax = 1
    # Max time
    Tmax = 2
    # Timestep
    dt = 0.01

    # Cell faces, cell spacing
    Zf, dz = np.linspace(0, Zmax, Nz, retstep=True)
    # Cell centers
    Zc = Zf[:-1] + dz/2

    # Diffusivity
    K0 = 1e-3
    K = K0 * np.ones_like(Zf)

    # Initial concentration (two Gaussians of different height and width)
    C0 = (2*norm(loc = Zmax/2 - Zmax/10, scale = Zmax/25).pdf(Zc) + 3*norm(loc = Zmax/2 + Zmax/20, scale = Zmax/15).pdf(Zc) ) / 5
    # Normalise, using midpoint method since we have concentration at cell centers
    C0 = C0 / np.sum(dz*C0)

    # Run calculation
    C, evap, biod = diffusion_solver(Zc, C0, K, Tmax, dt)

    # Calculate variance, as a function of time
    var = np.sum(dz*C*(Zc - Zmax/2)**2, axis=1)

    # Assert that variance increases as expected, starting
    # from the value at t = 0
    t = np.linspace(0, Tmax, C.shape[0])
    difference = (var[0] + 2*K0*t) - var
    assert difference == pytest.approx(0.0, abs=1e-6)

    # Assert that there is zero evaporation and biodegradation
    assert evap == pytest.approx(0.0, abs=1e-12)
    assert biod == pytest.approx(0.0, abs=1e-12)


def test_steady_state():
    '''
    Test that mass is evenly distributed, and conserved, after a long time.
    Non-constant diffusivity, zero biodegradation and zero mass transfer.
    '''
    # Number of points
    Nz = 1001
    # Max depth
    Zmax = 1
    # Max time
    Tmax = 20
    # Timestep
    dt = 0.01


    # Cell faces, cell spacing
    Zf, dz = np.linspace(0, Zmax, Nz, retstep=True)
    # Cell centers
    Zc = Zf[:-1] + dz/2

    # Non-constant diffusivity
    K0 = 1e-2
    K1 = 1e-1
    K = K0 + K1 * np.sin(np.pi*Zf)

    # Initial concentration (two Gaussians of different height and width)
    C0 = (2*norm(loc = Zmax/2 - Zmax/10, scale = Zmax/25).pdf(Zc) + 3*norm(loc = Zmax/2 + Zmax/20, scale = Zmax/15).pdf(Zc) ) / 5
    # Normalise, using midpoint method since we have concentration at cell centers
    C0 = C0 / np.sum(dz*C0)

    # Run calculation
    C, evap, biod = diffusion_solver(Zc, C0, K, Tmax, dt)

    # Assert that mass is conserved
    # Using midpoint method since we have concentration at cell centers
    mass = np.sum(dz*(C[-1,:]))
    assert mass == pytest.approx(1, abs=1e-8)

    # Assert that the concentration is constant
    difference = np.amax(C[-1,:]) - np.amin(C[-1,:])
    assert difference == pytest.approx(0.0, abs=1e-7)

    # Assert that there is zero evaporation and biodegradation
    assert evap == pytest.approx(0.0, abs=1e-12)
    assert biod == pytest.approx(0.0, abs=1e-12)


def test_biodegradation():
    '''
    Test that biodegradation happens as fast as expected, and that
    the mass is successfully accounted for.
    '''
    # Number of points
    Nz = 1001
    # Max depth
    Zmax = 1
    # Max time
    Tmax = 20
    # Timestep
    dt = 0.1
    # Biodegradation rate
    halflife = 5
    Q = np.log(2) / halflife


    # Cell faces, cell spacing
    Zf, dz = np.linspace(0, Zmax, Nz, retstep=True)
    # Cell centers
    Zc = Zf[:-1] + dz/2

    # Non-constant diffusivity
    K0 = 1e-3
    K1 = 1e-2
    K = K0 + K1 * np.sin(np.pi*Zf)

    # Initial concentration (two Gaussians of different height and width)
    C0 = (2*norm(loc = Zmax/2 - Zmax/10, scale = Zmax/25).pdf(Zc) + 3*norm(loc = Zmax/2 + Zmax/20, scale = Zmax/15).pdf(Zc) ) / 5
    # Normalise, using midpoint method since we have concentration at cell centers
    C0 = C0 / np.sum(dz*C0)

    # Run calculation
    C, evap, biod = diffusion_solver(Zc, C0, K, Tmax, dt, Q = Q)

    # Assert that half the mass is biodegraded after t = halflife
    it = int(halflife / dt)
    mass = np.sum(dz*C[it,:])
    assert mass == pytest.approx(0.5, 2e-5)

    # Assert that three quarters of the mass is biodegraded after t = 2*halflife
    it = int(2*halflife / dt)
    mass = np.sum(dz*C[it,:])
    assert mass == pytest.approx(0.25, 5e-5)

    # Assert that mass is accounted for, by adding up remaining and biodegraded
    mass = np.sum(dz*C[-1,:]) + biod[-1]
    assert mass == pytest.approx(1, 1e-5)

    # Assert that there is zero evaporation
    assert evap == pytest.approx(0.0, abs=1e-12)


def test_mass_transfer():
    '''
    Test that mass transfer happens as fast as expected, and that
    the mass is successfully accounted for. This test case uses an
    analytical solution that is approximately valid when Bi << 1. See
    en.wikipedia.org/wiki/Biot_number#Mass_transfer_analogue
    and
    physics.stackexchange.com/questions/599628/cases-for-reasoning-about-mass-transfer
    for further details.
    '''
    # Number of points
    Nz = 1201
    # Max depth
    Zmax = 1.2

    # Cell faces, cell spacing
    Zf, dz = np.linspace(0, Zmax, Nz, retstep=True)
    # Cell centers
    Zc = Zf[:-1] + dz/2

    # Max time
    Tmax = 20
    # Timestep
    dt = 0.1
    # Mass transfer coefficient
    kw = 0.003
    # Constant diffusivity
    K0 = 1e-1
    K = K0 * np.ones_like(Zf)

    # Initial concentration, constant in this case.
    C0 = np.ones_like(Zc) * (1/Zmax)

    # Run calculation
    C, evap, biod = diffusion_solver(Zc, C0, K, Tmax, dt, kw = kw)

    # Compare to analytical solution
    modelled_mass = np.sum(dz*C[-1,:])
    analytical_mass = np.exp(-Tmax*kw/Zmax)
    assert modelled_mass == pytest.approx(analytical_mass, abs=1e-3)

    # Assert that the remaining mass and the evaporated mass sum to 1
    assert modelled_mass + evap[-1] == pytest.approx(1.0, abs=1e-8)


    # One more test, with another mass transfer coefficient and another diffusivity
    kw = 0.023
    # Constant diffusivity
    K0 = 3e-1
    K = K0 * np.ones_like(Zf)

    # Run calculation
    C, evap, biod = diffusion_solver(Zc, C0, K, Tmax, dt, kw = kw)

    # Compare to analytical solution
    modelled_mass = np.sum(dz*C[-1,:])
    analytical_mass = np.exp(-Tmax*kw/Zmax)
    assert modelled_mass == pytest.approx(analytical_mass, abs=1e-2)

    # Assert that the remaining mass and the evaporated mass sum to 1
    assert modelled_mass + evap[-1] == pytest.approx(1.0, abs=1e-8)


    # One more test, with another mass transfer coefficient and non-constant diffusivity
    kw = 0.0023
    # Non-constant diffusivity
    K0 = 5e-2
    K1 = 5e-1
    K = K0  + K1*np.sin(np.pi*Zf/Zmax)

    # Run calculation
    C, evap, biod = diffusion_solver(Zc, C0, K, Tmax, dt, kw = kw)

    # Compare to analytical solution
    modelled_mass = np.sum(dz*C[-1,:])
    analytical_mass = np.exp(-Tmax*kw/Zmax)
    assert modelled_mass == pytest.approx(analytical_mass, abs=1e-3)

    # Assert that the remaining mass and the evaporated mass sum to 1
    assert modelled_mass + evap[-1] == pytest.approx(1.0, abs=1e-8)


def test_mass_conservation():
    '''
    Test that the remaining, evaporated and biodegraded mass sums to 1.
    '''

    # Number of points
    Nz = 1001
    # Max depth
    Zmax = 1.3

    # Cell faces, cell spacing
    Zf, dz = np.linspace(0, Zmax, Nz, retstep=True)
    # Cell centers
    Zc = Zf[:-1] + dz/2

    # Max time
    Tmax = 12
    # Timestep
    dt = 0.05
    # Biodegradation rate
    halflife = 7
    Q = np.log(2)/halflife
    # Mass transfer coefficient
    kw = 0.003
    # Non-constant diffusivity
    K0 = 1e-3
    K1 = 1e-2
    K = K0 + K1 * np.sin(np.pi*Zf/Zmax)

    # Initial concentration (two Gaussians of different height and width)
    C0 = (2*norm(loc = Zmax/2 - Zmax/10, scale = Zmax/25).pdf(Zc) + 3*norm(loc = Zmax/2 + Zmax/20, scale = Zmax/15).pdf(Zc) ) / 5
    # Normalise, using midpoint method since we have concentration at cell centers
    C0 = C0 / np.sum(dz*C0)

    # Run calculation
    C, evap, biod = diffusion_solver(Zc, C0, K, Tmax, dt, kw = kw, Q = Q)

    # Calculate remaining mass for all timesteps by integrating over z
    mass = np.sum(dz*C[:,:], axis = 1)

    # Assert that the remainining, evaporated and biodegraded mass
    # sums to 1 at all times.
    assert mass + evap + biod == pytest.approx(1.0, abs=1e-8)

    # One more test, with another, non-symmetric diffusivity
    K0 = 1e-3
    K1 = 1e-2
    K = K0 + K1 * np.sin(0.5*np.pi*Zf/Zmax)

    # Run calculation
    C, evap, biod = diffusion_solver(Zc, C0, K, Tmax, dt, kw = kw, Q = Q)

    # Calculate remaining mass for all timesteps by integrating over z
    mass = np.sum(dz*C[:,:], axis = 1)

    # Assert that the remainining, evaporated and biodegraded mass
    # sums to 1 at all times.
    assert mass + evap + biod == pytest.approx(1.0, abs=1e-8)
