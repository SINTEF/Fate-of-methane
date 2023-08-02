#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.sparse import diags
from numba import njit

# Based on Wikipedia article about TDMA
@njit
def thomas_solver(a, b, c, d):
    # Solves Ax = d,
    # where layout of matrix A is
    # b1 c1 ......... 0
    # a2 b2 c2 ........
    # .. a3 b3 c3 .....
    # .................
    # .............. cN-1
    # 0 ..........aN bN
    # Note index offset of a
    N = len(d)
    c_ = np.zeros(N-1)
    d_ = np.zeros(N)
    x  = np.zeros(N)
    c_[0] = c[0]/b[0]
    d_[0] = d[0]/b[0]
    for i in range(1, N-1):
        q = (b[i] - a[i-1]*c_[i-1])
        c_[i] = c[i]/q
        d_[i] = (d[i] - a[i-1]*d_[i-1])/q
    d_[N-1] = (d[N-1] - a[N-2]*d_[N-2])/(b[N-1] - a[N-2]*c_[N-2])
    x[-1] = d_[-1]
    for i in range(N-2, -1, -1):
        x[i] = d_[i] - c_[i]*x[i+1]
    return x


def thomas(A, b):
    # Solves Ax = b to find x
    # This is a wrapper function, which unpacks
    # A from a sparse array structure into separate diagonals,
    # and passes them to the numba-compiled solver defined above.
    # Note, this method needs A to be diagonally dominant.
    x = thomas_solver(A.diagonal(-1), A.diagonal(0), A.diagonal(1), b)
    return x


def get_system_matrices(Nx, dx, dt, K, Q, kw):
    # Prefactors
    a  = dt/(2*dx**2)
    b  = dt/2
    # Check that K is given on cell faces
    assert len(K) == Nx + 1

    # Create system matrices
    # Equation is L C^{i+1} = R C^i + BC
    # Matrices are tridiagonal, and additionally we have L = -R
    # (except for the 1 on the main diagonal, see Appendix A in the report),
    # hence we only need to construct the diagonals of L.

    # We assume that K = [K(z_{-1/2}), K(z_{1/2}), K(z_{3/2}), ... , K(z_{N-1/2}],
    # where z_{-1/2} is the sea surface and z_{N-1/2} is the sea floor

    # Upper diagonal
    upper = -a*K[1:-1]
    # Main diagonal
    main = a*K[1:] + a*K[:-1] + b*Q
    # Lower diagonal
    lower = -a*K[1:-1]

    # Overwrite boundary points to handle boundary conditions
    # Prescribed-flux BC at surface
    main[0]  =  a*K[1] + b*kw/dx + b*Q
    # No-diffusive-flux BC at sea floor
    main[-1] =  a*K[-2] + b*Q
    # Create sparse matrix
    L  = diags(diagonals = [ upper, 1 + main,  lower], offsets = [1, 0, -1])
    R  = diags(diagonals = [-upper, 1 - main, -lower], offsets = [1, 0, -1])

    #print('FVM, L: ', L.todense()[:3,:3])
    #print('FVM, R: ', R.todense()[:3,:3])
    return L, R

def diffusion_solver(X, C0, K, Tmax, dt, Q = 0, kw = 0, K_times = None):
    '''
    This function solves the diffusion-reaction equation, using
    2nd-order central space discretisation and the Crank-Nicolson
    method, which approximates the time-derivative in the PDE as

    (C(t + dt) - C(t) ) / dt = ( F(C(t+dt)) + F(C(t)) ) / 2

    where F(C(t)) is the right-hand side of the PDE. The method is
    implicit, since the equation to find C(t + dt) is given in terms
    of C(t + dt) itself. This means that at every timestep, a linear
    system of equations is solved. The method consists of creating two
    matrices, L and R, and solving the system

    L*C(t + dt) = R*C(t).

    The system matrices are tri-diagonal, and we have chosen to use
    scipy.sparse.dia_matrix to store them. The matrices are constructed
    with scipy.sparse.diags.

    Parameters
    ----------
    X : ndarray
        Vector of cell center positions (constant spacing assumed).
    C0 : ndarray
        Vector of initial concentrations, at positions given by `X`.
    K : ndarray
        Vector of diffusivity values, at positions given by `X`.
        or
        2D array of diffusivity values, at positions given by `X` and times given by K_times.
    Tmax : float
        Time to integrate to.
    dt : float
        Integration timestep.
    Q : float
        Biodegradation rate.
    kw : float
        Mass transfer coefficient at surface.
    K_times : ndarray
        1D array of times for when K values are valid, given in seconds since simulation start.

    Returns
    -------
    C : ndarray
        2D array with concentrations as a function of position and time.
    evap : ndarray
        1D array with total evaporated amount as a function of time.
    biod : ndarray
        1D array with total biodegraded amount as a function of time.
    '''

    # Check for consistency of input
    if K_times is not None:
        assert len(K.shape) == 2
        assert K.shape[0] == len(X) + 1
        assert K.shape[1] == len(K_times)
    else:
        assert len(K.shape) == 1
        assert K.shape[0] == len(X) + 1

    # Numerical parameters
    dx = X[1] - X[0]

    Nx = X.size
    Nt = int(Tmax / dt)

    # 2D array to hold concetration results
    C      = np.zeros((Nt+1, Nx))
    # 1D arrays to track amounts evaporated and biodegraded
    evap   = np.zeros(Nt+1)
    biod   = np.zeros(Nt+1)

    # System matrices for Crank-Nicolson solver
    if K_times is None:
        # In this case, K is a 1D array
        L, R = get_system_matrices(Nx, dx, dt, K, Q, kw)
    else:
        # In this case, K is a 2D array
        L, R = get_system_matrices(Nx, dx, dt, K[:,0], Q, kw)


    #############################
    ####    Loop over time   ####
    #############################

    # Initialise concentration
    C[0,:] = C0
    for n in range(Nt):

        # Solve matrix system, using iterative methods
        # for sparse systems: bicgstab is faster, but does not
        # always converge, therefore using gmres as fallback.
        rhs = R.dot(C[n,:])
        C[n+1,:] = thomas(L, rhs)

        # Here, keep track of the disappeared mass.
        # Calculate outwards flux at time t
        flux1 = kw*C[n,0]
        #print('flux1 FVM: ', flux1, kw, C[n,0], dx)
        # Calculated biodegraded rate at time t
        biod1 = dx*Q*np.sum(C[n,:])
        # Calculate outwards flux at time t + dt
        flux2 = kw*C[n+1,0]
        # Calculated biodegraded rate at time t + dt
        biod2 = dx*Q*np.sum(C[n+1,:])

       # print(n, flux1, flux2, C[n,0], C[n+1,0])

        # Store time-averaged values over two timesteps
        # (cumulative amounts)
        evap[n+1] = evap[n] + dt * (flux1 + flux2) / 2
        biod[n+1] = biod[n] + dt * (biod1 + biod2) / 2

        # If K is variable, recalculate system matrices
        if K_times is not None:
            t = n*dt
            i = np.argmin(np.abs(K_times - t))
            L, R = get_system_matrices(Nx, dx, dt, K[:,i], Q, kw)

    return C, evap, biod
