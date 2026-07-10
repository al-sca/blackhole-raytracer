"""
Black Hole Trajectory Analysis

This script analyzes the quality of different integration methods for photon trajectories
around a Schwarzschild black hole. It tracks physical quantities like angular momentum 
and energy conservation, analyzes periastron accuracy, and compares methods for different
impact parameters.
"""
import os
import time
import math
import warnings
import traceback
from matplotlib import patches
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import ticker
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.colors import LinearSegmentedColormap
# import seaborn as sns
import pandas as pd
from numba import njit
import concurrent.futures
from functools import partial
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import ellipk, ellipe
from scipy.optimize import minimize_scalar


# Constants
G = 1.0  # Gravitational constant (in geometric units)
M = 0.5  # Black hole mass (corresponds to r_s = 1)
C = 1.0  # Speed of light
R_S = 2 * G * M / (C * C)  # Schwarzschild radius
B_CRIT = 3 * np.sqrt(3) * R_S / 2

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 10,
    'figure.figsize': (8, 6),
    'figure.dpi': 300
})

@njit
def pvcalc_sa(f, y, h2):
    """
    Calculate derivatives for geodesic equations in Schwarzschild spacetime.
    
    Args:
        f: Output array for derivatives
        y: Input state vector [x, y, z, vx, vy, vz]
        h2: Squared angular momentum
        
    Returns:
        Updated f array with derivatives
    """
    f[0] = y[3]
    f[1] = y[4]
    f[2] = y[5]
    q = 1 / math.pow(y[0]*y[0] + y[1]*y[1] + y[2]*y[2], 2.5)
    f[3] = -1.5 * h2 * y[0] * q
    f[4] = -1.5 * h2 * y[1] * q
    f[5] = -1.5 * h2 * y[2] * q
    return f

@njit
def rk4_step_sa(y, f, y_temp, point, velocity, k1, k2, k3, k4, h2, h):
    """
    Perform a single step of 4th-order Runge-Kutta integration.
    
    Args:
        y: State vector
        f: Derivatives
        y_temp: Temporary storage
        point: Position vector (x, y, z)
        velocity: Velocity vector (vx, vy, vz)
        k1, k2, k3, k4: RK4 stage vectors
        h2: Squared angular momentum
        h: Step size
    """
    oldx = point[0]
    oldy = point[1]
    oldz = point[2]

    y[0] = point[0]
    y[1] = point[1]
    y[2] = point[2]
    y[3] = velocity[0]
    y[4] = velocity[1]
    y[5] = velocity[2]
    
    pvcalc_sa(f, y, h2)
    for i in range(6):
        k1[i] = f[i]
        y_temp[i] = y[i] + 0.5 * h * k1[i]
    
    pvcalc_sa(f, y_temp, h2)
    for i in range(6):
        k2[i] = f[i]
        y_temp[i] = y[i] + 0.5 * h * k2[i]
    
    pvcalc_sa(f, y_temp, h2)
    for i in range(6):
        k3[i] = f[i]
        y_temp[i] = y[i] + h * k3[i]
    
    pvcalc_sa(f, y_temp, h2)
    for i in range(6):
        k4[i] = f[i]

    q = 1./6. * h
    velocity[0] += q * (k1[3] + 2*k2[3] + 2*k3[3] + k4[3])
    velocity[1] += q * (k1[4] + 2*k2[4] + 2*k3[4] + k4[4])
    velocity[2] += q * (k1[5] + 2*k2[5] + 2*k3[5] + k4[5])

    point[0] += q * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
    point[1] += q * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
    point[2] += q * (k1[2] + 2*k2[2] + 2*k3[2] + k4[2])

    y_temp[0] = oldx
    y_temp[1] = oldy
    y_temp[2] = oldz
    
    return

@njit
def euler_step_sa(y, f, y_temp, point, velocity, k1, k2, k3, k4, h2, h):
    """
    Perform a single step of Euler integration.
    
    Args:
        y: State vector
        f: Derivatives
        y_temp: Temporary storage
        point: Position vector (x, y, z)
        velocity: Velocity vector (vx, vy, vz)
        k1, k2, k3, k4: Storage vectors (k1 used)
        h2: Squared angular momentum
        h: Step size
    """
    oldx = point[0]
    oldy = point[1]
    oldz = point[2]

    y[0] = point[0]
    y[1] = point[1]
    y[2] = point[2]
    y[3] = velocity[0]
    y[4] = velocity[1]
    y[5] = velocity[2]
    
    pvcalc_sa(f, y, h2)
    for i in range(6):
        k1[i] = f[i]
    
    velocity[0] += h * k1[3]
    velocity[1] += h * k1[4]
    velocity[2] += h * k1[5]

    point[0] += h * k1[0]
    point[1] += h * k1[1]
    point[2] += h * k1[2]

    y_temp[0] = oldx
    y_temp[1] = oldy
    y_temp[2] = oldz
    
    return

@njit
def adams_bashforth_step_sa(y, f, y_temp, point, velocity, k1, k2, k3, k4, h2, h):
    """
    Perform a single step of 2nd-order Adams-Bashforth integration.
    
    Args:
        y: State vector
        f: Derivatives
        y_temp: Temporary storage
        point: Position vector (x, y, z)
        velocity: Velocity vector (vx, vy, vz)
        k1, k2, k3, k4: Storage vectors (k1 used)
        h2: Squared angular momentum
        h: Step size
    """
    oldx = point[0]
    oldy = point[1]
    oldz = point[2]

    y[0] = point[0]
    y[1] = point[1]
    y[2] = point[2]
    y[3] = velocity[0]
    y[4] = velocity[1]
    y[5] = velocity[2]
    
    pvcalc_sa(f, y, h2)
    current_f = f.copy()
    
    # use euler first
    if k1[0] == 0 and k1[1] == 0 and k1[2] == 0:
        velocity[0] += h * current_f[3]
        velocity[1] += h * current_f[4]
        velocity[2] += h * current_f[5]
        
        point[0] += h * current_f[0]
        point[1] += h * current_f[1]
        point[2] += h * current_f[2]
    else:
        # 2nd Adams-Bashforth
        velocity[0] += h * (1.5 * current_f[3] - 0.5 * k1[3])
        velocity[1] += h * (1.5 * current_f[4] - 0.5 * k1[4])
        velocity[2] += h * (1.5 * current_f[5] - 0.5 * k1[5])
        
        point[0] += h * (1.5 * current_f[0] - 0.5 * k1[0])
        point[1] += h * (1.5 * current_f[1] - 0.5 * k1[1])
        point[2] += h * (1.5 * current_f[2] - 0.5 * k1[2])
    
    for i in range(6):
        k1[i] = current_f[i]
        
    y_temp[0] = oldx
    y_temp[1] = oldy
    y_temp[2] = oldz
    
    return

@njit
def adams_bashforth4_step_sa(y, f, y_temp, point, velocity, k1, k2, k3, k4, h2, h):
    """
    Perform a single step of 4th-order Adams-Bashforth integration.
    
    Args:
        y: State vector
        f: Derivatives
        y_temp: Temporary storage
        point: Position vector (x, y, z)
        velocity: Velocity vector (vx, vy, vz)
        k1, k2, k3, k4: Storage vectors for previous steps
        h2: Squared angular momentum
        h: Step size
    """
    oldx = point[0]
    oldy = point[1]
    oldz = point[2]

    y[0] = point[0]
    y[1] = point[1]
    y[2] = point[2]
    y[3] = velocity[0]
    y[4] = velocity[1]
    y[5] = velocity[2]
    
    pvcalc_sa(f, y, h2)
    current_f = f.copy()
    
    # use lower-order methods
    if k3[0] == 0 and k3[1] == 0 and k3[2] == 0:
        if k2[0] == 0 and k2[1] == 0 and k2[2] == 0:
            if k1[0] == 0 and k1[1] == 0 and k1[2] == 0:
                # Use Euler for first step
                velocity[0] += h * current_f[3]
                velocity[1] += h * current_f[4]
                velocity[2] += h * current_f[5]
                
                point[0] += h * current_f[0]
                point[1] += h * current_f[1]
                point[2] += h * current_f[2]
            else:
                # Use 2nd-order Adams-Bashforth for second step
                velocity[0] += h * (1.5 * current_f[3] - 0.5 * k1[3])
                velocity[1] += h * (1.5 * current_f[4] - 0.5 * k1[4])
                velocity[2] += h * (1.5 * current_f[5] - 0.5 * k1[5])
                
                point[0] += h * (1.5 * current_f[0] - 0.5 * k1[0])
                point[1] += h * (1.5 * current_f[1] - 0.5 * k1[1])
                point[2] += h * (1.5 * current_f[2] - 0.5 * k1[2])
        else:
            # Use 3rd-order Adams-Bashforth for third step
            velocity[0] += h * ((23./12.) * current_f[3] - (16./12.) * k1[3] + (5./12.) * k2[3])
            velocity[1] += h * ((23./12.) * current_f[4] - (16./12.) * k1[4] + (5./12.) * k2[4])
            velocity[2] += h * ((23./12.) * current_f[5] - (16./12.) * k1[5] + (5./12.) * k2[5])
            
            point[0] += h * ((23./12.) * current_f[0] - (16./12.) * k1[0] + (5./12.) * k2[0])
            point[1] += h * ((23./12.) * current_f[1] - (16./12.) * k1[1] + (5./12.) * k2[1])
            point[2] += h * ((23./12.) * current_f[2] - (16./12.) * k1[2] + (5./12.) * k2[2])
    else:
        # Use 4th-order Adams-Bashforth
        velocity[0] += h * ((55./24.) * current_f[3] - (59./24.) * k1[3] + (37./24.) * k2[3] - (9./24.) * k3[3])
        velocity[1] += h * ((55./24.) * current_f[4] - (59./24.) * k1[4] + (37./24.) * k2[4] - (9./24.) * k3[4])
        velocity[2] += h * ((55./24.) * current_f[5] - (59./24.) * k1[5] + (37./24.) * k2[5] - (9./24.) * k3[5])
        
        point[0] += h * ((55./24.) * current_f[0] - (59./24.) * k1[0] + (37./24.) * k2[0] - (9./24.) * k3[0])
        point[1] += h * ((55./24.) * current_f[1] - (59./24.) * k1[1] + (37./24.) * k2[1] - (9./24.) * k3[1])
        point[2] += h * ((55./24.) * current_f[2] - (59./24.) * k1[2] + (37./24.) * k2[2] - (9./24.) * k3[2])
    
    # Shift history for next step
    for i in range(6):
        k3[i] = k2[i]
        k2[i] = k1[i]
        k1[i] = current_f[i]
        
    y_temp[0] = oldx
    y_temp[1] = oldy
    y_temp[2] = oldz
    
    return

@njit
def adams_moulton4_step_sa(y, f, y_temp, point, velocity, k1, k2, k3, k4, h2, h):
    """
    Perform a single step of 4th-order Adams-Moulton integration.
    
    Args:
        y: State vector
        f: Derivatives
        y_temp: Temporary storage
        point: Position vector (x, y, z)
        velocity: Velocity vector (vx, vy, vz)
        k1, k2, k3, k4: Storage vectors for previous steps
        h2: Squared angular momentum
        h: Step size
    """
    max_iter = 15
    tol = 1e-8

    oldx = point[0]
    oldy = point[1]
    oldz = point[2]
    oldvx = velocity[0]
    oldvy = velocity[1]
    oldvz = velocity[2]

    y[0] = point[0]
    y[1] = point[1]
    y[2] = point[2]
    y[3] = velocity[0]
    y[4] = velocity[1]
    y[5] = velocity[2]
    
    pvcalc_sa(f, y, h2)
    current_f = f.copy()
    
    # use RK4 first
    if k2[0] == 0 and k2[1] == 0 and k2[2] == 0:
        for i in range(6):
            k4[i] = current_f[i]
            y_temp[i] = y[i] + 0.5 * h * k4[i]
        
        pvcalc_sa(f, y_temp, h2)
        for i in range(6):
            k3[i] = f[i]
            y_temp[i] = y[i] + 0.5 * h * k3[i]
        
        pvcalc_sa(f, y_temp, h2)
        for i in range(6):
            k2[i] = f[i]
            y_temp[i] = y[i] + h * k2[i]
        
        pvcalc_sa(f, y_temp, h2)
        for i in range(6):
            k1[i] = f[i]

        q = h / 6.0
        velocity[0] += q * (k4[3] + 2*k3[3] + 2*k2[3] + k1[3])
        velocity[1] += q * (k4[4] + 2*k3[4] + 2*k2[4] + k1[4])
        velocity[2] += q * (k4[5] + 2*k3[5] + 2*k2[5] + k1[5])
        
        point[0] += q * (k4[0] + 2*k3[0] + 2*k2[0] + k1[0])
        point[1] += q * (k4[1] + 2*k3[1] + 2*k2[1] + k1[1])
        point[2] += q * (k4[2] + 2*k3[2] + 2*k2[2] + k1[2])
        
        y[0] = point[0]
        y[1] = point[1]
        y[2] = point[2]
        y[3] = velocity[0]
        y[4] = velocity[1]
        y[5] = velocity[2]
        
        pvcalc_sa(f, y, h2)
        current_f = f.copy()
    else:
        # Initial guess using Explicit Euler
        new_point_x = point[0] + h * current_f[0]
        new_point_y = point[1] + h * current_f[1]
        new_point_z = point[2] + h * current_f[2]
        new_vel_x = velocity[0] + h * current_f[3]
        new_vel_y = velocity[1] + h * current_f[4]
        new_vel_z = velocity[2] + h * current_f[5]
        
        converged = False
        for iter in range(max_iter):
            y[0] = new_point_x
            y[1] = new_point_y
            y[2] = new_point_z
            y[3] = new_vel_x
            y[4] = new_vel_y
            y[5] = new_vel_z
            
            pvcalc_sa(f, y, h2)
            f_np1 = f.copy()
            
            # AM4: y_{n+1} = y_n + (h/24)*(9*f_{n+1} + 19*f_n - 5*f_{n-1} + f_{n-2})
            h24 = h/24.0
            new_point_x_updated = oldx + h24 * (9.0*f_np1[0] + 19.0*current_f[0] - 5.0*k1[0] + k2[0])
            new_point_y_updated = oldy + h24 * (9.0*f_np1[1] + 19.0*current_f[1] - 5.0*k1[1] + k2[1])
            new_point_z_updated = oldz + h24 * (9.0*f_np1[2] + 19.0*current_f[2] - 5.0*k1[2] + k2[2])
            new_vel_x_updated = oldvx + h24 * (9.0*f_np1[3] + 19.0*current_f[3] - 5.0*k1[3] + k2[3])
            new_vel_y_updated = oldvy + h24 * (9.0*f_np1[4] + 19.0*current_f[4] - 5.0*k1[4] + k2[4])
            new_vel_z_updated = oldvz + h24 * (9.0*f_np1[5] + 19.0*current_f[5] - 5.0*k1[5] + k2[5])
            
            dx = abs(new_point_x_updated - new_point_x)
            dy = abs(new_point_y_updated - new_point_y)
            dz = abs(new_point_z_updated - new_point_z)
            dvx = abs(new_vel_x_updated - new_vel_x)
            dvy = abs(new_vel_y_updated - new_vel_y)
            dvz = abs(new_vel_z_updated - new_vel_z)
            
            if max(dx, dy, dz, dvx, dvy, dvz) < tol:
                converged = True
                break
                
            new_point_x = new_point_x_updated
            new_point_y = new_point_y_updated
            new_point_z = new_point_z_updated
            new_vel_x = new_vel_x_updated
            new_vel_y = new_vel_y_updated
            new_vel_z = new_vel_z_updated
        
        point[0] = new_point_x
        point[1] = new_point_y
        point[2] = new_point_z
        velocity[0] = new_vel_x
        velocity[1] = new_vel_y
        velocity[2] = new_vel_z

    for i in range(6):
        k3[i] = k2[i]
        k2[i] = k1[i]
        k1[i] = current_f[i]
    
    y_temp[0] = oldx
    y_temp[1] = oldy
    y_temp[2] = oldz
    
    return

@njit
def bowie_single_step_sa(u, u_prime, h, k):
    """
    Perform a single step of Bowie integration method.
    
    Args:
        u: Current position (u = 1/r)
        u_prime: Current derivative (du/dphi)
        h: Step size (delta phi)
        k: Constant (GM/c^2)
        
    Returns:
        tuple: (u_next, u_prime_next) - Updated position and derivative
    """
    f_y = 3.0 * k * u * u - u
    df_dy = 6.0 * k * u - 1.0
    d2f_dy2 = 6.0 * k

    u_prime_squared = u_prime * u_prime
    
    u_next = u + h * u_prime + \
                (h * h / 2.0) * f_y + \
                (h * h * h / 6.0) * df_dy * u_prime + \
                (h * h * h * h / 24.0) * (d2f_dy2 * u_prime_squared + df_dy * f_y)
    
    u_prime_next = u_prime + h * f_y + \
                    (h * h / 2.0) * df_dy * u_prime + \
                    (h * h * h / 6.0) * (d2f_dy2 * u_prime_squared + df_dy * f_y)
    
    return u_next, u_prime_next

@njit
def bowie_step_sa(y, f, y_temp, point, velocity, k1, k2, k3, k4, h2, h):
    """
    Perform a single step of Bowie integration method.
    
    Bowie's method is specialized for the Schwarzschild geodesic equations in
    Binet's form (u = 1/r).
    
    Args:
        y: State vector
        f: Derivatives
        y_temp: Temporary storage
        point: Position vector (x, y, z)
        velocity: Velocity vector (vx, vy, vz)
        k1, k2, k3, k4: Storage vectors (not used)
        h2: Squared angular momentum (not used directly)
        h: Step size
    """
    oldx = point[0]
    oldy = point[1]
    oldz = point[2]
    
    x, y, z = point
    vx, vy, vz = velocity
    
    r = math.sqrt(x*x + y*y + z*z)
    
    # L = r x v
    Lx = y*vz - z*vy
    Ly = z*vx - x*vz
    Lz = x*vy - y*vx
    
    L_squared = Lx*Lx + Ly*Ly + Lz*Lz
    L_mag = math.sqrt(max(L_squared, 1e-10))
    
    u = 1.0 / r

    # Radial unit vector
    er_x = x/r
    er_y = y/r
    er_z = z/r  
    v_r = vx * er_x + vy * er_y + vz * er_z
    
    # azimuthal velocity
    v_phi = L_mag / r
    
    # du/dphi
    epsilon = 1e-10
    if abs(v_phi) < epsilon:
        u_prime = 0.0
    else:
        u_prime = -v_r / (r * v_phi)
    
    k = 0.5  # GM/c^2, corresponds to R_S = 2*GM/c^2 = 1.0
    u_next, u_prime_next = bowie_single_step_sa(u, u_prime, h, k)
    
    # Convert back to Cartesian
    r_new = 1.0 / u_next
    v_phi_new = L_mag / r_new
    v_r_new = -u_prime_next * r_new * v_phi_new
    
    # rotate by delta_phi around angular momentum vec
    delta_phi = h
    
    # Normalize angular momentum vec
    if L_mag < 1e-10:
        L_hat_x, L_hat_y, L_hat_z = 0.0, 0.0, 1.0
    else:
        inv_L = 1.0 / L_mag
        L_hat_x = Lx * inv_L
        L_hat_y = Ly * inv_L
        L_hat_z = Lz * inv_L
    
    # Rodrigues' rotation formula to rotate er by delta_phi around L_hat
    cos_phi = math.cos(delta_phi)
    sin_phi = math.sin(delta_phi)
    one_minus_cos = 1.0 - cos_phi
    
    # er x L_hat
    cross_x = er_y * L_hat_z - er_z * L_hat_y
    cross_y = er_z * L_hat_x - er_x * L_hat_z
    cross_z = er_x * L_hat_y - er_y * L_hat_x
    
    # L_hat dot er
    dot = er_x * L_hat_x + er_y * L_hat_y + er_z * L_hat_z
    
    er_new_x = er_x * cos_phi + cross_x * sin_phi + L_hat_x * dot * one_minus_cos
    er_new_y = er_y * cos_phi + cross_y * sin_phi + L_hat_y * dot * one_minus_cos
    er_new_z = er_z * cos_phi + cross_z * sin_phi + L_hat_z * dot * one_minus_cos
    
    point[0] = r_new * er_new_x
    point[1] = r_new * er_new_y
    point[2] = r_new * er_new_z
    
    # Also need to rotate the tangential component
    et_x = L_hat_y * er_z - L_hat_z * er_y
    et_y = L_hat_z * er_x - L_hat_x * er_z
    et_z = L_hat_x * er_y - L_hat_y * er_x
    
    et_mag = math.sqrt(et_x*et_x + et_y*et_y + et_z*et_z)
    if et_mag > 1e-10:
        inv_et = 1.0 / et_mag
        et_x *= inv_et
        et_y *= inv_et
        et_z *= inv_et
        
        # Rotate tangential component
        cross_t_x = et_y * L_hat_z - et_z * L_hat_y
        cross_t_y = et_z * L_hat_x - et_x * L_hat_z
        cross_t_z = et_x * L_hat_y - et_y * L_hat_x
        
        dot_t = et_x * L_hat_x + et_y * L_hat_y + et_z * L_hat_z
        
        et_new_x = et_x * cos_phi + cross_t_x * sin_phi + L_hat_x * dot_t * one_minus_cos
        et_new_y = et_y * cos_phi + cross_t_y * sin_phi + L_hat_y * dot_t * one_minus_cos
        et_new_z = et_z * cos_phi + cross_t_z * sin_phi + L_hat_z * dot_t * one_minus_cos
        
        velocity[0] = v_r_new * er_new_x + v_phi_new * et_new_x
        velocity[1] = v_r_new * er_new_y + v_phi_new * et_new_y
        velocity[2] = v_r_new * er_new_z + v_phi_new * et_new_z
    else:
        velocity[0] = v_r_new * er_new_x
        velocity[1] = v_r_new * er_new_y
        velocity[2] = v_r_new * er_new_z
    
    y_temp[0] = oldx
    y_temp[1] = oldy
    y_temp[2] = oldz
    
    return

@njit
def obrechkoff_single_step_sa(u, u_prime, h, k):
    """
    Perform a single step of Obrechkoff integration method.
    
    Args:
        u: Current position (u = 1/r)
        u_prime: Current derivative (du/dphi)
        h: Step size (delta phi)
        k: Constant (GM/c^2)
        
    Returns:
        tuple: (u_next, u_prime_next) - Updated position and derivative
    """
    max_iter = 12
    tol = 1e-8
    
    f_y = 3.0 * k * u * u - u
    
    # Initial guess using Euler
    y_next = u + h * u_prime
    y_prime_next = u_prime + h * f_y

    h_half = h * 0.5
    h_squared_12 = h * h / 12.0
    three_k = 3.0 * k
    six_k = 6.0 * k
    
    # Newton iteration to solve the implicit system
    for _ in range(max_iter):
        f_y_next = three_k * y_next * y_next - y_next
        
        F1 = y_next - u - h_half * (u_prime + y_prime_next) - h_squared_12 * (-u + y_next + three_k * (u*u - y_next*y_next))
        F2 = y_prime_next - u_prime - h_half * (-u - y_next + three_k * (u*u + y_next*y_next)) - h_squared_12 * (y_prime_next - u_prime + six_k * (u*u_prime - y_next*y_prime_next))

        df_dy_next = six_k * y_next - 1.0
        J11 = 1.0 - h_squared_12 * (1.0 - df_dy_next)
        J12 = -h_half
        J21 = -h_half * (-1.0 + df_dy_next) - h_squared_12 * (-six_k * y_prime_next)
        J22 = 1.0 - h_squared_12 * (1.0 - six_k * y_next)

        det_J = J11 * J22 - J12 * J21
        if abs(det_J) < 1e-10:
            det_J = 1e-10 if det_J >= 0 else -1e-10

        delta_y = (-J22 * F1 + J12 * F2) / det_J
        delta_y_prime = (-J11 * F2 + J21 * F1) / det_J

        y_next += delta_y
        y_prime_next += delta_y_prime

        if abs(delta_y) < tol and abs(delta_y_prime) < tol:
            break
    
    return y_next, y_prime_next

@njit
def obrechkoff_step_sa(y, f, y_temp, point, velocity, k1, k2, k3, k4, h2, h):
    """
    Perform a single step of Obrechkoff integration method.
    
    Args:
        y: State vector
        f: Derivatives
        y_temp: Temporary storage
        point: Position vector (x, y, z)
        velocity: Velocity vector (vx, vy, vz)
        k1, k2, k3, k4: Storage vectors (not used)
        h2: Squared angular momentum (not used directly)
        h: Step size
    """
    oldx = point[0]
    oldy = point[1]
    oldz = point[2]
    
    x, y, z = point
    vx, vy, vz = velocity
    
    r_squared = x*x + y*y + z*z
    r = math.sqrt(r_squared)
    
    if r < 1e-6:
        y_temp[0] = oldx
        y_temp[1] = oldy
        y_temp[2] = oldz
        return
    
    Lx = y*vz - z*vy
    Ly = z*vx - x*vz
    Lz = x*vy - y*vx
    
    L_squared = Lx*Lx + Ly*Ly + Lz*Lz
    L_mag = math.sqrt(max(L_squared, 1e-10))
    
    inv_r = 1.0 / r
    er_x = x * inv_r
    er_y = y * inv_r
    er_z = z * inv_r
    
    v_r = vx * er_x + vy * er_y + vz * er_z
    
    u = inv_r
    v_phi = L_mag / r
    
    if abs(v_phi) < 1e-8:
        u_prime = 0.0
    else:
        u_prime = -v_r / (r * v_phi)
    
    k = 0.5
    u_next, u_prime_next = obrechkoff_single_step_sa(u, u_prime, h, k)
    
    r_new = 1.0 / u_next
    v_phi_new = L_mag / r_new
    v_r_new = -u_prime_next * r_new * v_phi_new
    
    delta_phi = h

    if L_mag < 1e-8:
        L_hat_x, L_hat_y, L_hat_z = 0.0, 0.0, 1.0
    else:
        inv_L = 1.0 / L_mag
        L_hat_x = Lx * inv_L
        L_hat_y = Ly * inv_L
        L_hat_z = Lz * inv_L
    
    cos_phi = math.cos(delta_phi)
    sin_phi = math.sin(delta_phi)
    one_minus_cos = 1.0 - cos_phi
    
    # er x L_hat
    cross_x = er_y * L_hat_z - er_z * L_hat_y
    cross_y = er_z * L_hat_x - er_x * L_hat_z
    cross_z = er_x * L_hat_y - er_y * L_hat_x
    
    # L_hat dot er
    dot = er_x * L_hat_x + er_y * L_hat_y + er_z * L_hat_z
    
    # Rodrigues' formula
    er_new_x = er_x * cos_phi + cross_x * sin_phi + L_hat_x * dot * one_minus_cos
    er_new_y = er_y * cos_phi + cross_y * sin_phi + L_hat_y * dot * one_minus_cos
    er_new_z = er_z * cos_phi + cross_z * sin_phi + L_hat_z * dot * one_minus_cos
    
    point[0] = r_new * er_new_x
    point[1] = r_new * er_new_y
    point[2] = r_new * er_new_z
    
    et_x = L_hat_y * er_z - L_hat_z * er_y
    et_y = L_hat_z * er_x - L_hat_x * er_z
    et_z = L_hat_x * er_y - L_hat_y * er_x
    
    et_sq = et_x*et_x + et_y*et_y + et_z*et_z
    if et_sq > 1e-10:
        inv_et = 1.0 / math.sqrt(et_sq)
        et_x *= inv_et
        et_y *= inv_et
        et_z *= inv_et
        
        cross_t_x = et_y * L_hat_z - et_z * L_hat_y
        cross_t_y = et_z * L_hat_x - et_x * L_hat_z
        cross_t_z = et_x * L_hat_y - et_y * L_hat_x
        
        dot_t = et_x * L_hat_x + et_y * L_hat_y + et_z * L_hat_z
        
        et_new_x = et_x * cos_phi + cross_t_x * sin_phi + L_hat_x * dot_t * one_minus_cos
        et_new_y = et_y * cos_phi + cross_t_y * sin_phi + L_hat_y * dot_t * one_minus_cos
        et_new_z = et_z * cos_phi + cross_t_z * sin_phi + L_hat_z * dot_t * one_minus_cos

        velocity[0] = v_r_new * er_new_x + v_phi_new * et_new_x
        velocity[1] = v_r_new * er_new_y + v_phi_new * et_new_y
        velocity[2] = v_r_new * er_new_z + v_phi_new * et_new_z
    else:
        velocity[0] = v_r_new * er_new_x
        velocity[1] = v_r_new * er_new_y
        velocity[2] = v_r_new * er_new_z
    
    y_temp[0] = oldx
    y_temp[1] = oldy
    y_temp[2] = oldz
    
    return

class TrajectoryAnalyzer:
    """Class to analyze and compare different integration methods for photon trajectories."""
    
    def __init__(self, output_dir="trajectory_analysis", num_threads=None):
        """Initialize the trajectory analyzer with default parameters."""
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.num_threads = num_threads if num_threads is not None else os.cpu_count()
        print(f"Using {self.num_threads} threads for parallel processing")
        
        # CPU-compatible integrators
        self.integrators = {
            "Runge-Kutta 4": self.rk4_step,
            "Euler": self.euler_step,
            "Adams-Bashforth": self.adams_bashforth_step,
            "Adams-Bashforth4": self.adams_bashforth4_step,
            "Adams-Moulton4": self.adams_moulton4_step,
            "Bowie": self.bowie_step,
            "Obrechkoff": self.obrechkoff_step,
        }
        
        # Default integrator parameters
        # self.default_params = {
        #     "Runge-Kutta 4": {"timestep": 0.08, "maxsteps": 1000},
        #     "Euler": {"timestep": 0.01, "maxsteps": 4000},
        #     "Adams-Bashforth": {"timestep": 0.05, "maxsteps": 1200},
        #     "Adams-Bashforth4": {"timestep": 0.05, "maxsteps": 1200},
        #     "Adams-Moulton4": {"timestep": 0.05, "maxsteps": 1200},
        #     "Bowie": {"timestep": 0.002, "maxsteps": 12000},
        #     "Obrechkoff": {"timestep": 0.002, "maxsteps": 12000}
        # }
        # self.default_params = {
        #     "Runge-Kutta 4": {"timestep": 0.002, "maxsteps": 12000},
        #     "Euler": {"timestep": 0.002, "maxsteps": 12000},
        #     "Adams-Bashforth": {"timestep": 0.002, "maxsteps": 12000},
        #     "Adams-Bashforth4": {"timestep": 0.002, "maxsteps": 12000},
        #     "Adams-Moulton4": {"timestep": 0.002, "maxsteps": 12000},
        #     "Bowie": {"timestep": 0.002, "maxsteps": 12000},
        #     "Obrechkoff": {"timestep": 0.002, "maxsteps": 12000}
        # }
        self.default_params = {
            "Runge-Kutta 4": {"timestep": 0.005, "maxsteps": 5000},
            "Euler": {"timestep": 0.005, "maxsteps": 5000},
            "Adams-Bashforth": {"timestep": 0.005, "maxsteps": 5000},
            "Adams-Bashforth4": {"timestep": 0.005, "maxsteps": 5000},
            "Adams-Moulton4": {"timestep": 0.005, "maxsteps": 5000},
            "Bowie": {"timestep": 0.005, "maxsteps": 5000},
            "Obrechkoff": {"timestep": 0.005, "maxsteps": 5000}
        }
        
        # Impact parameters to test
        self.impact_parameters = [2.58, 2.6, 2.62, 2.64, 2.66, 2.68, 2.7, 2.72, 2.74, 2.76, 2.78, 2.8, 
                                  3.0, 3.2, 3.4, 3.6, 3.8, 
                                  4.0, 4.2, 4.4, 4.6, 4.8, 
                                  5.0, 5.1, 5.19, 5.2, 5.3, 
                                  6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 
                                  24.0, 28.0, 30.0]
        # self.impact_parameters = [2.58, 2.6, 2.62, 2.64, 2.66, 2.68, 2.7, 2.72, 2.74, 2.76, 2.78, 2.8, 5.0, 5.1, 5.19, 5.2, 5.3, 6.0, 8.0, 10.0, 20.0, 30.0]
        
        # self.colors = plt.cm.viridis(np.linspace(0, 1, len(self.integrators)))
        self.colors = plt.cm.managua(np.linspace(0, 1, len(self.integrators)))
    
    def pvcalc(self, f, y, h2):
        """Wrapper for the numba-optimized pvcalc function."""
        return pvcalc_sa(f, y, h2)
    
    def rk4_step(self, y, f, y_temp, point, velocity, k1, k2, k3, k4, h2, h):
        """Wrapper for the numba-optimized rk4_step function."""
        return rk4_step_sa(y, f, y_temp, point, velocity, k1, k2, k3, k4, h2, h)
    
    def euler_step(self, y, f, y_temp, point, velocity, k1, k2, k3, k4, h2, h):
        """Wrapper for the numba-optimized euler_step function."""
        return euler_step_sa(y, f, y_temp, point, velocity, k1, k2, k3, k4, h2, h)
    
    def adams_bashforth_step(self, y, f, y_temp, point, velocity, k1, k2, k3, k4, h2, h):
        """Wrapper for the numba-optimized adams_bashforth_step function."""
        return adams_bashforth_step_sa(y, f, y_temp, point, velocity, k1, k2, k3, k4, h2, h)

    def adams_bashforth4_step(self, y, f, y_temp, point, velocity, k1, k2, k3, k4, h2, h):
        """Wrapper for the numba-optimized adams_bashforth4_step function."""
        return adams_bashforth4_step_sa(y, f, y_temp, point, velocity, k1, k2, k3, k4, h2, h)

    def adams_moulton4_step(self, y, f, y_temp, point, velocity, k1, k2, k3, k4, h2, h):
        """Wrapper for the numba-optimized adams_moulton4_step function."""
        return adams_moulton4_step_sa(y, f, y_temp, point, velocity, k1, k2, k3, k4, h2, h)

    def bowie_step(self, y, f, y_temp, point, velocity, k1, k2, k3, k4, h2, h):
        """Wrapper for the numba-optimized bowie_step function."""
        return bowie_step_sa(y, f, y_temp, point, velocity, k1, k2, k3, k4, h2, h)

    def obrechkoff_step(self, y, f, y_temp, point, velocity, k1, k2, k3, k4, h2, h):
        """Wrapper for the numba-optimized obrechkoff_step function."""
        return obrechkoff_step_sa(y, f, y_temp, point, velocity, k1, k2, k3, k4, h2, h)

    
    @staticmethod
    def calculate_initial_conditions(impact_parameter, distance=20.0):
        """
        Calculate initial position and velocity for a photon with a given impact parameter.
        
        Args:
            impact_parameter (float): Impact parameter of the photon
            distance (float): Initial distance from the black hole
            
        Returns:
            tuple: (position, velocity) - Initial position and velocity 3D vectors
        """

        position = np.array([-distance, impact_parameter, 0.0], dtype=np.float64)
        
        # Initial velocity points towards the origin with magnitude 1
        velocity_direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        velocity = velocity_direction / np.linalg.norm(velocity_direction)
        
        return position, velocity
    
    @staticmethod
    def calculate_angular_momentum(position, velocity):
        """
        Calculate angular momentum per unit mass for a photon.
        
        Args:
            position (ndarray): 3D position vector
            velocity (ndarray): 3D velocity vector
            
        Returns:
            ndarray: Angular momentum vector
            float: Magnitude of angular momentum
        """

        angular_momentum = np.cross(position, velocity)
        angular_momentum_mag = np.linalg.norm(angular_momentum)

        return angular_momentum, angular_momentum_mag
    
    # @staticmethod
    # def calculate_energy(position, velocity):
    #     """
    #     Calculate energy per unit mass in Schwarzschild spacetime.
        
    #     Args:
    #         position (ndarray): 3D position vector
    #         velocity (ndarray): 3D velocity vector
            
    #     Returns:
    #         float: Energy per unit mass
    #     """
    #     r = np.linalg.norm(position)
    #     v_squared = np.sum(velocity**2)
    #     # For photon E^2 = V^2 (1 - r_s/r)
    #     # this is a proxy for energy conservation
    #     schwarzschild_factor = 1.0 - R_S / r if r >= R_S else 0.0
    #     energy = v_squared * schwarzschild_factor
        
    #     return energy

    @staticmethod
    def calculate_energy(position, velocity):
        """Calculate conserved energy proxy E^2 = (1-R_s/r)v^2 for photons."""
        r_sq = position[0]**2 + position[1]**2 + position[2]**2
        if r_sq < 1e-12: 
            return 0.0
        r = math.sqrt(r_sq)
        if r <= R_S: 
            return 0.0 
        # v_sq = velocity[0]**2 + velocity[1]**2 + velocity[2]**2
        # schwarzschild_factor = 1.0 - R_S / r
        # v_sq = velocity[0]**2 + velocity[1]**2 + velocity[2]
        v_dot_r = position[0]*velocity[0] + position[1]*velocity[1] + position[2]*velocity[2]
        dr_dlam_sq = (v_dot_r * v_dot_r) / r_sq if r_sq > 1e-12 else 0.0

        Lx = position[1]*velocity[2] - position[2]*velocity[1]
        Ly = position[2]*velocity[0] - position[0]*velocity[2]
        Lz = position[0]*velocity[1] - position[1]*velocity[0]
        L_sq = Lx*Lx + Ly*Ly + Lz*Lz

        if r <= R_S: 
            return 0.0
        term2 = (1.0 - R_S / r) * L_sq / r_sq
        conserved_quantity_sq = dr_dlam_sq + term2

        return conserved_quantity_sq 
    
    @staticmethod
    def calculate_carter_constant(position, velocity, angular_momentum):
        """
        Calculate Carter constant (Schwarzschild version).
        
        In Schwarzschild, this is not distinct from angular momentum,
        but we include it for completeness and future extension to Kerr.
        
        Args:
            position (ndarray): 3D position vector
            velocity (ndarray): 3D velocity vector
            angular_momentum (ndarray): Angular momentum vector
            
        Returns:
            float: Carter constant-like quantity
        """
        # In Schwarzschild, Carter constant simplifies to L^2
        return np.sum(angular_momentum**2)
    
    def calculate_theoretical_deflection(self, impact_parameter):
        """
        Calculate theoretical deflection angle using elliptic integrals.
        
        Args:
            impact_parameter (float): Impact parameter
            
        Returns:
            float: Deflection angle in degrees
        """
        
        # Find turning point (periastron)
        def effective_potential(r, b):
            return (1.0 - R_S / r) * (b**2 / r**2)
        
        result = minimize_scalar(lambda r: -effective_potential(r, impact_parameter), 
                                bounds=(R_S + 0.01, 20), method='bounded')
        r_min = result.x
        
        def integrand(r, b):
            if r < R_S + 1e-10:
                return 0.0
                
            u = 1.0 / r
            term = (1.0 / b**2) - u**2 * (1.0 - R_S * u)
            
            if term < 0:
                return 0.0
                
            return 1.0 / np.sqrt(term)
        
        # compute the deflection angle with finite bounds
        try:
            upper_bound = 1000.0 * r_min
            
            result, _ = quad(integrand, r_min, upper_bound, 
                                    args=(impact_parameter),
                                    limit=100, epsabs=1e-6, epsrel=1e-6)
            
            deflection_angle = 2 * result - np.pi
            deflection_angle_deg = deflection_angle * 180 / np.pi
            
            if abs(impact_parameter - B_CRIT) < 0.1:
                return min(deflection_angle_deg, 720)  # Max 2 full orbits
                
            return deflection_angle_deg
        except Exception as e:
            print(f"Error calculating theoretical deflection for b={impact_parameter}: {e}")
            return np.nan

    def calculate_theoretical_shapiro_delay(self, impact_parameter, d1, d2):
        """Calculates Shapiro delay using numerical integration with better error handling."""

        b = impact_parameter
        if b < 1e-9: 
            return 0.0

        periastron = self.calculate_analytical_periastron(b)
        if np.isnan(periastron) or periastron <= R_S:
            return np.nan 

        # Integrands
        def integrand_rel(r, b):
            # term1_sq = (1.0 - rs_param / r)**2
            # term2 = 1.0 - (b**2 / r**2) * (1.0 - rs_param / r)
            term1 = (1.0 - R_S / r)
            term2 = (b**2 / r**2) * (1 - R_S / r)
            if term1 <= 0 or term2 <= 0: 
                return 0.0 
            denom = term1*math.sqrt(1 - term2)
            
            # denom = (1.0 - rs_param / r) * math.sqrt(term2)
            if denom <= 1e-12: 
                return 0.0

            return 1.0 / denom

        def integrand_flat(r, b):
            term_sq = r**2 - b**2
            if term_sq <= 0: 
                return 0.0
            denom = math.sqrt(term_sq)
            if denom <= 1e-12: 
                return 0.0
            return r / denom

        quad_limit = 200  # 100
        quad_eps = 1e-6   # 1e-8
        
        # Numerical issues at the turning point
        periastron_offset = periastron + 1e-6
        
        try:              
            # Periastron to d1
            t_rel1, err_rel1 = quad(integrand_rel, periastron_offset, d1, 
                                    args=(b,), limit=quad_limit, 
                                    epsabs=quad_eps, epsrel=quad_eps)
            
            t_flat1, err_flat1 = quad(integrand_flat, periastron_offset, d1, 
                                    args=(b,), limit=quad_limit, 
                                    epsabs=quad_eps, epsrel=quad_eps)

            # Periastron to d2
            t_rel2, err_rel2 = quad(integrand_rel, periastron_offset, d2, 
                                    args=(b,), limit=quad_limit, 
                                    epsabs=quad_eps, epsrel=quad_eps)
            
            t_flat2, err_flat2 = quad(integrand_flat, periastron_offset, d2, 
                                    args=(b,), limit=quad_limit, 
                                    epsabs=quad_eps, epsrel=quad_eps)

            t_rel_total = t_rel1 + t_rel2
            t_flat_total = t_flat1 + t_flat2

            max_rel_err = max(
                err_rel1 / abs(t_rel1) if abs(t_rel1)>1e-12 else 0,
                err_rel2 / abs(t_rel2) if abs(t_rel2)>1e-12 else 0,
                err_flat1 / abs(t_flat1) if abs(t_flat1)>1e-12 else 0,
                err_flat2 / abs(t_flat2) if abs(t_flat2)>1e-12 else 0
            )
            
            if max_rel_err > 1e-4:  # 1e-5
                print(f"Large relative integration error ({max_rel_err:.2e}) in Shapiro delay for b={b:.4f}")

        except Exception as e:
            print(f"Integration error for Shapiro delay, b={b:.4f}: {e}", exc_info=False)
            return np.nan

        delay = t_rel_total - t_flat_total
        if delay < -1e-6:
            print(f"Calculated Shapiro delay is negative ({delay:.2e}) for b={b:.4f}. May indicate integration issues near periastron.")
        return max(0, delay)

    def calculate_theoretical_shapiro_delay_approx(self, impact_parameter, d1,d2):
        """
            Approximation, which is not valid close to the critical impact parameter
        """
        if impact_parameter > 10 * R_S:
            return 2 * M * np.log(4 * d1 * d2 / (impact_parameter**2))
        k = 2*math.sqrt(R_S*impact_parameter) / (impact_parameter + R_S)
        delay = 2*M * (ellipk(k) - ellipe(k))

        return delay
    
    def calculate_winding_angle(self, positions):
        """
        Calculate the total winding angle of a trajectory around the black hole.
        """
        # Angles in the x-y plane
        phi_values = np.arctan2(positions[:, 1], positions[:, 0])
        phi_unwrapped = np.unwrap(phi_values)
        total_angle = np.abs(phi_unwrapped[-1] - phi_unwrapped[0]) * 180 / np.pi
        
        return total_angle
    
    def calculate_analytical_periastron(self, impact_parameter):
        """
        Calculates the analytical periastron distance using the effective potential.
        Finds the largest real root > R_S of the cubic r^3 - b^2*r + b^2*R_S = 0.
        """
        b = impact_parameter
        if b < 0: 
            b = abs(b)

        # Radial infall
        if b < 1e-9:
            return R_S

        # If b < B_CRIT, the photon is captured. The "periastron" in the sense
        # of closest approach before capture isn't well-defined by this cubic's
        # roots outside R_S. Define periastron as R_S for capture.
        if b < B_CRIT:
            return R_S
        elif abs(b - B_CRIT) < 1e-9:
             # Circular orbit at photon sphere
             return 1.5 * R_S

        # --- r^3 + 0*r^2 - b^2*r + b^2*R_S = 0 ---
        coeffs = [1, 0, -b**2, b**2 * R_S]
        try:
            roots = np.roots(coeffs)

            # Filter for real roots greater than R_S + epsilon
            real_roots_gt_rs = [r.real for r in roots if abs(r.imag) < 1e-10 and r.real > R_S + 1e-10]

            if not real_roots_gt_rs:
                 real_roots_near_rs = [r.real for r in roots if abs(r.imag) < 1e-10 and abs(r.real - R_S) < 1e-9]
                 if real_roots_near_rs:
                     print(f"Warning: Analytical periastron root very close to R_S for b={b}. Returning R_S.")
                     return R_S
                 else:
                     print(f"Warning: No real root > R_S found for analytical periastron, b={b}. Roots: {roots}. Returning NaN.")
                     return np.nan

            periastron = max(real_roots_gt_rs)
            return periastron

        except Exception as e:
            print(f"Error solving cubic for analytical periastron, b={b}: {e}")
            traceback.print_exc()
            return np.nan
        
    # def calculate_analytical_periastron(self, impact_parameter):
    #     """
    #     Calculates the periastron (closest approach distance) for a photon
    #     orbiting a Schwarzschild black hole using robust root-finding method.

    #     Args:
    #         impact_parameter (float): The impact parameter of the photon.
    #         M (float): Mass of the black hole.

    #     Returns:
    #         float: The periastron distance.
    #     """
    #     R_S = 2 * M

    #     def periastron_equation(r):
    #         """
    #         The equation whose largest real root is the periastron.
    #         """
    #         return 1 - (impact_parameter**2 / r**2) * (1 - R_S / r)

    #     # b = 0 (straight into the black hole)
    #     if impact_parameter == 0:
    #         return R_S

    #     # Use brentq to find root.  Periastron must be greater than R_S.
    #     try:
    #         # brentq to find the largest root, which is the periastron. Start search at r_s
    #         # and put an upper bound, which is related to b, where the periastron is found.
    #         periastron = brentq(periastron_equation, R_S, impact_parameter * 5 + R_S)

    #     except ValueError:
    #         # very large b (weak field): periastron approx impact_parameter
    #         periastron = impact_parameter
            
    #     return periastron
    
    def analyze_shapiro_accuracy(self, all_results):
        """
        Analyze the accuracy of Shapiro delay calculations for all methods.
        """
        impact_parameters = sorted(all_results.keys())
        methods = set()
        for results in all_results.values():
            methods.update(results.keys())
        methods = sorted(methods)

        plt.figure(figsize=(12, 8))

        # Percentage error and absolute error
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        for i, method in enumerate(methods):
            error_percentages = []
            error_absolute = []
            valid_impact_parameters = []

            for b in impact_parameters:
                if method in all_results[b]:
                    result = all_results[b][method]
                    if "shapiro_delays" in result:
                        final_delay = result["shapiro_delays"][-1]

                        d1 = np.linalg.norm(result["positions"][0])
                        d2 = np.linalg.norm(result["positions"][-1])

                        theoretical = self.calculate_theoretical_shapiro_delay(b, d1, d2)

                        if np.isnan(theoretical) or np.isnan(final_delay):
                            error_percent = np.nan
                            error_abs = np.nan
                        else:
                            error_percent = abs((final_delay - theoretical) / theoretical) * 100
                            error_abs = abs(final_delay - theoretical)


                        error_percentages.append(error_percent)
                        error_absolute.append(error_abs)
                        valid_impact_parameters.append(b)

            if valid_impact_parameters:
                ax1.plot(valid_impact_parameters, error_percentages, 'o-',
                        lw=1.0, markersize=6,
                        color=self.colors[i % len(self.colors)],
                        label=method)

                ax2.plot(valid_impact_parameters, error_absolute, 'o-',
                        lw=1.0, markersize=6,
                        color=self.colors[i % len(self.colors)],
                        label=method)

        ax1.set_xlabel('Impact Parameter')
        ax1.set_ylabel('Shapiro Delay Error (%)')
        ax1.set_title('Percentage Error in Shapiro Delay Calculation')
        ax1.set_yscale('log')
        ax1.grid(True)
        ax1.legend()

        ax2.set_xlabel('Impact Parameter')
        ax2.set_ylabel('Shapiro Delay Error (Absolute)')
        ax2.set_title('Absolute Error in Shapiro Delay Calculation')
        ax2.grid(True)
        ax2.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "shapiro_error_analysis.png"),
                dpi=300)
        plt.close()

    def plot_time_dilation_map(self, trajectory_data):
        """
        Create a visualization of time dilation along the trajectory.
        """
        positions = trajectory_data["positions"]
        times = trajectory_data["times"]
        flat_times = trajectory_data["flat_times"]
        
        dilation_factors = np.gradient(times) / np.gradient(flat_times)
        
        plt.figure(figsize=(10, 10))
    
        vmin = np.percentile(dilation_factors, 5)
        vmax = np.percentile(dilation_factors, 95)
        
        points = plt.scatter(positions[:, 0], positions[:, 1], 
                        c=dilation_factors, cmap='plasma', 
                        s=2, alpha=0.8, vmin=vmin, vmax=vmax)
        
        cbar = plt.colorbar(points)
        cbar.set_label('Time Dilation Factor')
        
        circle = plt.Circle((0, 0), R_S, color='black')
        plt.gca().add_patch(circle)
        
        photon_sphere = plt.Circle((0, 0), 3.0 * R_S / 2.0, fill=False, 
                                color='red', linestyle='--', alpha=0.6)
        plt.gca().add_patch(photon_sphere)
        plt.annotate('Photon Sphere', xy=(0, 3.0 * R_S / 2.0), 
                xytext=(0, 3.0 * R_S / 2.0 + 1), 
                ha='center', fontsize=8, color='red',
                # arrowprops=dict(arrowstyle='->', color='red', alpha=0.6)
                )
        
        impact_parameter = trajectory_data["impact_parameter"]
        plt.annotate(
            f'Impact Parameter b={impact_parameter}',
            xy=(0.95, 0.95),
            xycoords='axes fraction',
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
            ha='right', va='top'
        )
        
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Time Dilation Map - {trajectory_data["method"]} (b={trajectory_data["impact_parameter"]})')
        plt.axis('equal')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 
                            f"time_dilation_map_{trajectory_data['method'].replace(' ', '_')}_b{trajectory_data['impact_parameter']}.png"), 
                dpi=300)
        plt.close()
    
    def compare_proper_vs_coordinate_time(self, integrator_name, impact_parameter):
        """
        Compare proper time vs. coordinate time for a single trajectory.
        """
        # Trace with proper time
        traj = self.trace_single_trajectory(
            integrator_name, impact_parameter)
        
        plt.figure(figsize=(10, 6))
        
        plt.plot(traj["path_lengths"], traj["times"], 
            'b-', lw=1.0, label='Coordinate Time')
        plt.plot(traj["path_lengths"], traj["tau_stat"], 
            'r-', lw=1.0, label='Proper Time')
        plt.plot(traj["path_lengths"], traj["flat_times"], 
            'k--', lw=1.0, label='Flat Spacetime')
        
        plt.xlabel('Path Length')
        plt.ylabel('Time')
        plt.title(f'Proper vs. Coordinate Time - {integrator_name} (b={impact_parameter})')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 
                            f"time_comparison_{integrator_name.replace(' ', '_')}_b{impact_parameter}.png"), 
                dpi=300)
        plt.close()

    def extract_deflection_angles(self, all_results):
        """Extract deflection angles from existing trajectory data."""
        deflection_data = {}
        
        for b in all_results:
            deflection_data[b] = {}
            
            for method in all_results[b]:
                trajectory = all_results[b][method]
                
                v_initial = trajectory["positions"][0] - trajectory["positions"][2]
                v_final = trajectory["positions"][-1] - trajectory["positions"][-3]
                
                v_initial = v_initial / np.linalg.norm(v_initial)
                v_final = v_final / np.linalg.norm(v_final)
                
                cos_angle = np.dot(v_initial, v_final)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle) * 180/np.pi
                
                theoretical = 4*G*M/(C**2*b) * 180/np.pi
                
                if b < 20*R_S:
                    theoretical += np.pi**2*G*M/(2*C**2*b) * 4*G*M/(C**2*b) * 180/np.pi
                
                deflection_data[b][method] = {
                    "measured": angle,
                    "theoretical": theoretical,
                    "error_percent": 100 * abs(angle - theoretical) / theoretical
                }
        
        return deflection_data

    def photon_sphere_orbit_test(self, integrator_methods=None, orbit_periods=100.0):
        """
        Test stability of integrators for circular orbits at photon sphere.
        
        Args:
            integrator_methods (list, optional): List of methods to test
            orbit_periods (float): Number of orbit periods to simulate
            
        Returns:
            dict: Results of stability analysis for each method
        """
        
        r_photon = 3.0 * R_S / 2.0  # Photon sphere radius
        
        # For circular orbit at photon sphere, v_phi = c/sqrt(3)
        v_phi = C / math.sqrt(3.0)
        
        position = np.array([r_photon, 0, 0])
        velocity = np.array([0, v_phi, 0])
        # velocity[0] = -0.0001 * v_phi # tiny inward component
        
        results = {}
        
        methods = integrator_methods or list(self.integrators.keys())
        
        for method in methods:
            print(f"Starting {method} test...")
            # Relativistic orbital period at photon sphere
            # T = 2pi r_photon/v_phi 
            period = 2.0 * math.pi * r_photon * math.sqrt(3.0) / C
            # timestep = self.default_params[method]["timestep"]
            timestep = 0.005
            target_orbits = 100
            # base_stability_score = 0.0

            # maxsteps = int(period * orbit_periods / timestep) + 1000 # add some more points
            maxsteps = int(period * orbit_periods / timestep)
            
            result = self.trace_single_trajectory(
                method, impact_parameter=None,
                timestep=timestep, maxsteps=maxsteps,
                pos_init=position, vel_init=velocity, orbit_test=True, adaptive_stepping=True,
            )
            
            radii = np.linalg.norm(result["positions"], axis=1)
            radial_deviation = (radii - r_photon) / r_photon  
            
            L_deviation = (result["angular_momenta"] - result["angular_momenta"][0]) / result["angular_momenta"][0]
            energy_deviation = (result["energies"] - result["energies"][0]) / result["energies"][0]
            
            phi_values = np.arctan2(result["positions"][:, 1], result["positions"][:, 0])
            phi_unwrapped = np.unwrap(phi_values)
            
            orbits_completed = (phi_unwrapped[-1] - phi_unwrapped[0]) / (2.0 * math.pi)

            max_L_error = np.max(np.abs(L_deviation))
            max_energy_error = np.max(np.abs(energy_deviation))

            rms_radial_error = np.sqrt(np.mean(radial_deviation**2))
            max_radial_error = np.max(np.abs(radial_deviation))
            radial_score = 1.0 / (1.0 + max_radial_error)
            orbit_factor = min(abs(orbits_completed) / target_orbits, 1.0)
            efficiency = abs(orbits_completed) / len(result['positions'])

            r_start = radii[0]
            r_end = radii[-1]
            orbital_decay = (r_end - r_start) / r_start

            print(f"Method {method} completed with {len(result['positions'])} points")
            print(f"Termination reason: r={np.linalg.norm(position)}, periastron={result['periastron']}")

            results[method] = {
                "positions": result["positions"],
                "velocities": result["velocities"],
                "radial_deviation": radial_deviation,
                "max_radial_error": max_radial_error,
                "rms_radial_error": rms_radial_error,
                "L_deviation": L_deviation,
                "energy_deviation": energy_deviation,
                "max_L_error": max_L_error,
                "max_energy_error": max_energy_error,
                "rms_L_error": np.sqrt(np.mean(L_deviation**2)),
                "orbits_completed": orbits_completed,
                "orbital_decay": orbital_decay,
                "orbit_factor": orbit_factor,
                "efficiency": efficiency,
                # "orbit_stability_score": 1.0 / (1.0 + np.max(np.abs(radial_deviation)))
            }

        max_efficiency = 0.0
        for method, result in results.items():
            num_points = len(result["positions"])
            efficiency = abs(result["orbits_completed"]) / num_points
            max_efficiency = max(max_efficiency, efficiency)

        w_radial = 0.1     # Weight for radial error
        w_rms = 0.1        # Weight for RMS error
        w_e = 0.2          # Weight for energy conservation
        w_L = 0.2          # Weight for angular momentum conservation
        w_orbit = 0.2      # Weight for orbit completion
        w_efficiency = 0.2 # Weight for points efficiency
        # w_L = 0.25          # Weight for angular momentum conservation
        # w_orbit = 0.25      # Weight for orbit completion
        # w_efficiency = 0.3 # Weight for points efficiency

        for method, result in results.items():
            result["normalized_efficiency"] = result["efficiency"] / max_efficiency
            L_score = 1.0 / (1.0 + result["max_L_error"])
            energy_score = 1.0 / (1.0 + result["max_energy_error"])
            result["orbit_stability_score"] = ( w_radial * radial_score + 
                            w_rms * (1.0 / (1.0 + result["rms_radial_error"])) + 
                            w_e * energy_score + 
                            w_L * L_score + 
                            w_orbit * result["orbit_factor"] +
                            w_efficiency * result["efficiency"])

        min_val = min(abs(results[m]["orbit_stability_score"]) for m in methods)
        max_val = max(abs(results[m]["orbit_stability_score"]) for m in methods)

        for method, result in results.items():
            result["orbit_stability_score_normalized"] = (result["orbit_stability_score"] - min_val) / (max_val - min_val) * 100
        

        self.visualize_photon_sphere_test(results)
        # self.create_orbit_stability_dashboard(results)
        self.create_separate_orbit_stability_plots(results)
        self.create_stability_radar_chart(results)
        
        return results
    
    def visualize_photon_sphere_test(self, results):
        """
        Create visualization of photon sphere stability test results.
        
        Args:
            results (dict): Results from photon_sphere_orbit_test
        """
        methods = list(results.keys())

        plt.figure(figsize=(12, 12))

        circle = plt.Circle((0, 0), R_S, color='black', alpha=0.8)
        plt.gca().add_patch(circle)
        
        photon_sphere = plt.Circle((0, 0), 3.0 * R_S / 2.0, 
                                    fill=False, color='red', 
                                    linestyle='--', alpha=0.6)
        plt.gca().add_patch(photon_sphere)
        
        for i, method in enumerate(methods):
            positions = results[method]["positions"]
            plt.plot(positions[:, 0], positions[:, 1], '-', 
                    lw=1.0, color=self.colors[i % len(self.colors)], 
                    label=f"{method}")
        
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Photon Sphere Orbit Stability Test')
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "photon_sphere_test_orbits.png"), 
                    dpi=300)
        plt.close()

        plt.figure(figsize=(12, 8))
        
        x = np.arange(len(methods))
        width = 0.35
        
        radial_errors = [results[method]["max_radial_error"] * 100 for method in methods]
        L_errors = [results[method]["max_L_error"] * 100 for method in methods]
        
        plt.bar(x - width/2, radial_errors, width, label='Max Radial Error (%)')
        plt.bar(x + width/2, L_errors, width, label='Max Angular Momentum Error (%)')
        
        plt.xlabel('Integration Method')
        plt.ylabel('Error (%)')
        plt.title('Photon Sphere Orbit Stability - Error Comparison')
        plt.xticks(x, methods, rotation=45, ha='right')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "photon_sphere_test_errors.png"), 
                    dpi=300)
        plt.close()

    def create_stability_radar_chart(self, results):
        methods = list(results.keys())

        metrics = [
            'Radial Stability',
            'AM Conservation',
            'Orbit Completion',
            'Decay Resistance',
            'Orbit Stability'
        ]

        N = len(metrics)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

        plt.xticks(angles[:-1], metrics)

        ax.set_rlabel_position(0)
        plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)
        plt.ylim(0, 1)

        
        max_abs_rad_error = max(abs(results[m]["max_radial_error"]) for m in methods)
        max_abs_L_error = max(abs(results[m]["max_L_error"]) for m in methods)
        max_abs_orbits = max(abs(results[m]["orbits_completed"]) for m in methods)
        # max_abs_orbit_decay = max(abs(results[m]["orbital_decay"]) for m in methods)
        max_abs_orbit_decay = max(abs(results[m]["orbital_decay"]) for m in methods)
        max_abs_stability = max(abs(results[m]["orbit_stability_score_normalized"]) for m in methods)
        if max_abs_orbits == 0:
            max_abs_orbits = 1

        for i, method in enumerate(methods):
            # max_rad_error = results[method]["max_radial_error"]
            # rad_stability = 1.0 / (1.0 + max_rad_error * 10)

            # max_L_error = results[method]["max_L_error"]
            # am_conservation = 1.0 / (1.0 + max_L_error * 100)

            # orbit_completion = abs(results[method]["orbits_completed"]) / max_abs_orbits

            # decay_abs = abs(results[method]["orbital_decay"])
            # decay_resistance = 1.0 / (1.0 + decay_abs * 10)

            # stability_score = results[method]["orbit_stability_score"]
            rad_stability = abs(results[method]["max_radial_error"]) / max_abs_rad_error
            am_conservation = abs(results[method]["max_L_error"]) / max_abs_L_error
            orbit_completion = abs(results[method]["orbits_completed"]) / max_abs_orbits 
            decay_resistance = abs(results[method]["orbital_decay"]) / max_abs_orbit_decay
            stability_score = abs(results[method]["orbit_stability_score_normalized"]) / max_abs_stability

            values = [rad_stability, am_conservation, orbit_completion, decay_resistance, stability_score]
            values_plot = values + [values[0]]

            ax.plot(angles, values_plot, linewidth=2, linestyle='solid',
                    label=method, color=self.colors[i % len(self.colors)])
            ax.fill(angles, values_plot, alpha=0.1, color=self.colors[i % len(self.colors)])

        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title('Integration Method Stability Radar Chart', size=15, y=1.1)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "stability_radar_chart.png"), dpi=300)
        plt.close()

    def create_orbit_stability_dashboard(self, results):
        """Create plots for comparing photon sphere stability."""
        methods = list(results.keys())
        
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 3)
        
        # 1. Trajectory plot (top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        circle = plt.Circle((0, 0), R_S, color='black', alpha=0.8)
        ax1.add_patch(circle)
        photon_sphere = plt.Circle((0, 0), 3.0 * R_S / 2.0, 
                                fill=False, color='red', 
                                linestyle='--', alpha=0.6)
        ax1.add_patch(photon_sphere)
        
        for i, method in enumerate(methods):
            positions = results[method]["positions"]
            ax1.plot(positions[:, 0], positions[:, 1], '-', 
                    lw=1.0, color=self.colors[i % len(self.colors)], 
                    label=f"{method}")
        
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_title('Photon Sphere Orbits')
        ax1.set_aspect('equal')
        ax1.grid(True)
        ax1.legend()
        
        # 2. Error metrics bar chart (top-center)
        ax2 = fig.add_subplot(gs[0, 1])
        x = np.arange(len(methods))
        width = 0.3
        
        radial_errors = [results[method]["max_radial_error"] * 100 for method in methods]
        L_errors = [results[method]["max_L_error"] * 100 for method in methods]
        rms_radial = [results[method]["rms_radial_error"] * 100 for method in methods]
        
        ax2.bar(x - width, radial_errors, width, label='Max Radial Error (%)')
        ax2.bar(x, L_errors, width, label='Max Angular Momentum Error (%)')
        ax2.bar(x + width, rms_radial, width, label='RMS Radial Error (%)')
        
        ax2.set_xlabel('Integration Method')
        ax2.set_ylabel('Error (%)')
        ax2.set_title('Stability Error Metrics')
        ax2.set_xticks(x)
        ax2.set_xticklabels(methods, rotation=45, ha='right')
        ax2.legend()
        
        # 3. Orbits completed (top-right)
        ax3 = fig.add_subplot(gs[0, 2])
        orbits = [results[method]["orbits_completed"] for method in methods]
        stability = [results[method]["orbit_stability_score"] * 100 for method in methods]
        
        ax3.bar(methods, orbits, color='green', alpha=0.7)
        ax3.set_xlabel('Method')
        ax3.set_ylabel('Orbits Completed', color='green')
        ax3.tick_params(axis='y', labelcolor='green')
        ax3.set_title('Orbit Completion')
        ax3.set_xticklabels(methods, rotation=45, ha='right')
        
        ax3b = ax3.twinx()
        ax3b.plot(methods, stability, 'ro-', linewidth=2)
        ax3b.set_ylabel('Stability Score', color='red')
        ax3b.tick_params(axis='y', labelcolor='red')
        
        # 4. Radial deviation over time (middle row)
        ax4 = fig.add_subplot(gs[1, :])
        for i, method in enumerate(methods):
            steps = np.arange(len(results[method]["radial_deviation"]))
            steps_normalized = steps / max(1, len(steps) - 1)
            
            ax4.plot(steps_normalized, results[method]["radial_deviation"] * 100, 
                '-', lw=1.0, color=self.colors[i % len(self.colors)], 
                label=f"{method}")
        
        ax4.set_xlabel('Normalized Step')
        ax4.set_ylabel('Radial Deviation (%)')
        ax4.set_title('Radial Deviation During Integration')
        ax4.grid(True)
        ax4.legend()
        ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # 5. Angular momentum conservation (bottom-left)
        ax5 = fig.add_subplot(gs[2, 0])
        for i, method in enumerate(methods):
            if method != "Euler":
                steps = np.arange(len(results[method]["L_deviation"]))
                steps_normalized = steps / max(1, len(steps) - 1)
                
                ax5.plot(steps_normalized, results[method]["L_deviation"] * 100, 
                    '-', lw=1.0, color=self.colors[i % len(self.colors)], 
                    label=f"{method}")
        
        ax5.set_xlabel('Normalized Step')
        ax5.set_ylabel('Angular Momentum Deviation (%)')
        ax5.set_title('Angular Momentum Conservation')
        ax5.grid(True)
        ax5.legend()
        ax5.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # 6. Summary table (bottom-center and bottom-right)
        ax6 = fig.add_subplot(gs[2, 1:])
        ax6.axis('tight')
        ax6.axis('off')
        
        table_data = []
        for method in methods:
            table_data.append([
                method,
                f"{results[method]['max_radial_error']*100:.4f}%",
                f"{results[method]['rms_radial_error']*100:.4f}%",
                f"{results[method]['max_L_error']*100:.8f}%",
                f"{results[method]['orbits_completed']:.2f}",
                f"{results[method]['orbital_decay']*100:.4f}%",
                f"{results[method]['orbit_stability_score']:.4f}"
            ])
        
        column_labels = [
            'Method', 
            'Max Radial\nError (%)', 
            'RMS Radial\nError (%)',
            'Max AM\nError (%)', 
            'Orbits\nCompleted',
            'Orbital\nDecay (%)',
            'Stability\nScore'
        ]
        
        table = ax6.table(
            cellText=table_data, 
            colLabels=column_labels,
            loc='center',
            cellLoc='center'
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)
        
        ax6.set_title('Stability Metrics')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "photon_sphere_metrics.png"), 
                    dpi=300, bbox_inches='tight')
        plt.close()

    def create_separate_orbit_stability_plots(self, results, print_table=True):
        """Create separate plots for comparing photon sphere stability.
        
        Args:
            results: Dictionary of results for different methods
            print_table: If True, prints a formatted summary table to console
        """
        methods = list(results.keys())
        
        # 1. Trajectory plot
        fig1 = plt.figure(figsize=(10, 8))
        ax1 = fig1.add_subplot(111)
        circle = plt.Circle((0, 0), R_S, color='black', alpha=0.8)
        ax1.add_patch(circle)
        photon_sphere = plt.Circle((0, 0), 3.0 * R_S / 2.0, 
                                fill=False, color='red', 
                                linestyle='--', alpha=0.6)
        ax1.add_patch(photon_sphere)
        
        for i, method in enumerate(methods):
            positions = results[method]["positions"]
            ax1.plot(positions[:, 0], positions[:, 1], '-', 
                    lw=1.0, color=self.colors[i % len(self.colors)], 
                    label=f"{method} (Orbits: {results[method]["orbits_completed"]:.2f})")
        
        ax1.plot(3.0 * R_S / 2.0, 0, color='green', marker='o', markersize=4, linestyle='')

        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_title('Photon Sphere Orbits')
        ax1.set_aspect('equal')
        ax1.grid(True)
        ax1.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "ps_trajectories.png"), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Error metrics bar chart
        fig2 = plt.figure(figsize=(10, 8))
        ax2 = fig2.add_subplot(111)
        x = np.arange(len(methods))
        width = 0.3
        
        radial_errors = [results[method]["max_radial_error"] * 100 for method in methods]
        L_errors = [results[method]["max_L_error"] * 100 for method in methods]
        rms_radial = [results[method]["rms_radial_error"] * 100 for method in methods]
        
        ax2.bar(x - width, radial_errors, width, label='Max Radial Error (%)')
        ax2.bar(x, L_errors, width, label='Max Angular Momentum Error (%)')
        ax2.bar(x + width, rms_radial, width, label='RMS Radial Error (%)')
        
        ax2.set_xlabel('Integration Method')
        ax2.set_ylabel('Error (%)')
        ax2.set_title('Stability Error Metrics')
        ax2.set_xticks(x)
        ax2.set_xticklabels(methods, rotation=45, ha='right')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "ps_error_metrics.png"), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Orbits completed
        fig3 = plt.figure(figsize=(10, 8))
        ax3 = fig3.add_subplot(111)
        orbits = [results[method]["orbits_completed"] for method in methods]
        stability = [results[method]["orbit_stability_score"] * 100 for method in methods]
        
        ax3.bar(methods, orbits, color='green', alpha=0.7)
        ax3.set_xlabel('Method')
        ax3.set_ylabel('Orbits Completed', color='green')
        ax3.tick_params(axis='y', labelcolor='green')
        ax3.set_title('Orbit Completion')
        ax3.set_xticklabels(methods, rotation=45, ha='right')
        
        ax3b = ax3.twinx()
        ax3b.plot(methods, stability, 'ro-', linewidth=2)
        ax3b.set_ylabel('Stability Score', color='red')
        ax3b.tick_params(axis='y', labelcolor='red')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "ps_orbit_completion.png"), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Radial deviation over time
        fig4 = plt.figure(figsize=(15, 8))
        ax4 = fig4.add_subplot(111)
        for i, method in enumerate(methods):
            steps = np.arange(len(results[method]["radial_deviation"]))
            steps_normalized = steps / max(1, len(steps) - 1)
            
            ax4.plot(steps_normalized, results[method]["radial_deviation"] * 100, 
                '-', lw=1.0, color=self.colors[i % len(self.colors)], 
                label=f"{method}")
        
        ax4.set_xlabel('Normalized Step')
        ax4.set_ylabel('Radial Deviation (%)')
        ax4.set_title('Radial Deviation During Integration')
        ax4.grid(True)
        ax4.legend()
        ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "ps_radial_deviation.png"), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Angular momentum conservation
        fig5 = plt.figure(figsize=(15, 8))
        ax5 = fig5.add_subplot(111)
        for i, method in enumerate(methods):
            if method != "Euler":
                steps = np.arange(len(results[method]["L_deviation"]))
                steps_normalized = steps / max(1, len(steps) - 1)
                
                ax5.plot(steps_normalized, results[method]["L_deviation"] * 100, 
                    '-', lw=1.0, color=self.colors[i % len(self.colors)], 
                    label=f"{method}")
        
        ax5.set_xlabel('Normalized Step')
        ax5.set_ylabel('Angular Momentum Deviation (%)')
        ax5.set_title('Angular Momentum Conservation')
        ax5.grid(True)
        ax5.legend()
        ax5.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "ps_angular_momentum.png"), 
                    dpi=300, bbox_inches='tight')
        plt.close()

        # 6. Energy conservation
        fig5 = plt.figure(figsize=(15, 8))
        ax6 = fig5.add_subplot(111)
        for i, method in enumerate(methods):
            if method != "Euler":
                steps = np.arange(len(results[method]["energy_deviation"]))
                steps_normalized = steps / max(1, len(steps) - 1)
                
                ax6.plot(steps_normalized, results[method]["energy_deviation"] * 100, 
                    '-', lw=1.0, color=self.colors[i % len(self.colors)], 
                    label=f"{method}")
        
        ax6.set_xlabel('Normalized Step')
        ax6.set_ylabel('Energy Deviation (%)')
        ax6.set_title('Energy Conservation')
        ax6.grid(True)
        ax6.legend()
        ax6.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "ps_energy.png"), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. Summary table
        fig7 = plt.figure(figsize=(15, 8))
        ax7 = fig7.add_subplot(111)
        ax7.axis('tight')
        ax7.axis('off')
        
        table_data = []
        for method in methods:
            table_data.append([
                method,
                f"{results[method]['max_radial_error']*100:.4f}%",
                f"{results[method]['rms_radial_error']*100:.4f}%",
                f"{results[method]['max_L_error']*100:.9f}%",
                f"{results[method]['max_energy_error']*100:.9f}%",
                f"{results[method]['orbits_completed']:.2f}",
                # f"{results[method]['orbital_decay']*100:.4f}%",
                f"{len(results[method]["positions"])}",
                # f"{results[method]['orbit_stability_score']:.4f}"
                f"{results[method]['orbit_stability_score_normalized']:.1f}"
            ])
        
        column_labels = [
            'Method', 
            'Max Radial\nError (%)', 
            'RMS Radial\nError (%)',
            'Max AM\nError (%)', 
            'Max Energy\nError (%)', 
            'Orbits\nCompleted',
            # 'Orbital\nDecay (%)',
            'Points',
            # 'Stability\nScore',
            'Stability\nScore Normalized'
        ]
        
        if print_table:
            print("\n" + "="*100)
            print("PHOTON SPHERE STABILITY METRICS SUMMARY")
            print("="*100)
            
            console_columns = [
                'Method', 
                'Max Radial Error (%)', 
                'RMS Radial Error (%)',
                'Max AM Error (%)', 
                'Max Energy Error (%)', 
                'Orbits Completed',
                # 'Orbital Decay (%)',
                'Points',
                # 'Stability Score',
                'Stability Score Normalized'
            ]
            
            col_widths = [max(len(col), max([len(str(row[i])) for row in table_data]) + 2) 
                        for i, col in enumerate(console_columns)]
            
            header = "| " + " | ".join(f"{col:<{width}}" for col, width in zip(console_columns, col_widths)) + " |"
            print(header)
            print("-" * len(header))
            
            for row in table_data:
                print("| " + " | ".join(f"{cell:<{width}}" for cell, width in zip(row, col_widths)) + " |")
            
            print("-" * len(header))
            print(f"Total methods compared: {len(methods)}")
            
            best_method_idx = max(range(len(methods)), 
                                key=lambda i: results[methods[i]]["orbit_stability_score_normalized"])
            best_method = methods[best_method_idx]
            print(f"Best method by stability score: {best_method} " +
                f"(Score: {results[best_method]['orbit_stability_score_normalized']:.4f})")
            print("="*100 + "\n")
        
        table = ax7.table(
            cellText=table_data, 
            colLabels=column_labels,
            loc='center',
            cellLoc='center'
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)
        
        ax7.set_title('Stability Metrics')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "ps_metrics_table.png"), 
                    dpi=300, bbox_inches='tight')
        plt.close()

    def trace_single_trajectory(self, integrator_name, impact_parameter, 
                                timestep=None, maxsteps=None, use_proper_time=True, 
                                pos_init=None, vel_init=None, orbit_test=False, 
                                adaptive_stepping = True):
        """
        Trace a single photon trajectory using the specified integrator.
        
        Args:
            integrator_name (str): Name of the integrator to use
            impact_parameter (float): Impact parameter
            timestep (float, optional): Time step for integration
            maxsteps (int, optional): Maximum number of integration steps
            
        Returns:
            dict: Trajectory data including positions, velocities and conserved quantities
        """
        integrator = self.integrators[integrator_name]
        
        if timestep is None:
            timestep = self.default_params[integrator_name]["timestep"]
        if maxsteps is None:
            maxsteps = self.default_params[integrator_name]["maxsteps"]

        # adaptive_stepping = True
        min_timestep = timestep * 0.1
        max_timestep = timestep * 2.0
        current_timestep = timestep
        adaptation_threshold = 0.01
        
        if impact_parameter:
            pos_init, vel_init = self.calculate_initial_conditions(impact_parameter)
        
        positions = np.zeros((maxsteps, 3), dtype=np.float64)
        velocities = np.zeros((maxsteps, 3), dtype=np.float64)
        angular_momenta = np.zeros(maxsteps, dtype=np.float64)
        energies = np.zeros(maxsteps, dtype=np.float64)
        carter_constants = np.zeros(maxsteps, dtype=np.float64)

        y = np.zeros(6, dtype=np.float64)
        f = np.zeros(6, dtype=np.float64)
        y_temp = np.zeros(6, dtype=np.float64)
        k1 = np.zeros(6, dtype=np.float64)
        k2 = np.zeros(6, dtype=np.float64)
        k3 = np.zeros(6, dtype=np.float64)
        k4 = np.zeros(6, dtype=np.float64)
        
        position = pos_init.copy()
        velocity = vel_init.copy()

        L_vec, L_mag = self.calculate_angular_momentum(position, velocity)
        
        # First point
        positions[0] = position
        velocities[0] = velocity
        angular_momenta[0] = L_mag
        energies[0] = self.calculate_energy(position, velocity)
        carter_constants[0] = self.calculate_carter_constant(position, velocity, L_vec)
        
        initial_angular_momentum = L_mag
        initial_energy = energies[0]
        initial_carter = carter_constants[0]
        
        times = np.zeros(maxsteps, dtype=np.float64)
        flat_times = np.zeros(maxsteps, dtype=np.float64)
        tau_stat = np.zeros(maxsteps, dtype=np.float64)
        path_lengths = np.zeros(maxsteps, dtype=np.float64)
        shapiro_delays = np.zeros(maxsteps, dtype=np.float64)
        
        times[0] = 0.0
        flat_times[0] = 0.0
        path_lengths[0] = 0.0
        tau_stat[0] = 0.0

        # Trace trajectory
        i = 1
        periastron = float('inf')
        periastron_index = 0
        r_previous = np.linalg.norm(position)

        execution_time = time.time()
        
        while i < maxsteps:
            h2 = np.sum(np.cross(position, velocity)**2)
            
            old_position = position.copy()
            r_previous = np.linalg.norm(position)

            if adaptive_stepping:
                r_current = r_previous
                proximity_factor = min(1.0, (r_current - R_S) / (5 * R_S))
                if proximity_factor < adaptation_threshold:
                    current_timestep = max(min_timestep, timestep * proximity_factor)
                else:
                    current_timestep = min(max_timestep, 
                                        current_timestep * (1.0 + adaptation_threshold))
            
            integrator(y, f, y_temp, position, velocity, k1, k2, k3, k4, h2, current_timestep)

            dl = np.linalg.norm(position - old_position)
            path_lengths[i] = path_lengths[i-1] + dl

            flat_times[i] = flat_times[i-1] + dl

            r_new = np.linalg.norm(position)
            r_prev = np.linalg.norm(old_position)
            dr = r_new - r_prev
            dt = 0.0
            if r_prev > R_S:
                schwarz_factor = 1.0 - R_S / r_prev
                if schwarz_factor > 1e-12: 
                    inv_schwarz_factor = 1.0 / schwarz_factor

                    dt_squared = inv_schwarz_factor * (dl**2) + \
                                (R_S / r_prev) * (inv_schwarz_factor**2) * (dr**2)

                    if dt_squared >= 0:
                         dt = math.sqrt(dt_squared)
                         dtau_stat = math.sqrt(schwarz_factor) * dt
                    else:

                         dt = dl * inv_schwarz_factor # Fallback

                else:
                    dt = dl * 1e6
                    dtau_stat = 0.0


            times[i] = times[i-1] + dt
            tau_stat[i] = tau_stat[i-1] + dtau_stat

            # Calculate accumulated Shapiro delay
            # shapiro_delays[i] = times[i] - flat_times[i]
            shapiro_delays[i] = 1.5*(times[i] - flat_times[i])

            current_r = np.linalg.norm(position)
            if current_r < R_S:
                break
                
            # Photon too far
            if current_r > 100:
                break

            if orbit_test:
                if current_r >= 2:
                    break
                
            positions[i] = position
            velocities[i] = velocity
            
            L_vec, L_mag = self.calculate_angular_momentum(position, velocity)
            angular_momenta[i] = L_mag
            energies[i] = self.calculate_energy(position, velocity)
            carter_constants[i] = self.calculate_carter_constant(position, velocity, L_vec)

            if current_r < periastron:
                periastron = current_r
                periastron_index = i
                
            # if i > periastron_index and r > np.linalg.norm(pos_init):
                # break
                
            i += 1
            r_previous = r_new
    
        # Cut arrays to points calculated
        positions = positions[:i]
        velocities = velocities[:i]
        angular_momenta = angular_momenta[:i]
        energies = energies[:i]
        carter_constants = carter_constants[:i]
        # tau_stat = tau_stat[:i]
        
        # Normalize conserved quantities relative to initial values
        angular_momenta_rel = angular_momenta / initial_angular_momentum
        energies_rel = energies / initial_energy
        carter_constants_rel = carter_constants / initial_carter
        
        mean_angular_momentum = np.mean(angular_momenta)
        std_angular_momentum = np.std(angular_momenta)
        rel_std_angular_momentum = std_angular_momentum / mean_angular_momentum * 100
        
        mean_energy = np.mean(energies)
        std_energy = np.std(energies)
        rel_std_energy = std_energy / mean_energy * 100
        
        execution_time = time.time() - execution_time
        final_shapiro_delay = shapiro_delays[i-1] if i > 0 else 0.0

        return {
            "method": integrator_name,
            "impact_parameter": impact_parameter,
            "timestep": timestep,
            "maxsteps": maxsteps,
            "positions": positions[:i],
            "velocities": velocities[:i],
            "angular_momenta": angular_momenta[:i],
            "angular_momenta_rel": angular_momenta_rel[:i],
            "energies": energies[:i],
            "energies_rel": energies_rel[:i],
            "carter_constants": carter_constants[:i],
            "carter_constants_rel": carter_constants_rel[:i],
            "periastron": periastron,
            "periastron_index": periastron_index,
            "num_points": i,
            "statistics": {
                "mean_angular_momentum": mean_angular_momentum,
                "std_angular_momentum": std_angular_momentum,
                "rel_std_angular_momentum": rel_std_angular_momentum,
                "mean_energy": mean_energy,
                "std_energy": std_energy,
                "rel_std_energy": rel_std_energy
            },
            "times": times[:i],
            "flat_times": flat_times[:i],
            "tau_stat": tau_stat[:i],
            "path_lengths": path_lengths[:i],
            "shapiro_delays": shapiro_delays[:i],
            "final_shapiro_delay": final_shapiro_delay,
            "execution_time": execution_time
        }
    
    def test_single_method(self, integrator_name, impact_parameter=6.0, 
                      timestep=None, maxsteps=None, create_plots=True):
        """
        Test a single integration method with a specific impact parameter.
        
        Args:
            integrator_name (str): Name of the integrator to test
            impact_parameter (float): Impact parameter to use
            timestep (float, optional): Time step for integration
            maxsteps (int, optional): Maximum number of integration steps
            create_plots (bool): Whether to create plots (should be False in threads)
            
        Returns:
            dict: Trajectory data and analysis
        """
        print(f"Testing {integrator_name} with impact parameter {impact_parameter}...")

        if abs(impact_parameter - B_CRIT) < 0.5:
            print(f"  Near-critical impact parameter detected (B_CRIT ~= {B_CRIT:.4f})")
            
            if timestep is None:
                original_timestep = self.default_params[integrator_name]["timestep"]
                timestep = original_timestep * 0.25
                print(f"  Using reduced timestep: {timestep} (normal: {original_timestep})")
            
            if maxsteps is None:
                original_maxsteps = self.default_params[integrator_name]["maxsteps"]
                maxsteps = original_maxsteps * 4
                print(f"  Using increased maxsteps: {maxsteps} (normal: {original_maxsteps})")

        try:
            trajectory_data = self.trace_single_trajectory(
                integrator_name, impact_parameter, timestep, maxsteps
            )
            
            stats = trajectory_data["statistics"]
            print(f"  Number of points: {trajectory_data['num_points']}")
            print(f"  Radial range: {np.min(np.linalg.norm(trajectory_data['positions'], axis=1)):.4f} to "
                f"{np.max(np.linalg.norm(trajectory_data['positions'], axis=1)):.4f}")
            print(f"  Periastron: {trajectory_data['periastron']:.4f}")
            print(f"  Angular momentum conservation: Mean={stats['mean_angular_momentum']:.6f}, "
                f"StdDev={stats['std_angular_momentum']:.6f} ({stats['rel_std_angular_momentum']:.4f}%)")
            print(f"  Energy conservation: Mean={stats['mean_energy']:.6f}, "
                f"StdDev={stats['std_energy']:.6f} ({stats['rel_std_energy']:.4f}%)")
            
            if create_plots:
                self.plot_single_trajectory(trajectory_data)

                if "shapiro_delays" in trajectory_data:
                    self.plot_shapiro_delay(trajectory_data)
                    self.plot_time_dilation_map(trajectory_data)
        
            winding_angle = self.calculate_winding_angle(trajectory_data["positions"])
            print(f"  Winding angle: {winding_angle:.2f} degrees")
            
            return trajectory_data
            
        except Exception as e:
            print(f"Error testing {integrator_name}: {e}")
            traceback.print_exc()
            return None
    
    def analyze_conservation_with_periastron(self, all_results):
        """
        Analyze how conservation errors correlate with periastron distance.

        Args:
            all_results (dict): Results dictionary from analyze_impact_parameters
        """
        print("\nAnalyzing conservation errors vs periastron distance...")

        methods = set()
        for results in all_results.values():
            methods.update(results.keys())
        methods = sorted(methods)

        data = []

        for method in methods:
            periastrons = []
            am_errors = []
            energy_errors = []

            for b in all_results:
                if method in all_results[b]:
                    result = all_results[b][method]

                    periastrons.append(result["periastron"])
                    am_errors.append(result["statistics"]["rel_std_angular_momentum"])
                    energy_errors.append(result["statistics"]["rel_std_energy"])

            if periastrons:
                data.append({
                    "method": method,
                    "periastrons": periastrons,
                    "am_errors": am_errors,
                    "energy_errors": energy_errors
                })

        plt.figure(figsize=(12, 8))

        markers = ['o', 's', '^', 'd', 'p', '*', 'x']

        for i, item in enumerate(data):
            method = item["method"]
            marker = markers[i % len(markers)]

            plt.scatter(item["periastrons"], item["energy_errors"],
                    marker=marker, s=50,
                    color=self.colors[i % len(self.colors)],
                    label=f"{method} (Energy)")
            # plt.plot(item["periastrons"], item["energy_errors"],
            #         marker=marker, linestyle='-', linewidth=2, 
            #         markersize=5,
            #         color=self.colors[i % len(self.colors)],
            #         label=f"{method} (Energy)")

        plt.axvline(x=3.0, color='r', linestyle='--',
                label='Photon Sphere (3M)')

        plt.xlabel('Periastron Distance (r)')
        plt.ylabel('Conservation Error (%)')
        plt.title('Energy Conservation Error vs Periastron Distance')
        plt.grid(True)
        plt.legend()
        all_values_positive = all( all(e > 0 for e in item["energy_errors"]) for item in data)

        if all_values_positive:
            plt.yscale('log')
        else:
            print("Warning: Some conservation errors are zero or negative, using linear scale")

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "conservation_vs_periastron.png"), dpi=300)
        plt.close()
    
    def create_plots_for_results(self, all_results):
        """
        Create all plots for results after parallel processing is complete.
        This ensures all plotting happens in the main thread.
        
        Args:
            all_results (dict): Results from analyze_impact_parameters
        """
        print("\nCreating visualization plots...")
        
        for impact_parameter, methods_results in all_results.items():
            for method_name, result in methods_results.items():
                print(f"Current plot with settings: {impact_parameter}, {method_name}")
                try:
                    self.plot_single_trajectory(result)
                    
                    if "shapiro_delays" in result:
                        self.plot_shapiro_delay(result)
                        self.plot_time_dilation_map(result)
                        
                except Exception as e:
                    print(f"Error creating plots for {method_name} at b={impact_parameter}: {e}")
            
            if methods_results:
                try:
                    self.create_combined_plots(methods_results, impact_parameter)
                except Exception as e:
                    print(f"Error creating combined plots for b={impact_parameter}: {e}")

        try:
            self.create_impact_parameter_analysis(all_results)
            self.analyze_shapiro_accuracy(all_results)
            self.analyze_conservation_with_periastron(all_results)
            self.create_difference_plots(all_results, reference_method="Runge-Kutta 4")
            self.create_winding_angle_analysis(all_results)
            self.create_combined_time_dilation_maps(all_results)
            self.create_combined_time_dilation_maps_with_lines(all_results)
        except Exception as e:
            print(f"Error creating analysis plots: {e}")

    def plot_single_trajectory(self, trajectory_data):
        """
        Create plots for a single trajectory.
        
        Args:
            trajectory_data (dict): Trajectory data from trace_single_trajectory
        """
        method = trajectory_data["method"]
        impact_parameter = trajectory_data["impact_parameter"]
        positions = trajectory_data["positions"]

        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(2, 3)

        ax1 = fig.add_subplot(gs[0, 0], projection='3d')
        ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', lw=1.0)
        ax1.scatter(positions[0, 0], positions[0, 1], positions[0, 2], c='g', s=40, label='Start')
        ax1.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], c='r', s=40, label='End')
        
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        x = R_S * np.outer(np.cos(u), np.sin(v))
        y = R_S * np.outer(np.sin(u), np.sin(v))
        z = R_S * np.outer(np.ones(np.size(u)), np.cos(v))
        ax1.plot_surface(x, y, z, color='black', alpha=0.6)
        
        # Set equal aspect ratio
        max_range = np.max([
            np.max(positions[:, 0]) - np.min(positions[:, 0]),
            np.max(positions[:, 1]) - np.min(positions[:, 1]),
            np.max(positions[:, 2]) - np.min(positions[:, 2])
        ])
        mid_x = (np.max(positions[:, 0]) + np.min(positions[:, 0])) * 0.5
        mid_y = (np.max(positions[:, 1]) + np.min(positions[:, 1])) * 0.5
        mid_z = (np.max(positions[:, 2]) + np.min(positions[:, 2])) * 0.5
        ax1.set_xlim(mid_x - max_range * 0.5, mid_x + max_range * 0.5)
        ax1.set_ylim(mid_y - max_range * 0.5, mid_y + max_range * 0.5)
        ax1.set_zlim(mid_z - max_range * 0.5, mid_z + max_range * 0.5)
        
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title(f'3D Trajectory - {method} (b={impact_parameter})')
        ax1.legend()
        
        # 2D trajectory plots (X-Y plane)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(positions[:, 0], positions[:, 1], 'b-', lw=1.0)
        ax2.scatter(positions[0, 0], positions[0, 1], c='g', s=40, label='Start')
        ax2.scatter(positions[-1, 0], positions[-1, 1], c='r', s=40, label='End')

        circle = plt.Circle((0, 0), R_S, color='black', alpha=0.6)
        ax2.add_patch(circle)
        
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title(f'X-Y Projection - {method} (b={impact_parameter})')
        ax2.set_aspect('equal')
        ax2.grid(True)
        ax2.legend()
        
        # 2D trajectory plots (X-Z plane)
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(positions[:, 0], positions[:, 2], 'b-', lw=1.0)
        ax3.scatter(positions[0, 0], positions[0, 2], c='g', s=40, label='Start')
        ax3.scatter(positions[-1, 0], positions[-1, 2], c='r', s=40, label='End')
        
        circle = plt.Circle((0, 0), R_S, color='black', alpha=0.6)
        ax3.add_patch(circle)
        
        ax3.set_xlabel('X')
        ax3.set_ylabel('Z')
        ax3.set_title(f'X-Z Projection - {method} (b={impact_parameter})')
        ax3.set_aspect('equal')
        ax3.grid(True)
        ax3.legend()
        
        # Phase space plot (u vs u')
        ax4 = fig.add_subplot(gs[1, 0])
        
        r_values = np.linalg.norm(positions, axis=1)
        u_values = 1 / r_values
        
        phi_values = np.arctan2(positions[:, 1], positions[:, 0])
        phi_values = np.unwrap(phi_values)
        
        # Calculate du/dphi
        u_prime = np.zeros_like(u_values)
        u_prime[1:-1] = (u_values[2:] - u_values[:-2]) / (phi_values[2:] - phi_values[:-2])
        u_prime[0] = (u_values[1] - u_values[0]) / (phi_values[1] - phi_values[0])
        u_prime[-1] = (u_values[-1] - u_values[-2]) / (phi_values[-1] - phi_values[-2])
        
        ax4.plot(u_values, u_prime, 'b-', lw=1.0)
        ax4.scatter(u_values[0], u_prime[0], c='g', s=40, label='Start')
        ax4.scatter(u_values[-1], u_prime[-1], c='r', s=40, label='End')
        
        ax4.set_xlabel('u = 1/r')
        ax4.set_ylabel("u' = du/dphi")
        ax4.set_title(f'Phase Space - {method} (b={impact_parameter})')
        ax4.grid(True)
        ax4.legend()
        
        # Conservation plots
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot(range(len(trajectory_data["angular_momenta_rel"])), 
                trajectory_data["angular_momenta_rel"], 'b-', lw=1.0, label='Angular Momentum')
        ax5.plot(range(len(trajectory_data["energies_rel"])), 
                trajectory_data["energies_rel"], 'r-', lw=1.0, label='Energy')
        
        ax5.set_xlabel('Integration Step')
        ax5.set_ylabel('Relative Value')
        ax5.set_title(f'Conservation Laws - {method} (b={impact_parameter})')
        ax5.set_ylim(0.9, 1.1)  # 10% deviation range
        ax5.axhline(y=1.0, color='k', linestyle='--', alpha=0.3)
        ax5.grid(True)
        ax5.legend()
        
        # Radial profile
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.plot(range(len(r_values)), r_values, 'b-', lw=1.0)
        if trajectory_data["periastron_index"] < len(r_values):
            ax6.scatter(trajectory_data["periastron_index"], 
                      r_values[trajectory_data["periastron_index"]], 
                      c='m', s=80, label=f'Periastron (r={trajectory_data["periastron"]:.4f})')
        
        ax6.set_xlabel('Integration Step')
        ax6.set_ylabel('Radial Distance (r)')
        ax6.set_title(f'Radial Profile - {method} (b={impact_parameter})')
        ax6.axhline(y=R_S, color='k', linestyle='--', alpha=0.3, label='Schwarzschild Radius')
        ax6.grid(True)
        ax6.legend()
        
        plt.tight_layout()

        plt.savefig(os.path.join(self.output_dir, 
                              f"{method.replace(' ', '_')}_b{impact_parameter}.png"), 
                  dpi=300)
        plt.close()
    
    def test_all_methods(self, impact_parameter=6.0):
        """
        Test all integration methods with a specific impact parameter.
        
        Args:
            impact_parameter (float): Impact parameter to use
            
        Returns:
            dict: Results for each method
        """
        print(f"\nTesting all methods with impact parameter {impact_parameter}...\n")
        
        results = {}
        
        for method in self.integrators.keys():
            try:
                timestep = self.default_params[method]["timestep"]
                maxsteps = self.default_params[method]["maxsteps"]
                
                result = self.test_single_method(method, impact_parameter, timestep, maxsteps)
                if result is not None:
                    results[method] = result
                    
            except Exception as e:
                print(f"Error testing {method}: {e}")
                traceback.print_exc()
        
        if results:
            self.create_combined_plots(results, impact_parameter)
            
        return results
    
    def create_combined_plots(self, results, impact_parameter):
        """
        Create combined plots comparing all methods.
        
        Args:
            results (dict): Dictionary with results for each method
            impact_parameter (float): Impact parameter used
        """
        methods = list(results.keys())
        
        # First, 3D Plot
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(2, 2)
        
        ax1 = fig.add_subplot(gs[0, 0], projection='3d')

        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        x = R_S * np.outer(np.cos(u), np.sin(v))
        y = R_S * np.outer(np.sin(u), np.sin(v))
        z = R_S * np.outer(np.ones(np.size(u)), np.cos(v))
        ax1.plot_surface(x, y, z, color='black', alpha=0.6)

        max_ranges = []
        for i, method in enumerate(methods):
            positions = results[method]["positions"]
            ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                '-', lw=1.0, color=self.colors[i], label=method, 
                alpha=0.7)

            max_range = np.max([
                np.max(positions[:, 0]) - np.min(positions[:, 0]),
                np.max(positions[:, 1]) - np.min(positions[:, 1]),
                np.max(positions[:, 2]) - np.min(positions[:, 2])
            ])
            max_ranges.append(max_range)
        
        max_range = max(max_ranges) if max_ranges else 10
        ax1.set_xlim(-max_range * 0.6, max_range * 0.6)
        ax1.set_ylim(-max_range * 0.6, max_range * 0.6)
        ax1.set_zlim(-max_range * 0.6, max_range * 0.6)
        
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title(f'3D Trajectory Comparison (b={impact_parameter})')
        ax1.legend()
        
        # X-Z plane projection (2D)
        ax2 = fig.add_subplot(gs[0, 1])

        circle = plt.Circle((0, 0), R_S, color='black', alpha=0.6)
        ax2.add_patch(circle)

        for i, method in enumerate(methods):
            positions = results[method]["positions"]
            line_style = ['-', '--', '-.', ':'][i % 4]
            ax2.plot(positions[:, 0], positions[:, 2], 
                line_style, lw=1.0, color=self.colors[i], label=method,
                alpha=0.8)
        
        ax2.set_xlabel('X')
        ax2.set_ylabel('Z')
        ax2.set_title(f'X-Z Projection Comparison (b={impact_parameter})')
        ax2.set_aspect('equal')
        ax2.set_xlim(-max_range * 0.6, max_range * 0.6)
        ax2.set_ylim(-max_range * 0.6, max_range * 0.6)
        ax2.grid(True)
        ax2.legend()

        ax3 = fig.add_subplot(gs[1, 0])

        line_styles = ['-', '--', '-.', ':', '-', '--', '-.']
        markers = ['o', 's', '^', 'd', 'p', '*', 'x']

        for i, method in enumerate(methods):
            positions = results[method]["positions"]
            
            r_values = np.linalg.norm(positions, axis=1)
            u_values = 1 / r_values
            
            phi_values = np.arctan2(positions[:, 1], positions[:, 0])
            phi_values = np.unwrap(phi_values)

            u_prime = np.zeros_like(u_values)
            u_prime[1:-1] = (u_values[2:] - u_values[:-2]) / (phi_values[2:] - phi_values[:-2])
            u_prime[0] = (u_values[1] - u_values[0]) / (phi_values[1] - phi_values[0])
            u_prime[-1] = (u_values[-1] - u_values[-2]) / (phi_values[-1] - phi_values[-2])

            stride = max(1, len(u_values) // 50)
            
            ls = line_styles[i % len(line_styles)]
            marker = markers[i % len(markers)]
            ax3.plot(u_values[::stride], u_prime[::stride], 
                marker=marker, ls=ls, lw=1.0, markersize=4,
                color=self.colors[i], label=method)
        
        # Phase space plot (u vs u')
        # ax3 = fig.add_subplot(gs[1, 0])
        
        # for i, method in enumerate(methods):
        #     positions = results[method]["positions"]
            
        #     r_values = np.linalg.norm(positions, axis=1)
        #     u_values = 1 / r_values

        #     phi_values = np.arctan2(positions[:, 1], positions[:, 0])
        #     phi_values = np.unwrap(phi_values)
            
        #     u_prime = np.zeros_like(u_values)
        #     u_prime[1:-1] = (u_values[2:] - u_values[:-2]) / (phi_values[2:] - phi_values[:-2])
        #     u_prime[0] = (u_values[1] - u_values[0]) / (phi_values[1] - phi_values[0])
        #     u_prime[-1] = (u_values[-1] - u_values[-2]) / (phi_values[-1] - phi_values[-2])
            
        #     ax3.plot(u_values, u_prime, '-', lw=1.0, color=self.colors[i], label=method)
        
        ax3.set_xlabel('u = 1/r')
        ax3.set_ylabel("u' = du/dphi")
        ax3.set_title(f'Phase Space Comparison (b={impact_parameter})')
        ax3.grid(True)
        ax3.legend()
        
        ax4 = fig.add_subplot(gs[1, 1])
        
        bar_width = 0.35
        x = np.arange(len(methods))
        
        angular_momentum_errors = []
        energy_errors = []
        
        for method in methods:
            angular_momentum_errors.append(results[method]["statistics"]["rel_std_angular_momentum"])
            energy_errors.append(results[method]["statistics"]["rel_std_energy"])
        
        ax4.bar(x - bar_width/2, angular_momentum_errors, bar_width, label='Angular Momentum Error (%)')
        ax4.bar(x + bar_width/2, energy_errors, bar_width, label='Energy Error (%)')
        
        ax4.set_xlabel('Integration Method')
        ax4.set_ylabel('Relative Error (%)')
        ax4.set_title(f'Conservation Error Comparison (b={impact_parameter})')
        ax4.set_xticks(x)
        ax4.set_xticklabels(methods, rotation=45, ha='right')
        ax4.legend()
        
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.output_dir, f"combined_comparison_b{impact_parameter}.png"), 
                  dpi=300)
        plt.close()
        
        # Create conservation over time plot
        plt.figure(figsize=(12, 8))
        
        for i, method in enumerate(methods):
                steps = np.arange(len(results[method]["angular_momenta_rel"]))
                steps_normalized = steps / len(steps) * 100
                
                plt.plot(steps_normalized, results[method]["angular_momenta_rel"], 
                    '-', lw=1.0, color=self.colors[i], label=f"{method} (AM)")
        
        plt.xlabel('Integration Progress (%)')
        plt.ylabel('Relative Angular Momentum')
        plt.title(f'Angular Momentum Conservation (b={impact_parameter})')
        plt.grid(True)
        # plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.3)
        plt.legend()
        # plt.ylim(0.9, 1.1)  # 10% range
        ax = plt.gca()
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax.yaxis.offsetText.set_fontsize(10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"angular_momentum_conservation_b{impact_parameter}.png"), 
                  dpi=300)
        plt.close()
        
        # Create energy conservation over time plot
        # plt.figure(figsize=(12, 8))
        # ax = plt.gca()

        # for i, method in enumerate(methods):
        #     if method != "Euler":
        #         steps = np.arange(len(results[method]["energies_rel"]))
        #         steps_normalized = steps / len(steps) * 100
                
        #         ax.plot(steps_normalized, results[method]["energies_rel"], 
        #             '-', lw=1.0, color=self.colors[i], label=f"{method} (E)")

        # axins = ax.inset_axes([0.15, 0.2, 0.5, 0.3])

        # for i, method in enumerate(methods):
        #     if method != "Euler":
        #         steps = np.arange(len(results[method]["energies_rel"]))
        #         steps_normalized = steps / len(steps) * 100
                
        #         axins.plot(steps_normalized, results[method]["energies_rel"], 
        #             '-', lw=1.0, color=self.colors[i])

        # axins.set_xlim(20, 40)
        # y_min = min([min(results[method]["energies_rel"][int(len(results[method]["energies_rel"])*0.2):
        #                                                 int(len(results[method]["energies_rel"])*0.4)]) 
        #             for method in methods if method != "Euler"])
        # y_max = max([max(results[method]["energies_rel"][int(len(results[method]["energies_rel"])*0.2):
        #                                                 int(len(results[method]["energies_rel"])*0.4)]) 
        #             for method in methods if method != "Euler"])
        # # margin = (y_max - y_min) * 0.1  # 10% margin
        # axins.set_ylim(-1e-10, 1e-10)

        # axins.set_title('Detail: 20-40% Progress', fontsize=10)
        # axins.tick_params(labelsize=8)
        # axins.grid(True, alpha=0.5)

        # ax.indicate_inset_zoom(axins, edgecolor="black")

        # ax.set_xlabel('Integration Progress (%)')
        # ax.set_ylabel('Relative Energy')
        # ax.set_title(f'Energy Conservation (b={impact_parameter})')
        # ax.grid(True)
        # ax.legend()
        # ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        # ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        # ax.yaxis.offsetText.set_fontsize(10)

        # plt.tight_layout()
        # plt.savefig(os.path.join(self.output_dir, f"energy_conservation_b{impact_parameter}.png"), 
        #         dpi=300)
        # plt.close()
        plt.figure(figsize=(12, 8))
        
        for i, method in enumerate(methods):
            if method != "Euler":
                steps = np.arange(len(results[method]["energies_rel"]))
                steps_normalized = steps / len(steps) * 100
                
                plt.plot(steps_normalized, results[method]["energies_rel"], 
                    '-', lw=1.0, color=self.colors[i], label=f"{method} (E)")
        
        plt.xlabel('Integration Progress (%)')
        plt.ylabel('Relative Energy')
        plt.title(f'Energy Conservation (b={impact_parameter})')
        plt.grid(True)
        # plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.3)
        plt.legend()
        # plt.ylim(0.9, 1.1)  # 10% range
        ax = plt.gca()
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax.yaxis.offsetText.set_fontsize(10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"energy_conservation_b{impact_parameter}.png"), 
                  dpi=300)
        plt.close()
        
        # Create periastron comparison
        plt.figure(figsize=(10, 6))
        
        periastrons = [results[method]["periastron"] for method in methods]
        
        plt.bar(methods, periastrons, color=self.colors[:len(methods)])
        plt.axhline(y=3.0, color='r', linestyle='--', 
                  label='Critical Value (3M)')
        
        plt.xlabel('Integration Method')
        plt.ylabel('Periastron (r)')
        plt.title(f'Periastron Comparison (b={impact_parameter})')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, axis='y')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"periastron_comparison_b{impact_parameter}.png"), 
                  dpi=300)
        plt.close()

    def create_winding_angle_analysis(self, all_results):
        """
        Create plot showing winding angle vs impact parameter.
        """
        impact_parameters = sorted(all_results.keys())
        methods = set()
        for results in all_results.values():
            methods.update(results.keys())
        methods = sorted(methods)
        
        plt.figure(figsize=(12, 8))
        
        plt.axvline(x=B_CRIT, color='r', linestyle='--', 
                label=f'Critical Impact Parameter ({B_CRIT:.4f})')
        
        for i, method in enumerate(methods):
            winding_angles = []
            valid_impact_parameters = []
            
            for b in impact_parameters:
                if method in all_results[b]:
                    positions = all_results[b][method]["positions"]
                    winding_angle = self.calculate_winding_angle(positions)
                    winding_angles.append(winding_angle)
                    valid_impact_parameters.append(b)
            
            if valid_impact_parameters:
                plt.plot(valid_impact_parameters, winding_angles, 
                    'o-', lw=1.0, markersize=6, color=self.colors[i % len(self.colors)], 
                    label=method)
        
        plt.axhline(y=180, color='k', linestyle=':', label='180° (Direct Deflection)')
        plt.axhline(y=360, color='k', linestyle='-.', label='360° (Full Loop)')
        
        plt.xlabel('Impact Parameter (b)')
        plt.ylabel('Winding Angle (degrees)')
        plt.title('Photon Trajectory Winding Angle vs Impact Parameter')
        plt.grid(True)
        plt.legend()
        
        plt.savefig(os.path.join(self.output_dir, "winding_angle_vs_impact.png"), dpi=300)
        plt.close()

    def analyze_impact_parameters(self):
        """
        Analyze different impact parameters for all methods in parallel.
        """
        print("\nAnalyzing all impact parameters in parallel...\n")
        
        all_results = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            test_function = partial(self.test_all_methods_for_parallel)
            
            future_to_impact = {
                executor.submit(test_function, impact_parameter): impact_parameter 
                for impact_parameter in self.impact_parameters
            }
            
            for future in concurrent.futures.as_completed(future_to_impact):
                impact_parameter = future_to_impact[future]
                try:
                    results = future.result()
                    all_results[impact_parameter] = results
                    print(f"Completed analysis for impact parameter {impact_parameter}")
                except Exception as e:
                    print(f"Error processing impact parameter {impact_parameter}: {e}")
                    traceback.print_exc()
        
        self.create_plots_for_results(all_results)
        
        return all_results
    
    def analyze_critical_behavior(self, results, impact_parameter):
        """
        Analyze critical behavior for near-critical impact parameters.
        
        Args:
            results (dict): Results dictionary for all methods at a specific impact parameter
            impact_parameter (float): The impact parameter used
        """
        print(f"Analyzing critical behavior for impact parameter b = {impact_parameter}...")
        
        methods = list(results.keys())
        
        plt.figure(figsize=(12, 8))
        
        for i, method in enumerate(methods):
            positions = results[method]["positions"]
            
            initial_dir = positions[0] - positions[2]
            initial_dir = initial_dir / np.linalg.norm(initial_dir)
            
            final_dir = positions[-1] - positions[-3]
            final_dir = final_dir / np.linalg.norm(final_dir)
            
            cos_angle = np.dot(initial_dir, final_dir)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            deflection_angle = np.arccos(cos_angle) * 180 / np.pi

            x_values = positions[:, 0]
            crossings = np.where(np.diff(np.signbit(x_values)))[0]
            num_loops = len(crossings) / 2  # divide by 2 since each loop has 2 crossings
            
            print(f"  {method}: Deflection Angle = {deflection_angle:.2f}°, Loops ≈ {num_loops:.1f}")
            
            plt.plot(positions[:, 0], positions[:, 1], 
                   '-', lw=1.0, color=self.colors[i], 
                   label=f"{method} (θ={deflection_angle:.1f}°, Loops≈{num_loops:.1f})")
        
        circle = plt.Circle((0, 0), R_S, color='black', alpha=0.6)
        plt.gca().add_patch(circle)
        
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Critical Orbit Analysis (b={impact_parameter})')
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"critical_orbit_b{impact_parameter:.2f}.png"), dpi=300)
        plt.close()
    
    def plot_shapiro_delay(self, trajectory_data):
        """
        Plot the Shapiro time delay along the trajectory.
        """
        method = trajectory_data["method"]
        impact_parameter = trajectory_data["impact_parameter"]
        positions = trajectory_data["positions"]
        path_lengths = trajectory_data["path_lengths"]
        shapiro_delays = trajectory_data["shapiro_delays"]

        d1 = np.linalg.norm(positions[0])
        d2 = np.linalg.norm(positions[-1])
        theoretical_delay = self.calculate_theoretical_shapiro_delay(impact_parameter, d1, d2)


        plt.figure(figsize=(10, 6))
        plt.plot(path_lengths, shapiro_delays, 'b-', lw=1.0,
            label='Numerical Delay')
        plt.axhline(y=theoretical_delay, color='r', linestyle='--',
                label=f'Theoretical ({theoretical_delay:.4f})')

        plt.xlabel('Path Length')
        plt.ylabel('Shapiro Delay')
        plt.title(f'Shapiro Effect - {method} (b={impact_parameter})')
        plt.grid(True)
        plt.legend()

        plt.savefig(os.path.join(self.output_dir,
                            f"shapiro_delay_{method.replace(' ', '_')}_b{impact_parameter}.png"),
                dpi=300)
        plt.close()

    def create_impact_parameter_analysis(self, all_results):
        """
        Create analysis plots comparing results across impact parameters.
        
        Args:
            all_results (dict): Dictionary of results for all impact parameters and methods
        """
        print("\nCreating impact parameter analysis plots...\n")
        
        impact_parameters = sorted(all_results.keys())
        all_methods = set()
        for results in all_results.values():
            all_methods.update(results.keys())
        methods = sorted(all_methods)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        theoretical_b = np.linspace(min(impact_parameters) + 0.5, max(impact_parameters), 50)
        theoretical_periastron = [self.calculate_analytical_periastron(b) for b in theoretical_b]
        
        zoom_theoretical_b = np.linspace(2.5, 3.0, 50)
        zoom_theoretical_periastron = [self.calculate_analytical_periastron(b) for b in zoom_theoretical_b]
        
        print("Debug - Theoretical periastron values:")
        for b, p in zip(theoretical_b[:5], theoretical_periastron[:5]):
            print(f"b={b}: periastron={p}")

        ax.plot(theoretical_b, theoretical_periastron, 'k-', lw=1.1, 
            label='Theoretical r^3 - b^2*r + b^2*R_S = 0')
        
        line_styles = ['-', '--', '-.', ':', '-', '--', '-.']
        markers = ['o', 's', '^', 'd', 'p', '*', 'x']

        zoom_data = {}
        
        for i, method in enumerate(methods):
            periastrons = []
            valid_impact_parameters = []

            zoom_periastrons = []
            zoom_valid_impact_parameters = []
            
            for b in impact_parameters:
                if method in all_results[b]:
                    periastrons.append(all_results[b][method]["periastron"])
                    valid_impact_parameters.append(b)

                    if 2.5 <= b <= 3.0:
                        zoom_periastrons.append(all_results[b][method]["periastron"])
                        zoom_valid_impact_parameters.append(b)
            
            if valid_impact_parameters:
                ls = line_styles[i % len(line_styles)]
                marker = markers[i % len(markers)]
                
                ax.plot(valid_impact_parameters, periastrons, 
                    marker=marker, ls=ls, lw=1.0, markersize=6, 
                    color=self.colors[i % len(self.colors)], 
                    label=method)

                if zoom_valid_impact_parameters:
                    zoom_data[method] = {
                        'x': zoom_valid_impact_parameters,
                        'y': zoom_periastrons,
                        'style': ls,
                        'marker': marker,
                        'color': self.colors[i % len(self.colors)]
                    }
        
        axins = ax.inset_axes([0.55, 0.2, 0.4, 0.33])  # [x, y, w, h] in rel. coord
        
        axins.plot(zoom_theoretical_b, zoom_theoretical_periastron, 'k-', lw=1.0)

        for method, data in zoom_data.items():
            axins.plot(data['x'], data['y'],
                marker=data['marker'], ls=data['style'], lw=1.0, markersize=5, 
                color=data['color'])
        
        ax.axhline(y=R_S, color='k', linestyle='--', 
                label='Schwarzschild Radius')
        ax.axhline(y=3.0, color='r', linestyle='--', 
                label='Photon Sphere (3M)')
        
        axins.axhline(y=R_S, color='k', linestyle='--')

        weak_field_b = np.linspace(min(impact_parameters), max(impact_parameters), 100)
        weak_field_periastron = weak_field_b * (1.0 - R_S / weak_field_b)
        ax.plot(weak_field_b, weak_field_periastron, 'k:', lw=1.0, 
            label='Weak-Field Approximation')

        zoom_weak_field_b = np.linspace(2.5, 3.0, 100)
        zoom_weak_field_periastron = zoom_weak_field_b * (1.0 - R_S / zoom_weak_field_b)
        axins.plot(zoom_weak_field_b, zoom_weak_field_periastron, 'k:', lw=1.0)

        axins.set_xlim(2.5, 2.85)
        axins.set_ylim(0.9, 2.10)
  
        axins.set_title('b=2.5-2.85', fontsize=10)

        axins.tick_params(labelsize=8)
        
        ax.indicate_inset_zoom(axins, edgecolor="black")
        
        ax.set_xlabel('Impact Parameter (b)')
        ax.set_ylabel('Periastron Distance (r)')
        ax.set_title('Periastron vs Impact Parameter')
        ax.grid(True)
        ax.legend()

        axins.grid(True, alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "periastron_vs_impact.png"), dpi=300)
        plt.close()
        
        # plt.figure(figsize=(12, 8))
        
        # for i, method in enumerate(methods):
        #     angular_momentum_errors = []
        #     energy_errors = []
        #     valid_impact_parameters = []
            
        #     for b in impact_parameters:
        #         if method in all_results[b]:
        #             angular_momentum_errors.append(
        #                 all_results[b][method]["statistics"]["rel_std_angular_momentum"])
        #             energy_errors.append(
        #                 all_results[b][method]["statistics"]["rel_std_energy"])
        #             valid_impact_parameters.append(b)
            
        #     if valid_impact_parameters:
        #         plt.plot(valid_impact_parameters, angular_momentum_errors, 
        #             'o-', lw=1.0, markersize=6, color=self.colors[i % len(self.colors)], 
        #             alpha=0.9, label=f"{method} (AM)")
        #         plt.plot(valid_impact_parameters, energy_errors, 
        #             's--', lw=1.0, markersize=6, color=self.colors[i % len(self.colors)], 
        #             alpha=0.5, label=f"{method} (E)")
        
        # plt.xlabel('Impact Parameter (b)')
        # plt.ylabel('Relative Error (%)')
        # plt.title('Conservation Errors vs Impact Parameter')
        # plt.yscale('log')
        # plt.grid(True)
        # plt.legend()
        
        # plt.tight_layout()
        # plt.savefig(os.path.join(self.output_dir, "conservation_errors_vs_impact.png"), dpi=300)
        # plt.close()

        plt.figure(figsize=(12, 6))
        for i, method in enumerate(methods):
            angular_momentum_errors = []
            # angular_momentum_errors_std = []
            valid_impact_parameters = []
            
            for b in impact_parameters:
                if method in all_results[b]:
                    angular_momentum_errors.append(all_results[b][method]["statistics"]["rel_std_angular_momentum"])
                    # angular_momentum_errors_std.append(np.std(all_results[b][method]["statistics"]["rel_std_angular_momentum"]))
                    valid_impact_parameters.append(b)
            
            if valid_impact_parameters:
                plt.plot(valid_impact_parameters, angular_momentum_errors, 
                    'x-', lw=1.0, markersize=6, color=self.colors[i % len(self.colors)], 
                    alpha=0.9, label=f"{method}")
                # plt.errorbar(valid_impact_parameters, angular_momentum_errors, 
                #     yerr=angular_momentum_errors_std,
                #     fmt='o-', lw=1.0, markersize=3, color=self.colors[i % len(self.colors)], 
                #     alpha=0.9, label=f"{method}")

        plt.xlabel('Impact Parameter (b)')
        plt.ylabel('Relative Error (%)')
        plt.title('Angular Momentum Conservation Errors vs Impact Parameter')
        plt.yscale('log')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "angular_momentum_errors_vs_impact.png"), dpi=300)
        plt.close()

        # Energy errors
        plt.figure(figsize=(12, 6))
        for i, method in enumerate(methods):
            energy_errors = []
            # energy_errors_std = []
            valid_impact_parameters = []
            
            for b in impact_parameters:
                if method in all_results[b]:
                    energy_errors.append(all_results[b][method]["statistics"]["rel_std_energy"])
                    # energy_errors_std.append(np.std(all_results[b][method]["statistics"]["rel_std_energy"]))
                    valid_impact_parameters.append(b)
            
            if valid_impact_parameters:
                plt.plot(valid_impact_parameters, energy_errors, 
                    'x-', lw=1.0, markersize=6, color=self.colors[i % len(self.colors)], 
                    alpha=0.9, label=f"{method}")
                # plt.errorbar(valid_impact_parameters, energy_errors, 
                #              yerr=energy_errors_std,
                #             fmt='s-', lw=1.0, markersize=3, color=self.colors[i % len(self.colors)], 
                #             alpha=0.9, label=f"{method}")

        plt.xlabel('Impact Parameter (b)')
        plt.ylabel('Relative Error (%)')
        plt.title('Energy Conservation Errors vs Impact Parameter')
        plt.yscale('log')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "energy_errors_vs_impact.png"), dpi=300)
        plt.close()
        
        # Create deflection angle vs impact parameter plot
        plt.figure(figsize=(12, 8))
        for i, method in enumerate(methods):
            deflection_angles = []
            valid_impact_parameters = []
            
            for b in impact_parameters:
                if b <= 8:
                    if method in all_results[b]:
                        positions = all_results[b][method]["positions"]  

                        initial_dir = positions[0] - positions[2]
                        final_dir = positions[-1] - positions[-3]
                        

                        initial_dir = initial_dir / np.linalg.norm(initial_dir)
                        final_dir = final_dir / np.linalg.norm(final_dir)
                        
                        cos_angle = np.dot(initial_dir, final_dir)
                        cos_angle = max(min(cos_angle, 1.0), -1.0)
                        deflection = np.arccos(cos_angle) * 180 / np.pi
                        
                        deflection = 180 - deflection
                        
                        deflection_angles.append(deflection)
                        valid_impact_parameters.append(b)
                        # positions = all_results[b][method]["positions"]
                        # velocities = all_results[b][method]["velocities"]
                        
                        # # Calculate deflection angle
                        # # initial_dir = positions[0] - positions[1]
                        # # initial_dir = initial_dir / np.linalg.norm(initial_dir)
                        # initial_vel = velocities[0]
                        # final_vel = velocities[-1]
                        
                        # # final_dir = positions[-1] - positions[-2]
                        # # final_dir = final_dir / np.linalg.norm(final_dir)
                        
                        # # cos_angle = np.dot(initial_dir, final_dir)
                        # # cos_angle = max(min(cos_angle, 1.0), -1.0)
                        # initial_vel = initial_vel / np.linalg.norm(initial_vel)
                        # final_vel = final_vel / np.linalg.norm(final_vel)
                        # cos_angle = np.dot(initial_vel, final_vel)
                        # cos_angle = max(min(cos_angle, 1.0), -1.0)
                        # angle = np.arccos(cos_angle) * 180 / np.pi
                        # # deflection_angle = np.arccos(cos_angle) * 180 / np.pi
                        # deflection_angle = 180 - angle
                        
                        # deflection_angles.append(deflection_angle)
                        # valid_impact_parameters.append(b)
            
            if valid_impact_parameters:
                plt.plot(valid_impact_parameters, deflection_angles, 
                       'o-', lw=1.0, markersize=6, color=self.colors[i % len(self.colors)], 
                       label=method)
        
        # Theoretical curve for weak-field limit (small deflection angles)
        theoretical_b = np.linspace(6, 8, 20)
        theoretical_deflection = 2 * R_S / theoretical_b * 180 / np.pi
        plt.plot(theoretical_b, theoretical_deflection, 'k--', lw=1.0, 
               label='Theoretical (Weak-Field)')
        
        plt.xlabel('Impact Parameter (b)')
        plt.ylabel('Deflection Angle (°)')
        plt.title('Deflection Angle vs Impact Parameter')
        # plt.yscale('log')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "deflection_angle_vs_impact.png"), dpi=300)
        plt.close()
        
        # Create performance analysis plot
        plt.figure(figsize=(12, 8))
        
        for i, method in enumerate(methods):
            points_calculated = []
            valid_impact_parameters = []
            
            for b in impact_parameters:
                if method in all_results[b]:
                    points_calculated.append(all_results[b][method]["num_points"])
                    valid_impact_parameters.append(b)
            
            # if valid_impact_parameters:
            plt.plot(valid_impact_parameters, points_calculated, 
                       'o-', lw=1.0, markersize=6, color=self.colors[i % len(self.colors)], 
                       label=method)
        
        plt.xlabel('Impact Parameter (b)')
        plt.ylabel('Number of Points Calculated')
        plt.title('Computational Effort vs Impact Parameter')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "points_calculated_vs_impact.png"), dpi=300)
        plt.close()

        print("Attempting to create execution time plot...")
        try:
            plt.figure(figsize=(12, 8))
            
            valid_method_count = 0
            for i, method in enumerate(methods):
                execution_times = []
                valid_impact_parameters = []
                
                for b in impact_parameters:
                    if method in all_results[b]:
                        if "execution_time" in all_results[b][method]:
                            execution_times.append(all_results[b][method]["execution_time"])
                            valid_impact_parameters.append(b)
                            print(f"  Found execution time for {method}, b={b}: {all_results[b][method]['execution_time']:.4f}s")
                
                if valid_impact_parameters:
                    valid_method_count += 1
                    plt.plot(valid_impact_parameters, execution_times, 
                        'o-', lw=1.0, markersize=6, color=self.colors[i % len(self.colors)], 
                        label=method)
            
            if valid_method_count == 0:
                print("  No valid execution time data found!")
                plt.close()
                return
            
            plt.xlabel('Impact Parameter (b)')
            plt.ylabel('Execution Times (s)')
            plt.title('Execution Times vs Impact Parameter')
            plt.grid(True)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "execution_times_vs_impact.png"), dpi=300)
            print("  Execution time plot saved successfully!")
        except Exception as e:
            print(f"  Error creating execution time plot: {e}")
            import traceback
            traceback.print_exc()
        finally:
            plt.close()
        
        self.create_summary_table(all_results)
    
    def create_summary_table(self, all_results):
        """
        Create a summary table of all results.
        
        Args:
            all_results (dict): Dictionary of results for all impact parameters and methods
        """
        impact_parameters = sorted(all_results.keys())
        all_methods = set()
        for results in all_results.values():
            all_methods.update(results.keys())
        methods = sorted(all_methods)
        
        data = []
        
        for method in methods:
            method_data = {"Method": method}
            
            for b in impact_parameters:
                if method in all_results[b]:
                    result = all_results[b][method]
                    
                    am_error = result["statistics"]["rel_std_angular_momentum"]
                    e_error = result["statistics"]["rel_std_energy"]
                    
                    method_data[f"b={b}_periastron"] = result["periastron"]
                    method_data[f"b={b}_AM_error"] = am_error
                    method_data[f"b={b}_E_error"] = e_error
                    method_data[f"b={b}_points"] = result["num_points"]
                else:
                    method_data[f"b={b}_periastron"] = float('nan')
                    method_data[f"b={b}_AM_error"] = float('nan')
                    method_data[f"b={b}_E_error"] = float('nan')
                    method_data[f"b={b}_points"] = float('nan')
            
            data.append(method_data)
        
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(self.output_dir, "summary_table.csv"), index=False)

        simple_data = []
        
        for method in methods:
            method_data = {"Method": method}
            
            periastron_errors = []
            am_errors = []
            energy_errors = []
            
            for b in impact_parameters:
                if method in all_results[b]:
                    result = all_results[b][method]
                    
                    # if b > 5.2:
                    true_periastron = self.calculate_analytical_periastron(b)
                    periastron_error = abs((result["periastron"] - true_periastron) / true_periastron) * 100
                    periastron_errors.append(periastron_error)
                    
                    am_errors.append(result["statistics"]["rel_std_angular_momentum"])
                    energy_errors.append(result["statistics"]["rel_std_energy"])
            
            if periastron_errors:
                print(f"\nDiagnostic - {method} periastron errors:")
                for b in impact_parameters:
                    if method in all_results[b]:
                        result = all_results[b][method]
                        calc_p = result["periastron"]
                        true_p = self.calculate_analytical_periastron(b)
                        err = abs((calc_p - true_p) / true_p) * 100
                        print(f"  b={b}: Analytical={true_p:.4f}, Calculated={calc_p:.4f}, Error={err:.4f}%")

            if periastron_errors:
                method_data["Avg_Periastron_Error_%"] = np.mean(periastron_errors)
                method_data["Avg_Periastron_Error_Std%"] = np.std(periastron_errors)
            else:
                method_data["Avg_Periastron_Error_%"] = float('nan')
                
            method_data["Avg_AM_Error_%"] = np.mean(am_errors) if am_errors else float('nan')
            method_data["Avg_AM_Error_Std%"] = np.std(am_errors) if am_errors else float('nan')
            method_data["Avg_Energy_Error_%"] = np.mean(energy_errors) if energy_errors else float('nan')
            method_data["Avg_Energy_Error_Std%"] = np.std(energy_errors) if energy_errors else float('nan')
            
            if not np.isnan(method_data["Avg_Periastron_Error_%"]):
                method_data["Overall_Error"] = (
                    method_data["Avg_AM_Error_%"] + 
                    method_data["Avg_Energy_Error_%"] + 
                    method_data["Avg_Periastron_Error_%"]
                ) / 3
                method_data["Overall_Error_Std"] = np.std(np.array([
                    method_data["Avg_AM_Error_%"],
                    method_data["Avg_Energy_Error_%"],
                    method_data["Avg_Periastron_Error_%"]
                ]))
            else:
                method_data["Overall_Error"] = (
                    method_data["Avg_AM_Error_%"] + 
                    method_data["Avg_Energy_Error_%"]
                ) / 2
                method_data["Overall_Error_Std"] = np.std(np.array([
                    method_data["Avg_AM_Error_%"],
                    method_data["Avg_Energy_Error_%"]
                ]))
            
            simple_data.append(method_data)
        
        simple_df = pd.DataFrame(simple_data)
        simple_df = simple_df.sort_values("Overall_Error")
        simple_df.to_csv(os.path.join(self.output_dir, "summary_simple.csv"), index=False)
        
        print("\nMethod Performance Summary (sorted by overall error):")
        print(simple_df.to_string(index=False))
        
        return simple_df
    
    def analyze_critical_shapiro(self):
        """
        Analyze Shapiro delay for a range of near-critical impact parameters.
        """
        
        # Impact parameters very close to critical
        near_critical = np.linspace(B_CRIT + 0.01, B_CRIT + 0.5, 10)
        
        method = "Obrechkoff"
        delays = []
        
        for b in near_critical:
            print(f"Analyzing near-critical b = {b}...")
            result = self.trace_single_trajectory(method, b)
            final_delay = result["shapiro_delays"][-1]
            delays.append(final_delay)
        
        plt.figure(figsize=(10, 6))
        plt.plot(near_critical - B_CRIT, delays, 'o-', lw=1.0)
        plt.xlabel('Impact Parameter Offset from Critical Value')
        plt.ylabel('Shapiro Delay')
        plt.title('Shapiro Delay Near Critical Impact Parameter')
        plt.grid(True)
        
        plt.savefig(os.path.join(self.output_dir, "shapiro_near_critical.png"), dpi=300)
        plt.close()
    
    def test_all_methods_for_parallel(self, impact_parameter):
        """
        Thread-safe version of test_all_methods for parallel execution.
        
        Args:
            impact_parameter (float): Impact parameter to use
            
        Returns:
            dict: Results for each method
        """
        print(f"\nTesting all methods with impact parameter {impact_parameter}...")
        
        results = {}
        
        for method in self.integrators.keys():
            try:
                timestep = self.default_params[method]["timestep"]
                maxsteps = self.default_params[method]["maxsteps"]
                
                result = self.test_single_method(
                    method, 
                    impact_parameter, 
                    timestep, 
                    maxsteps,
                    create_plots=False
                )
                if result is not None:
                    results[method] = result
                    
            except Exception as e:
                print(f"Error testing {method} with impact parameter {impact_parameter}: {e}")
                traceback.print_exc()

        return results
    
    def create_shapiro_error_by_method_plot(self, all_results):
        """
        Create a plot showing Shapiro delay error percentage by method across impact parameters.
        """
        print("Creating Shapiro Delay Error by Method plot...")
        
        impact_parameters = sorted(all_results.keys())
        methods = set()
        for results in all_results.values():
            methods.update(results.keys())
        methods = sorted(methods)
        
        plt.figure(figsize=(12, 8))
        
        for i, method in enumerate(methods):
            error_percentages = []
            valid_impact_parameters = []
            
            for b in impact_parameters:
                if method in all_results[b]:
                    result = all_results[b][method]
                    if "shapiro_delays" in result:
                        final_delay = result["shapiro_delays"][-1]
                        
                        theoretical = self.calculate_theoretical_shapiro_delay(
                            b, 
                            np.linalg.norm(result["positions"][0]),
                            np.linalg.norm(result["positions"][-1])
                        )
                        
                        error_percent = abs((final_delay - theoretical) / theoretical) * 100
                        
                        error_percentages.append(error_percent)
                        valid_impact_parameters.append(b)
            
            if valid_impact_parameters:
                plt.plot(valid_impact_parameters, error_percentages, 'o-', 
                    lw=1.0, markersize=6, 
                    color=self.colors[i % len(self.colors)],
                    label=method)
        
        plt.xlabel('Impact Parameter')
        plt.ylabel('Shapiro Delay Error (%)')
        plt.title('Shapiro Effect Calculation Error by Method')
        plt.yscale('log')
        plt.grid(True)
        plt.legend(loc='lower right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "shapiro_error_by_method.png"), 
                dpi=300)
        plt.close()
    
    def create_combined_time_dilation_maps(self, all_results):
        """
        Create combined time dilation maps for each method, showing trajectories 
        for all impact parameters with the same color scale.
        
        Args:
            all_results (dict): Results dictionary from analyze_impact_parameters
        """
        print("\nCreating combined time dilation maps...")
        
        all_methods = set()
        for results in all_results.values():
            all_methods.update(results.keys())
        methods = sorted(all_methods)
        
        impact_parameters = sorted(all_results.keys())
        
        for method in methods:
            trajectories = []
            
            for b in impact_parameters:
                if method in all_results[b] and "times" in all_results[b][method]:
                    traj_data = all_results[b][method]
                    
                    positions = traj_data["positions"]
                    times = traj_data["times"]
                    flat_times = traj_data["flat_times"]
                    
                    if len(positions) > 10 and len(times) == len(positions) and len(flat_times) == len(positions):
                        
                        dilation_factors = np.gradient(times) / np.gradient(flat_times)
                        trajectories.append({
                            "impact_parameter": b,
                            "positions": positions,
                            "dilation_factors": dilation_factors
                        })
            
            if not trajectories:
                continue
                
            plt.figure(figsize=(12, 12))
            
            all_dilation_factors = np.concatenate([t["dilation_factors"] for t in trajectories])
            vmin = np.percentile(all_dilation_factors, 5)
            vmax = np.percentile(all_dilation_factors, 95)

            circle = plt.Circle((0, 0), R_S, color='black')
            plt.gca().add_patch(circle)
            
            for traj in trajectories:
                b = traj["impact_parameter"]
                positions = traj["positions"]
                dilation_factors = traj["dilation_factors"]
                
                points = plt.scatter(
                    positions[:, 0], positions[:, 1], 
                    c=dilation_factors, cmap='plasma', 
                    s=2, alpha=0.8, vmin=vmin, vmax=vmax
                )
                
                plt.annotate(
                    f'b={b}', 
                    xy=(positions[0, 0], positions[0, 1]),
                    xytext=(positions[0, 0] - 0.5, positions[0, 1] + 0.5),
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
                )
            
            cbar = plt.colorbar(points)
            cbar.set_label('Time Dilation Factor')
            
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title(f'Combined Time Dilation Map - {method}')
            plt.axis('equal')
            plt.grid(True)
            
            custom_lines = []
            legend_labels = []
            
            for traj in trajectories:
                custom_lines.append(Line2D([0], [0], color='black', marker='o', 
                                        markersize=5, linestyle='None'))
                legend_labels.append(f'b={traj["impact_parameter"]}')
                
            plt.legend(custom_lines, legend_labels, 
                    title='Impact Parameters', 
                    loc='upper right',
                    bbox_to_anchor=(1.1, 1.0))
            
            plt.tight_layout()
            
            filename = f"combined_time_dilation_{method.replace(' ', '_')}.png"
            plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
            plt.close()
            
            print(f"Created combined time dilation map for {method}")

    def create_combined_time_dilation_maps_with_lines(self, all_results):
        """
        Create combined time dilation maps with trajectory lines for each method.
        This version shows the trajectories as lines with color indicating time dilation.
        
        Args:
            all_results (dict): Results dictionary from analyze_impact_parameters
        """
        print("\nCreating combined time dilation maps with trajectory lines...")
        
        all_methods = set()
        for results in all_results.values():
            all_methods.update(results.keys())
        methods = sorted(all_methods)
        
        impact_parameters = sorted(all_results.keys())
        
        for method in methods:
            trajectories = []
            
            for b in impact_parameters:
                if method in all_results[b] and "times" in all_results[b][method]:
                    traj_data = all_results[b][method]

                    positions = traj_data["positions"]
                    times = traj_data["times"]
                    flat_times = traj_data["flat_times"]

                    if len(positions) > 10 and len(times) == len(positions) and len(flat_times) == len(positions):
                        
                        dilation_factors = np.gradient(times) / np.gradient(flat_times)
                    
                        trajectories.append({
                            "impact_parameter": b,
                            "positions": positions,
                            "dilation_factors": dilation_factors
                        })
            
            if not trajectories:
                continue
                
            fig, ax = plt.subplots(figsize=(12, 12))
            
            all_dilation_factors = np.concatenate([t["dilation_factors"] for t in trajectories])
            vmin = np.percentile(all_dilation_factors, 5)
            vmax = np.percentile(all_dilation_factors, 95)
            
            circle = plt.Circle((0, 0), R_S, color='black')
            ax.add_patch(circle)
            
            # Plot photon sphere (r=3M)
            photon_sphere = plt.Circle((0, 0), 3.0 * R_S / 2.0, fill=False, 
                                    color='red', linestyle='--', alpha=0.6)
            ax.add_patch(photon_sphere)
            ax.annotate('Photon Sphere', xy=(0, 3.0 * R_S / 2.0), 
                        xytext=(0, 3.0 * R_S / 2.0 + 1), 
                        ha='center', fontsize=8, color='red',
                        # arrowprops=dict(arrowstyle='->', color='red', alpha=0.6)
                        )
            
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            cmap = plt.get_cmap('plasma')
            
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            
            for traj in trajectories:
                b = traj["impact_parameter"]
                positions = traj["positions"]
                dilation_factors = traj["dilation_factors"]

                points = np.array([positions[:, 0], positions[:, 1]]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                               
                lc = LineCollection(segments, cmap='plasma', norm=norm)
                lc.set_array(dilation_factors[:-1])
                lc.set_linewidth(2)
                line = ax.add_collection(lc)
                
                ax.plot(positions[0, 0], positions[0, 1], 'o', 
                        markersize=6, color='green')

                ax.annotate(
                    f'b={b}', 
                    xy=(positions[0, 0], positions[0, 1]),
                    xytext=(positions[0, 0] - 1, positions[0, 1] + 1),
                    fontsize=9,
                    fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
                )
                
            cbar = fig.colorbar(sm, ax=ax)
            cbar.set_label('Time Dilation Factor')
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title(f'Combined Time Dilation Map - {method}')
            ax.set_aspect('equal')
            ax.grid(True)

            legend_elements = []
            
            for traj in trajectories:
                legend_elements.append(
                    Line2D([0], [0], marker='o', color='w', 
                        markerfacecolor='green', markersize=6, 
                        label=f'b={traj["impact_parameter"]}')
                )
            
            ax.legend(handles=legend_elements, 
                    title='Impact Parameters', 
                    loc='upper right')
            
            plt.tight_layout()
            
            filename = f"combined_time_dilation_lines_{method.replace(' ', '_')}.png"
            plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
            plt.close()
            
            print(f"Created combined time dilation map with lines for {method}")

    def create_difference_plots(self, all_results, reference_method="Runge-Kutta 4"):
        """
        Create plots showing differences between each method and a reference method.
        
        Args:
            all_results (dict): Results dictionary from analyze_impact_parameters
            reference_method (str): Method to use as reference
        """
        print(f"\nCreating difference plots (reference: {reference_method})...")
        
        impact_parameters = sorted(all_results.keys())
        all_methods = set()
        for results in all_results.values():
            all_methods.update(results.keys())
        methods = sorted([m for m in all_methods if m != reference_method])
        
        # 1. Periastron difference plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        line_styles = ['-', '--', '-.', ':', '-', '--', '-.']
        markers = ['o', 's', '^', 'd', 'p', '*', 'x']
        
        zoom_data = {}
        
        for i, method in enumerate(methods):
            periastron_diffs = []
            valid_impact_parameters = []
            
            zoom_periastron_diffs = []
            zoom_valid_impact_parameters = []
            
            for b in impact_parameters:
                if b < 7:
                    if method in all_results[b] and reference_method in all_results[b]:
                        ref_periastron = all_results[b][reference_method]["periastron"]
                        periastron = all_results[b][method]["periastron"]
                        
                        perc_diff = 100 * (periastron - ref_periastron) / ref_periastron
                        
                        periastron_diffs.append(perc_diff)
                        valid_impact_parameters.append(b)
                        
                        if 2.5 <= b <= 3.0:
                            zoom_periastron_diffs.append(perc_diff)
                            zoom_valid_impact_parameters.append(b)
            
            if valid_impact_parameters:
                ls = line_styles[i % len(line_styles)]
                marker = markers[i % len(markers)]
                
                ax.plot(valid_impact_parameters, periastron_diffs, 
                    marker=marker, ls=ls, lw=1.0, markersize=6, 
                    color=self.colors[i % len(self.colors)], 
                    label=f"{method} vs {reference_method}")
                
                if zoom_valid_impact_parameters:
                    zoom_data[method] = {
                        'x': zoom_valid_impact_parameters,
                        'y': zoom_periastron_diffs,
                        'style': ls,
                        'marker': marker,
                        'color': self.colors[i % len(self.colors)]
                    }
        
        axins = ax.inset_axes([0.4, 0.2, 0.4, 0.33])  # [x, y, w, h] in rel. coord
        
        for method, data in zoom_data.items():
            axins.plot(data['x'], data['y'],
                marker=data['marker'], ls=data['style'], lw=1.0, markersize=5, 
                color=data['color'])

        axins.set_xlim(2.55, 2.85)
        axins.set_ylim(0, 0.06)
        
        axins.set_title('b=2.55-2.85, diff=0-0.06%', fontsize=10)
        axins.tick_params(labelsize=8)
        
        ax.indicate_inset_zoom(axins, edgecolor="black")
        
        ax.set_xlabel('Impact Parameter (b)')
        ax.set_ylabel('Periastron Difference (%)')
        ax.set_title(f'Periastron Difference vs Reference Method ({reference_method})')
        ax.grid(True)
        ax.legend()
        
        axins.grid(True, alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "periastron_difference.png"), dpi=300)
        plt.close()
        
        # 2. Deflection angle difference plot
        fig, ax = plt.subplots(figsize=(12, 8))
    
        line_styles = ['-', '--', '-.', ':', '-', '--', '-.']
        markers = ['o', 's', '^', 'd', 'p', '*', 'x']
        
        zoom_data = {}
        
        for i, method in enumerate(methods):
            angle_diffs = []
            valid_impact_parameters = []
            
            zoom_angle_diffs = []
            zoom_valid_impact_parameters = []
            
            for b in impact_parameters:
                if b < 7:
                    if method in all_results[b] and reference_method in all_results[b]:
                        ref_positions = all_results[b][reference_method]["positions"]
                        positions = all_results[b][method]["positions"]
                        
                        # Calculate reference deflection angle
                        initial_dir_ref = ref_positions[0] - ref_positions[1]
                        initial_dir_ref = initial_dir_ref / np.linalg.norm(initial_dir_ref)
                        
                        final_dir_ref = ref_positions[-1] - ref_positions[-2]
                        final_dir_ref = final_dir_ref / np.linalg.norm(final_dir_ref)
                        
                        cos_angle_ref = np.dot(initial_dir_ref, final_dir_ref)
                        cos_angle_ref = max(min(cos_angle_ref, 1.0), -1.0)
                        deflection_angle_ref = np.arccos(cos_angle_ref) * 180 / np.pi
                        
                        # Calculate method deflection angle
                        initial_dir = positions[0] - positions[1]
                        initial_dir = initial_dir / np.linalg.norm(initial_dir)
                        
                        final_dir = positions[-1] - positions[-2]
                        final_dir = final_dir / np.linalg.norm(final_dir)
                        
                        cos_angle = np.dot(initial_dir, final_dir)
                        cos_angle = max(min(cos_angle, 1.0), -1.0)
                        deflection_angle = np.arccos(cos_angle) * 180 / np.pi
                        
                        angle_diff = deflection_angle - deflection_angle_ref
                        
                        angle_diffs.append(angle_diff)
                        valid_impact_parameters.append(b)
                        
                        if 2.55 <= b <= 2.8:
                            zoom_angle_diffs.append(angle_diff)
                            zoom_valid_impact_parameters.append(b)
            
            if valid_impact_parameters:
                ls = line_styles[i % len(line_styles)]
                marker = markers[i % len(markers)]
                
                ax.plot(valid_impact_parameters, angle_diffs, 
                    marker=marker, ls=ls, lw=1.0, markersize=6, 
                    color=self.colors[i % len(self.colors)], 
                    label=f"{method} vs {reference_method}")
                
                if zoom_valid_impact_parameters:
                    zoom_data[method] = {
                        'x': zoom_valid_impact_parameters,
                        'y': zoom_angle_diffs,
                        'style': ls,
                        'marker': marker,
                        'color': self.colors[i % len(self.colors)]
                    }
        
        axins = ax.inset_axes([0.25, 0.1, 0.5, 0.4])  # [x, y, w, h] in rel. coord
        
        for method, data in zoom_data.items():
            axins.plot(data['x'], data['y'],
                marker=data['marker'], ls=data['style'], lw=1.0, markersize=5, 
                color=data['color'])
        
        axins.set_xlim(2.55, 2.8)
        axins.set_ylim(-0.5, 0.5)
        
        axins.set_title('b=2.55-2.8, diff=-0.5° to 0.5°', fontsize=10)
        axins.tick_params(labelsize=8)
        
        ax.indicate_inset_zoom(axins, edgecolor="black")
        
        ax.set_xlabel('Impact Parameter (b)')
        ax.set_ylabel('Deflection Angle Difference (degrees)')
        ax.set_title(f'Deflection Angle Difference vs Reference Method ({reference_method})')
        ax.grid(True)
        ax.legend()
        
        axins.grid(True, alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "deflection_angle_difference.png"), dpi=300)
        plt.close()

    def run_full_analysis(self):
        """
        Run a full trajectory analysis for all integration methods and impact parameters.
        """
        print("Starting full trajectory analysis...")
        start_time = time.time()

        std_impact = 6.0
        all_results = None
        # results = self.test_all_methods(std_impact)

        # print("\nComparing proper vs. coordinate time...")
        # for method in self.integrators.keys():
        #     print(f"{method}: Impact parameters: 2.8, 6.0, 15.0")
        #     self.compare_proper_vs_coordinate_time(method, 2.8)
        #     self.compare_proper_vs_coordinate_time(method, 6.0)
        #     self.compare_proper_vs_coordinate_time(method, 15.0)
        
        all_results = self.analyze_impact_parameters()

        # print("\nCreating combined time dilation maps...")
        # self.create_combined_time_dilation_maps(all_results)
        # self.create_combined_time_dilation_maps_with_lines(all_results)
        # for impact_parameter in self.impact_parameters:
        #     if abs(impact_parameter - B_CRIT) < 0.1:  # Near-critical detection
        #         if impact_parameter in all_results:
        #             self.analyze_critical_behavior(all_results[impact_parameter], impact_parameter)
        # print("\nCreating final report...")
        # self.create_final_report(all_results)
        self.analyze_critical_shapiro()
        self.create_shapiro_error_by_method_plot(all_results)
        # self.photon_sphere_orbit_test()
        # self.extract_deflection_angles(all_results)

        elapsed_time = time.time() - start_time
        print(f"\nAnalysis complete. Total execution time: {elapsed_time:.2f} seconds")
        print(f"Results saved to {self.output_dir}/")
        return all_results
    
    # This is generated and not correct
    def create_final_report(self, all_results):
        """
        Create a final report summarizing the analysis.
        
        Args:
            all_results (dict): Dictionary of results for all impact parameters and methods
        """

        summary_df = self.create_summary_table(all_results)
        
        with open(os.path.join(self.output_dir, "final_report.md"), 'w', encoding='utf-8') as f:
            f.write("# Black Hole Trajectory Analysis Report\n\n")
            
            f.write("## Overview\n\n")
            f.write("This report analyzes the performance and accuracy of different numerical ")
            f.write("integration methods for simulating photon trajectories around a ")
            f.write("Schwarzschild black hole. The analysis focuses on conservation of physical ")
            f.write("quantities (angular momentum and energy) and the accuracy of the ")
            f.write("calculated trajectories compared to analytical approximations.\n\n")
            
            f.write("## Methods Analyzed\n\n")
            for method in self.integrators.keys():
                params = self.default_params[method]
                f.write(f"- **{method}**: Timestep = {params['timestep']}, ")
                f.write(f"Max Steps = {params['maxsteps']}\n")
            f.write("\n")
            
            f.write("## Impact Parameters Tested\n\n")
            f.write("The following impact parameters were tested:\n\n")
            for b in self.impact_parameters:
                if b == 5.2:
                    f.write(f"- **{b}** (near-critical value)\n")
                else:
                    f.write(f"- {b}\n")
            f.write("\n")
            
            f.write("## Summary of Results\n\n")
            f.write("Methods ranked by overall error (lower is better):\n\n")
            f.write("| Method | Angular Momentum Error (%) | Energy Error (%) | Periastron Error (%) | Overall Error |\n")
            f.write("|--------|---------------------------|------------------|----------------------|---------------|\n")
            
            for _, row in summary_df.iterrows():
                method = row["Method"]
                am_error = row["Avg_AM_Error_%"]
                e_error = row["Avg_Energy_Error_%"]
                p_error = row["Avg_Periastron_Error_%"]
                overall = row["Overall_Error"]
                
                f.write(f"| {method} | {am_error:.6f} | {e_error:.6f} | ")
                if not np.isnan(p_error):
                    f.write(f"{p_error:.6f} | ")
                else:
                    f.write("N/A | ")
                f.write(f"{overall:.6f} |\n")
            
            f.write("\n")
            
            f.write("## Key Findings\n\n")
            
            am_errors = summary_df["Avg_AM_Error_%"].values
            best_am_idx = np.nanargmin(am_errors)
            best_am_method = summary_df.iloc[best_am_idx]["Method"]
            
            e_errors = summary_df["Avg_Energy_Error_%"].values
            best_e_idx = np.nanargmin(e_errors)
            best_e_method = summary_df.iloc[best_e_idx]["Method"]
            
            overall_errors = summary_df["Overall_Error"].values
            best_overall_idx = np.nanargmin(overall_errors)
            best_overall_method = summary_df.iloc[best_overall_idx]["Method"]
            
            f.write(f"- Best method for angular momentum conservation: **{best_am_method}** ")
            f.write(f"(avg error: {am_errors[best_am_idx]:.6f}%)\n")
            f.write(f"- Best method for energy conservation: **{best_e_method}** ")
            f.write(f"(avg error: {e_errors[best_e_idx]:.6f}%)\n")
            f.write(f"- Best method overall: **{best_overall_method}** ")
            f.write(f"(overall error: {overall_errors[best_overall_idx]:.6f})\n\n")
            
            f.write("## Critical Impact Parameter Behavior\n\n")
            f.write("Near the critical impact parameter (b ≈ 5.2), the photon's behavior becomes highly ")
            f.write("sensitive to small changes in the trajectory. This region provides a stringent test ")
            f.write("of integrator accuracy. The plots in the `critical_orbit_b*.png` files show the ")
            f.write("orbits and winding behavior near this critical value.\n\n")
            
            f.write("## Recommendations\n\n")
            best_overall = best_overall_method
            
            # Fastest methods (approximation)
            fast_methods = []
            # for method in ["Euler", "Runge-Kutta 4", "Adams-Bashforth"]:
            for method in self.integrators.keys():
                if method in summary_df["Method"].values:
                    fast_methods.append(method)
            
            # Find accurate methods for critical orbits
            accurate_near_critical = []
            # for method in ["Obrechkoff", "Bowie", "Adams-Moulton4"]:
            for method in self.integrators.keys():
                if method in summary_df["Method"].values:
                    accurate_near_critical.append(method)
            
            f.write(f"- For general purpose simulations: **{best_overall}**\n")
            if fast_methods:
                f.write(f"- For faster simulations where some accuracy can be sacrificed: ")
                f.write(f"**{', '.join(fast_methods)}**\n")
            if accurate_near_critical:
                f.write(f"- For critical orbit simulations requiring high accuracy: ")
                f.write(f"**{', '.join(accurate_near_critical)}**\n")
            
            f.write("\n")
            
            f.write("## Conclusion\n\n")
            f.write("The choice of numerical integrator significantly impacts the accuracy of photon ")
            f.write("trajectory simulations around black holes. For visualization purposes, methods ")
            f.write("like Runge-Kutta 4 provide a good balance of speed and accuracy. For scientific ")
            f.write("calculations requiring high precision, especially near critical orbits, higher-order ")
            f.write("methods like Obrechkoff or specialized integrators like Bowie show superior ")
            f.write("conservation properties.\n\n")
            
            f.write("See the generated plots and CSV files for detailed comparisons between methods ")
            f.write("and impact parameters.")
        
        # HTML report for easier viewing
        with open(os.path.join(self.output_dir, "final_report.html"), 'w', encoding='utf-8') as f:
            f.write("<!DOCTYPE html>\n<html>\n<head>\n")
            f.write("<meta charset=\"UTF-8\">\n")
            f.write("<title>Black Hole Trajectory Analysis Report</title>\n")
            f.write("<style>\n")
            f.write("body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }\n")
            f.write("h1, h2 { color: #333; }\n")
            f.write("table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }\n")
            f.write("th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }\n")
            f.write("th { background-color: #f2f2f2; }\n")
            f.write("img { max-width: 100%; height: auto; margin: 20px 0; }\n")
            f.write("</style>\n</head>\n<body>\n")
            
            f.write("<h1>Black Hole Trajectory Analysis Report</h1>\n")
            
            f.write("<h2>Overview</h2>\n")
            f.write("<p>This report analyzes the performance and accuracy of different numerical ")
            f.write("integration methods for simulating photon trajectories around a ")
            f.write("Schwarzschild black hole. The analysis focuses on conservation of physical ")
            f.write("quantities (angular momentum and energy) and the accuracy of the ")
            f.write("calculated trajectories compared to analytical approximations.</p>\n")
            
            f.write("<h2>Methods Analyzed</h2>\n<ul>\n")
            for method in self.integrators.keys():
                params = self.default_params[method]
                f.write(f"<li><strong>{method}</strong>: Timestep = {params['timestep']}, ")
                f.write(f"Max Steps = {params['maxsteps']}</li>\n")
            f.write("</ul>\n")
            
            f.write("<h2>Impact Parameters Tested</h2>\n<ul>\n")
            for b in self.impact_parameters:
                if b == 5.2:
                    f.write(f"<li><strong>{b}</strong> (near-critical value)</li>\n")
                else:
                    f.write(f"<li>{b}</li>\n")
            f.write("</ul>\n")
            
            f.write("<h2>Summary of Results</h2>\n")
            f.write("<p>Methods ranked by overall error (lower is better):</p>\n")
            f.write("<table>\n<tr>\n<th>Method</th>\n<th>Angular Momentum Error (%)</th>\n")
            f.write("<th>Energy Error (%)</th>\n<th>Periastron Error (%)</th>\n<th>Overall Error</th>\n</tr>\n")
            
            for _, row in summary_df.iterrows():
                method = row["Method"]
                am_error = row["Avg_AM_Error_%"]
                e_error = row["Avg_Energy_Error_%"]
                p_error = row["Avg_Periastron_Error_%"]
                overall = row["Overall_Error"]
                
                f.write(f"<tr>\n<td>{method}</td>\n<td>{am_error:.6f}</td>\n<td>{e_error:.6f}</td>\n")
                if not np.isnan(p_error):
                    f.write(f"<td>{p_error:.6f}</td>\n")
                else:
                    f.write("<td>N/A</td>\n")
                f.write(f"<td>{overall:.6f}</td>\n</tr>\n")
            
            f.write("</table>\n")
            
            f.write("<h2>Key Findings</h2>\n<ul>\n")
            f.write(f"<li>Best method for angular momentum conservation: <strong>{best_am_method}</strong> ")
            f.write(f"(avg error: {am_errors[best_am_idx]:.6f}%)</li>\n")
            f.write(f"<li>Best method for energy conservation: <strong>{best_e_method}</strong> ")
            f.write(f"(avg error: {e_errors[best_e_idx]:.6f}%)</li>\n")
            f.write(f"<li>Best method overall: <strong>{best_overall_method}</strong> ")
            f.write(f"(overall error: {overall_errors[best_overall_idx]:.6f})</li>\n")
            f.write("</ul>\n")
            
            f.write("<h2>Results Visualization</h2>\n")
            
            f.write("<h3>Periastron vs Impact Parameter</h3>\n")
            f.write('<img src="periastron_vs_impact.png" alt="Periastron vs Impact Parameter">\n')
            
            f.write("<h3>Conservation Errors vs Impact Parameter</h3>\n")
            f.write('<img src="conservation_errors_vs_impact.png" alt="Conservation Errors">\n')
            
            f.write("<h3>Deflection Angle vs Impact Parameter</h3>\n")
            f.write('<img src="deflection_angle_vs_impact.png" alt="Deflection Angle">\n')
            
            f.write("<h3>Sample Individual Method Analysis (Standard Impact Parameter b=6.0)</h3>\n")
            for method in ["Runge-Kutta 4", "Obrechkoff"]:
                if method in self.integrators.keys():
                    file_name = f"{method.replace(' ', '_')}_b6.0.png"
                    if os.path.exists(os.path.join(self.output_dir, file_name)):
                        f.write(f'<img src="{file_name}" alt="{method} Analysis">\n')
            
            f.write("<h2>Recommendations</h2>\n<ul>\n")
            f.write(f"<li>For general purpose simulations: <strong>{best_overall}</strong></li>\n")
            if fast_methods:
                f.write(f"<li>For faster simulations where some accuracy can be sacrificed: ")
                f.write(f"<strong>{', '.join(fast_methods)}</strong></li>\n")
            if accurate_near_critical:
                f.write(f"<li>For critical orbit simulations requiring high accuracy: ")
                f.write(f"<strong>{', '.join(accurate_near_critical)}</strong></li>\n")
            f.write("</ul>\n")
            
            f.write("<h2>Conclusion</h2>\n")
            f.write("<p>The choice of numerical integrator significantly impacts the accuracy of photon ")
            f.write("trajectory simulations around black holes. For visualization purposes, methods ")
            f.write("like Runge-Kutta 4 provide a good balance of speed and accuracy. For scientific ")
            f.write("calculations requiring high precision, especially near critical orbits, higher-order ")
            f.write("methods like Obrechkoff or specialized integrators like Bowie show superior ")
            f.write("conservation properties.</p>\n")
            
            f.write("<p>See the generated plots and CSV files for detailed comparisons between methods ")
            f.write("and impact parameters.</p>\n")
            
            f.write("</body>\n</html>")


if __name__ == "__main__":
    analyzer = TrajectoryAnalyzer(output_dir="trajectory_analysis_0.005_5000")
    analyzer.run_full_analysis()