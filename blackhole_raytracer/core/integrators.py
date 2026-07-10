"""
Integration methods for ray tracing simulation.
This module provides different numerical integrators for ray tracing.
"""
import math
import numba
from numba import cuda
import numpy as np

@numba.jit(nopython=True)
def pvcalc(f, y, h2):
    """Binet"""
    f[0] = y[3]
    f[1] = y[4]
    f[2] = y[5]
    q = 1 / math.pow(y[0]*y[0] + y[1]*y[1] + y[2]*y[2], 2.5)
    f[3] = -1.5 * h2 * y[0] * q
    f[4] = -1.5 * h2 * y[1] * q
    f[5] = -1.5 * h2 * y[2] * q
    return f

@cuda.jit(device=True)
def rk4_step(y, f, y_temp, point, velocity, k1, k2, k3, k4, h2, h):
    """Single step of Runge-Kutta 4 integration"""
    oldx = point[0]
    oldy = point[1]
    oldz = point[2]

    y[0] = point[0]
    y[1] = point[1]
    y[2] = point[2]
    y[3] = velocity[0]
    y[4] = velocity[1]
    y[5] = velocity[2]
    
    k1 = pvcalc(f, y, h2)
    for i in range(6):
        y_temp[i] = y[i] + 0.5 * h * k1[i]
    k2 = pvcalc(f, y_temp, h2)
    for i in range(6):
        y_temp[i] = y[i] + 0.5 * h * k2[i]
    k3 = pvcalc(f, y_temp, h2)
    for i in range(6):
        y_temp[i] = y[i] + h * k3[i]
    k4 = pvcalc(f, y_temp, h2)

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

@cuda.jit(device=True)
def euler_step(y, f, y_temp, point, velocity, k1, k2, k3, k4, h2, h):
    """Single step of Euler integration"""
    oldx = point[0]
    oldy = point[1]
    oldz = point[2]

    y[0] = point[0]
    y[1] = point[1]
    y[2] = point[2]
    y[3] = velocity[0]
    y[4] = velocity[1]
    y[5] = velocity[2]
    
    k1 = pvcalc(f, y, h2)
    
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

@cuda.jit(device=True)
def adams_bashforth_step(y, f, y_temp, point, velocity, k1, k2, k3, k4, h2, h):
    """Single step of Adams-Bashforth integration"""
    oldx = point[0]
    oldy = point[1]
    oldz = point[2]

    y[0] = point[0]
    y[1] = point[1]
    y[2] = point[2]
    y[3] = velocity[0]
    y[4] = velocity[1]
    y[5] = velocity[2]
    
    current_f = pvcalc(f, y, h2)
    
    if k1[0] == 0 and k1[1] == 0 and k1[2] == 0:
        velocity[0] += h * current_f[3]
        velocity[1] += h * current_f[4]
        velocity[2] += h * current_f[5]
        
        point[0] += h * current_f[0]
        point[1] += h * current_f[1]
        point[2] += h * current_f[2]
    else:
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

@cuda.jit(device=True)
def adams_bashforth4_step(y, f, y_temp, point, velocity, k1, k2, k3, k4, h2, h):
    """Single step of 4th-order Adams-Bashforth integration"""
    oldx = point[0]
    oldy = point[1]
    oldz = point[2]

    y[0] = point[0]
    y[1] = point[1]
    y[2] = point[2]
    y[3] = velocity[0]
    y[4] = velocity[1]
    y[5] = velocity[2]
    
    current_f = pvcalc(f, y, h2)
    
    if k3[0] == 0 and k3[1] == 0 and k3[2] == 0:
        if k2[0] == 0 and k2[1] == 0 and k2[2] == 0:
            if k1[0] == 0 and k1[1] == 0 and k1[2] == 0:
                velocity[0] += h * current_f[3]
                velocity[1] += h * current_f[4]
                velocity[2] += h * current_f[5]
                
                point[0] += h * current_f[0]
                point[1] += h * current_f[1]
                point[2] += h * current_f[2]
            else:
                velocity[0] += h * (1.5 * current_f[3] - 0.5 * k1[3])
                velocity[1] += h * (1.5 * current_f[4] - 0.5 * k1[4])
                velocity[2] += h * (1.5 * current_f[5] - 0.5 * k1[5])
                
                point[0] += h * (1.5 * current_f[0] - 0.5 * k1[0])
                point[1] += h * (1.5 * current_f[1] - 0.5 * k1[1])
                point[2] += h * (1.5 * current_f[2] - 0.5 * k1[2])
        else:
            velocity[0] += h * ((23./12.) * current_f[3] - (16./12.) * k1[3] + (5./12.) * k2[3])
            velocity[1] += h * ((23./12.) * current_f[4] - (16./12.) * k1[4] + (5./12.) * k2[4])
            velocity[2] += h * ((23./12.) * current_f[5] - (16./12.) * k1[5] + (5./12.) * k2[5])
            
            point[0] += h * ((23./12.) * current_f[0] - (16./12.) * k1[0] + (5./12.) * k2[0])
            point[1] += h * ((23./12.) * current_f[1] - (16./12.) * k1[1] + (5./12.) * k2[1])
            point[2] += h * ((23./12.) * current_f[2] - (16./12.) * k1[2] + (5./12.) * k2[2])
    else:
        velocity[0] += h * ((55./24.) * current_f[3] - (59./24.) * k1[3] + (37./24.) * k2[3] - (9./24.) * k3[3])
        velocity[1] += h * ((55./24.) * current_f[4] - (59./24.) * k1[4] + (37./24.) * k2[4] - (9./24.) * k3[4])
        velocity[2] += h * ((55./24.) * current_f[5] - (59./24.) * k1[5] + (37./24.) * k2[5] - (9./24.) * k3[5])
        
        point[0] += h * ((55./24.) * current_f[0] - (59./24.) * k1[0] + (37./24.) * k2[0] - (9./24.) * k3[0])
        point[1] += h * ((55./24.) * current_f[1] - (59./24.) * k1[1] + (37./24.) * k2[1] - (9./24.) * k3[1])
        point[2] += h * ((55./24.) * current_f[2] - (59./24.) * k1[2] + (37./24.) * k2[2] - (9./24.) * k3[2])
    
    for i in range(6):
        k3[i] = k2[i]
        k2[i] = k1[i]
        k1[i] = current_f[i]
        
    y_temp[0] = oldx
    y_temp[1] = oldy
    y_temp[2] = oldz
    
    return

@cuda.jit(device=True)
def adams_moulton4_step(y, f, y_temp, point, velocity, k1, k2, k3, k4, h2, h):
    """Single step of 4th-order Adams-Moulton integration
    
    Parameters:
    y, f, y_temp: Arrays for state vector, derivatives, and temporary storage
    point: Current position (x, y, z)
    velocity: Current velocity vector
    k1, k2, k3, k4: Arrays to store function evaluations from previous steps
        (k1 = f_n, k2 = f_{n-1}, k3 = f_{n-2})
    h2: Squared angular momentum parameter for pvcalc
    h: Step size
    
    Note: Adams-Moulton is an implicit method requiring iteration to solve for y_{n+1}
    """
    max_iter = 10
    tol = 1e-6

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
    
    current_f = pvcalc(f, y, h2)
    
    if k2[0] == 0 and k2[1] == 0 and k2[2] == 0:
        # Use rk4 for the first steps

        for i in range(6):
            k4[i] = current_f[i]
            y_temp[i] = y[i] + 0.5 * h * k4[i]
        
        pvcalc(f, y_temp, h2)
        for i in range(6):
            k3[i] = f[i]
            y_temp[i] = y[i] + 0.5 * h * k3[i]
        
        pvcalc(f, y_temp, h2)
        for i in range(6):
            k2[i] = f[i]
            y_temp[i] = y[i] + h * k2[i]
        
        pvcalc(f, y_temp, h2)
        for i in range(6):
            k1[i] = f[i]
            
        # Update with rk4
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
        
        current_f = pvcalc(f, y, h2)
    else:
        # Initial guess, use explicit Euler method
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
            
            f_np1 = pvcalc(f, y, h2)
            
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
        
        y[0] = point[0]
        y[1] = point[1]
        y[2] = point[2]
        y[3] = velocity[0]
        y[4] = velocity[1]
        y[5] = velocity[2]
        
        f_final = pvcalc(f, y, h2)
        for i in range(6):
            f_np1[i] = f_final[i]
    
    # Shift history: k2 <- k1, k1 <- current_f, current_f <- f_np1
    for i in range(6):
        k3[i] = k2[i]
        k2[i] = k1[i]
        k1[i] = current_f[i]
    
    y_temp[0] = oldx
    y_temp[1] = oldy
    y_temp[2] = oldz
    
    return

@cuda.jit(device=True)
def bowie_single_step(y, y_prime, h, k):
    """
    Bowie single step numerical integration method
    
    4th-order method for second-order ODEs of the form:
    d^2y / dphi^2 = f(y), where f(y) = 3ky^2 - y
    
    Parameters:
    y: Current position (u = 1/r)
    y_prime: Current derivative (dy/dphi)
    h: Step size (delta phi)
    k: Constant (GM/c^2)
    
    Returns:
    y_next, y_prime_next: Updated position and derivative
    """
    f_y = 3.0 * k * y * y - y
    df_dy = 6.0 * k * y - 1.0
    d2f_dy2 = 6.0 * k

    y_prime_squared = y_prime * y_prime
    
    y_next = y + h * y_prime + \
             (h * h / 2.0) * f_y + \
             (h * h * h / 6.0) * df_dy * y_prime + \
             (h * h * h * h / 24.0) * (d2f_dy2 * y_prime_squared + df_dy * f_y)
    
    y_prime_next = y_prime + h * f_y + \
                  (h * h / 2.0) * df_dy * y_prime + \
                  (h * h * h / 6.0) * (d2f_dy2 * y_prime_squared + df_dy * f_y)
    
    return y_next, y_prime_next

@cuda.jit(device=True)
def bowie_step_3d(point, velocity, h, k, r_s):
    """
    Perform Bowie single step integration for a 3D point in space
    
    This function transforms the 3D problem to the Binet equation form, 
    applies the Bowie single step method and transforms back.
    
    Parameters:
    point: 3D position vector (x, y, z)
    velocity: 3D velocity vector
    h: Step size (for azimuthal angle phi)
    k: Constant (GM/c^2)
    r_s: Schwarzschild radius (2GM/c^2)
    """
    x, y, z = point[0], point[1], point[2]
    vx, vy, vz = velocity[0], velocity[1], velocity[2]

    r = math.sqrt(x*x + y*y + z*z)
    
    # L = r x v
    Lx = y*vz - z*vy
    Ly = z*vx - x*vz
    Lz = x*vy - y*vx

    L_squared = Lx*Lx + Ly*Ly + Lz*Lz
    L_mag = math.sqrt(max(L_squared, 1e-10))
    
    inv_r = 1.0 / r
    er_x, er_y, er_z = x * inv_r, y * inv_r, z * inv_r
    
    v_r = vx * er_x + vy * er_y + vz * er_z
    
    u = inv_r
    
    # y'_0 = -v_r/(r^2_0 * v_phi)
    v_phi = L_mag / r
    
    epsilon = 1e-10
    if abs(v_phi) < epsilon:
        u_prime = 0.0
    else:
        u_prime = -v_r / (r * v_phi)
    
    u_next, u_prime_next = bowie_single_step(u, u_prime, h, k)
    r_new = 1.0 / u_next
    
    # v_r_new = -u_prime_next * r_new * v_phi_new
    v_phi_new = L_mag / r_new  
    v_r_new = -u_prime_next * r_new * v_phi_new
    
    delta_phi = h
    
    if L_mag < 1e-10:
        L_hat_x = 0.0
        L_hat_y = 0.0
        L_hat_z = 1.0
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
    
    # Rodrigues'
    er_new_x = er_x * cos_phi + cross_x * sin_phi + L_hat_x * dot * one_minus_cos
    er_new_y = er_y * cos_phi + cross_y * sin_phi + L_hat_y * dot * one_minus_cos
    er_new_z = er_z * cos_phi + cross_z * sin_phi + L_hat_z * dot * one_minus_cos
    
    x_new = r_new * er_new_x
    y_new = r_new * er_new_y
    z_new = r_new * er_new_z
    
    et_x = L_hat_y * er_z - L_hat_z * er_y
    et_y = L_hat_z * er_x - L_hat_x * er_z
    et_z = L_hat_x * er_y - L_hat_y * er_x
    
    et_mag = math.sqrt(et_x*et_x + et_y*et_y + et_z*et_z)
    if et_mag > 1e-10:
        inv_et = 1.0 / et_mag
        et_x *= inv_et
        et_y *= inv_et
        et_z *= inv_et
        
        v_sq = vx*vx + vy*vy + vz*vz
        v_t = math.sqrt(max(0.0, v_sq - v_r*v_r))
        
        v_t_new = L_mag / r_new
        
        cross_t_x = et_y * L_hat_z - et_z * L_hat_y
        cross_t_y = et_z * L_hat_x - et_x * L_hat_z
        cross_t_z = et_x * L_hat_y - et_y * L_hat_x
        
        dot_t = et_x * L_hat_x + et_y * L_hat_y + et_z * L_hat_z
        
        et_new_x = et_x * cos_phi + cross_t_x * sin_phi + L_hat_x * dot_t * one_minus_cos
        et_new_y = et_y * cos_phi + cross_t_y * sin_phi + L_hat_y * dot_t * one_minus_cos
        et_new_z = et_z * cos_phi + cross_t_z * sin_phi + L_hat_z * dot_t * one_minus_cos
        
        vx_new = v_r_new * er_new_x + v_t_new * et_new_x
        vy_new = v_r_new * er_new_y + v_t_new * et_new_y
        vz_new = v_r_new * er_new_z + v_t_new * et_new_z
    else:
        vx_new = v_r_new * er_new_x
        vy_new = v_r_new * er_new_y
        vz_new = v_r_new * er_new_z
    
    point[0], point[1], point[2] = x_new, y_new, z_new
    velocity[0], velocity[1], velocity[2] = vx_new, vy_new, vz_new
    
    return

@cuda.jit(device=True)
def bowie_integrator(y, f, y_temp, point, velocity, k1, k2, k3, k4, h2, h):
    """ 
    Adapter function to match the signature of other integrators 
    
    Parameters:
    y: Not used directly, but required for consistency with other integrators
    f: Not used directly, but required for consistency with other integrators
    y_temp: Temporary storage for results
    point: 3D position vector to be updated
    velocity: 3D velocity vector to be updated
    k1, k2, k3, k4: Not used directly, can be used to pass constants
    h2: Not used directly, but required for consistency with other integrators
    h: Step size for integration
    """

    y_temp[0] = point[0]
    y_temp[1] = point[1]
    y_temp[2] = point[2]
    
    k = 0.5  # Default value for GM/c^2
    r_s = 2.0 * k  # Schwarzschild radius (2GM/c^2)
    
    bowie_step_3d(point, velocity, h, k, r_s)
    
    return

@cuda.jit(device=True)
def obrechkoff_single_step(y, y_prime, h, k):
    """
    Optimized Fourth-order Obrechkoff single step method
    
    Parameters:
    y: Current position (u = 1/r)
    y_prime: Current derivative (du/dphi)
    h: Step size (delta phi)
    k: Constant (GM/c^2)
    
    Returns:
    y_next, y_prime_next: Updated position and derivative
    """
    max_iter = 12
    tol = 1e-8
    
    f_y = 3.0 * k * y * y - y
    
    y_next = y + h * y_prime
    y_prime_next = y_prime + h * f_y

    h_half = h * 0.5
    h_squared_12 = h * h / 12.0
    three_k = 3.0 * k
    six_k = 6.0 * k
    
    for _ in range(max_iter):
        f_y_next = three_k * y_next * y_next - y_next
        
        F1 = y_next - y - h_half * (y_prime + y_prime_next) - h_squared_12 * (-y + y_next + three_k * (y*y - y_next*y_next))
        F2 = y_prime_next - y_prime - h_half * (-y - y_next + three_k * (y*y + y_next*y_next)) - h_squared_12 * (y_prime_next - y_prime + six_k * (y*y_prime - y_next*y_prime_next))

        # df_dy_next = six_k * y_next - 1.0
        # J11 = 1.0 - h_squared_12 * (1.0 - df_dy_next)
        J11 = 1.0 - h_squared_12 * (1.0 - six_k * y_next)
        J12 = -h_half
        # J21 = -h_half * (-1.0 + df_dy_next) - h_squared_12 * (-six_k * y_prime_next)
        J21 = h_half * (1.0 - six_k * y_next) + h_squared_12 * (six_k * y_prime_next)
        J22 = J11

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

@cuda.jit(device=True)
def obrechkoff_step_3d(point, velocity, h, k, r_s):
    """
    Optimized Obrechkoff step integration for a 3D point
    
    Parameters:
    point: 3D position vector (x, y, z)
    velocity: 3D velocity vector
    h: Step size (for azimuthal angle phi)
    k: Constant (GM/c^2)
    r_s: Schwarzschild radius (2GM/c^2)
    """
    x, y, z = point[0], point[1], point[2]
    vx, vy, vz = velocity[0], velocity[1], velocity[2]

    r_squared = x*x + y*y + z*z
    r = math.sqrt(r_squared)
    
    if r < 1e-6:
        return
    
    # L = r x v
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
    
    u_next, u_prime_next = obrechkoff_single_step(u, u_prime, h, k)
    r_new = 1.0 / u_next
    
    # L_mag conserved
    v_phi_new = L_mag / r_new
    v_r_new = -u_prime_next * r_new * v_phi_new
    
    delta_phi = h
    
    if L_mag < 1e-8:
        L_hat_x = 0.0
        L_hat_y = 0.0
        L_hat_z = 1.0
    else:
        inv_L = 1.0 / L_mag
        L_hat_x = Lx * inv_L
        L_hat_y = Ly * inv_L
        L_hat_z = Lz * inv_L
    
    cos_phi = math.cos(delta_phi)
    sin_phi = math.sin(delta_phi)
    
    # er x L_hat
    cross_x = er_y * L_hat_z - er_z * L_hat_y
    cross_y = er_z * L_hat_x - er_x * L_hat_z
    cross_z = er_x * L_hat_y - er_y * L_hat_x
    
    # L_hat dot er
    dot = er_x * L_hat_x + er_y * L_hat_y + er_z * L_hat_z
    one_minus_cos = 1.0 - cos_phi
    
    # Rodrigues'
    er_new_x = er_x * cos_phi + cross_x * sin_phi + L_hat_x * dot * one_minus_cos
    er_new_y = er_y * cos_phi + cross_y * sin_phi + L_hat_y * dot * one_minus_cos
    er_new_z = er_z * cos_phi + cross_z * sin_phi + L_hat_z * dot * one_minus_cos
    
    point[0] = r_new * er_new_x
    point[1] = r_new * er_new_y
    point[2] = r_new * er_new_z
    
    # et = L_hat x er
    et_x = L_hat_y * er_z - L_hat_z * er_y
    et_y = L_hat_z * er_x - L_hat_x * er_z
    et_z = L_hat_x * er_y - L_hat_y * er_x
    
    et_sq = et_x*et_x + et_y*et_y + et_z*et_z
    if et_sq > 1e-10:
        inv_et = 1.0 / math.sqrt(et_sq)
        et_x *= inv_et
        et_y *= inv_et
        et_z *= inv_et
        
        v_t_new = L_mag / r_new
        
        cross_t_x = et_y * L_hat_z - et_z * L_hat_y
        cross_t_y = et_z * L_hat_x - et_x * L_hat_z
        cross_t_z = et_x * L_hat_y - et_y * L_hat_x
        
        dot_t = et_x * L_hat_x + et_y * L_hat_y + et_z * L_hat_z
        
        et_new_x = et_x * cos_phi + cross_t_x * sin_phi + L_hat_x * dot_t * one_minus_cos
        et_new_y = et_y * cos_phi + cross_t_y * sin_phi + L_hat_y * dot_t * one_minus_cos
        et_new_z = et_z * cos_phi + cross_t_z * sin_phi + L_hat_z * dot_t * one_minus_cos

        velocity[0] = v_r_new * er_new_x + v_t_new * et_new_x
        velocity[1] = v_r_new * er_new_y + v_t_new * et_new_y
        velocity[2] = v_r_new * er_new_z + v_t_new * et_new_z
    else:
        velocity[0] = v_r_new * er_new_x
        velocity[1] = v_r_new * er_new_y
        velocity[2] = v_r_new * er_new_z
    
    return

@cuda.jit(device=True)
def obrechkoff_integrator(y, f, y_temp, point, velocity, k1, k2, k3, k4, h2, h):
    """ 
    Adapter function to match the signature of other integrators 
    
    Parameters:
    y: Not used directly, but required for consistency with other integrators
    f: Not used directly, but required for consistency with other integrators
    y_temp: Temporary storage for results
    point: 3D position vector to be updated
    velocity: 3D velocity vector to be updated
    k1, k2, k3, k4: Not used directly, can be used to pass constants
    h2: Not used directly, but required for consistency with other integrators
    h: Step size for integration
    """
    y_temp[0] = point[0]
    y_temp[1] = point[1]
    y_temp[2] = point[2]
    
    k = 0.5  # Default value for GM/c^2
    r_s = 2.0 * k  # Schwarzschild radius
    
    obrechkoff_step_3d(point, velocity, h, k, r_s)
    
    return

INTEGRATORS = {
    "Runge-Kutta 4": rk4_step,
    "Euler": euler_step,
    "Adams-Bashforth": adams_bashforth_step,
    "Adams-Bashforth4": adams_bashforth4_step,
    "Adams-Moulton4": adams_moulton4_step,
    "Bowie": bowie_integrator,
    "Obrechkoff": obrechkoff_integrator,
}

def get_available_methods():
    return list(INTEGRATORS.keys())