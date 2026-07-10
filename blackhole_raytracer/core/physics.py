"""
Physics calculations for black hole ray tracing.
"""
import math
import numpy as np
import numba
from numba import cuda
import operator

def make_trace_disk_redshift(integrator_step):
    """Factory function that creates a specialized trace function using the given integrator"""
    
    @cuda.jit(device=True)
    def trace_disk_redshift(y, f, y_temp, point, velocity, k1, k2, k3, k4, h2, h, 
                            max_iter, DISKINNERSQR, DISKOUTERSQR, DIST_CAM, SIN_I, 
                            sinalpha, idx, b, sin_psi, used_points):
        periastron = 100000
        m = math.sqrt(DISKINNERSQR)/2.0
        color = 0.0
        in_disc_r_x = 0
        in_disc_r_y = 0
        in_disc_r_z = 0
        is_in_disc = False
        was_in_disc = False
        crossed_disk = 0
        alpha = 0.0
        cos_alpha = 0.0

        for step in range(max_iter):
            used_points = step
            oldx = point[0]
            oldy = point[1]
            oldz = point[2]

            integrator_step(y, f, y_temp, point, velocity, k1, k2, k3, k4, h2, h)

            pointsqr = point[0]*point[0] + point[1]*point[1] + point[2]*point[2]
            crossed = operator.xor(oldy > 0.0, point[1] > 0.0)
            is_in_disc = operator.and_(pointsqr < DISKOUTERSQR, pointsqr > DISKINNERSQR)
            
            if pointsqr < periastron and pointsqr > DISKINNERSQR:
                periastron = pointsqr

            if pointsqr > 1.4*DIST_CAM:
                color = 0.0
                break

            if crossed and is_in_disc and crossed_disk < 1:
                t = -point[1]/velocity[1]
                in_disc_r_x = point[0] + t*velocity[0]
                in_disc_r_y = point[1] + t*velocity[1]
                in_disc_r_z = point[2] + t*velocity[2]
                v = math.sqrt(velocity[0]**2 + velocity[1]**2 + velocity[2]**2)
                cos_alpha = velocity[2] / v
                alpha_rad = math.acos(cos_alpha)
                sinalpha = math.sin(alpha_rad)
                was_in_disc = True
                crossed_disk += 1
            
            if pointsqr < 1.0 and (oldx*oldx + oldy*oldy + oldz*oldz) > 1.0:
                color = 0.0
                break

        # if was_in_disc:
        #     p = math.sqrt(periastron)
        #     bcalc = math.sqrt(p*p*p/(p-2.0))
            
        #     rid = math.sqrt(in_disc_r_x**2 + in_disc_r_y**2 + in_disc_r_z**2)
            
        #     phi = math.atan2(in_disc_r_x, in_disc_r_z)
        #     cos_psi = SIN_I*math.cos(phi)
        #     sinpsi = math.sin(math.acos(cos_psi))
            
        #     omega = math.pow(1/rid,3/2)
        #     singphi = math.sin(phi)/math.sqrt(1-math.cos(phi)**2*SIN_I**2)
        #     vs = omega*rid/math.sqrt(1-2/rid)
        #     lz = bcalc*SIN_I*singphi
            
        #     color = math.sqrt((1-vs**2)*(1-2/rid))/(1-omega*lz) - 1.0
            
        #     if math.isnan(color):
        #         color = 255.0
        if was_in_disc:
            rid = math.sqrt(in_disc_r_x**2 + in_disc_r_y**2 + in_disc_r_z**2) 

            if rid <= 3.0000001: # Avoid r <= 3M (ISCO)
                color = 0.0 
            else:
                omega = math.pow(rid, -1.5) # Keplerian omega (M=1)
                
                denominator = 1.0 - b * omega
                
                if abs(denominator) < 1e-9:
                    color = 0.0
                else:
                    numerator = math.sqrt(1.0 - 3.0 / rid)
                    g_factor = numerator / denominator

                    if math.isnan(g_factor) or g_factor < 0:
                        color = 0.0
                    elif g_factor < 1.0: # Redshift (g<1 -> z>0)
                        color = (1.0 / g_factor) - 1.0 
                    else: # Blueshift (g>1 -> z<0)
                        color = -(g_factor - 1.0) 

            if math.isnan(color):
                color = 0.0

        return color, used_points
    
    return trace_disk_redshift

def make_trace_disk_redshift_adaptive(integrator_step):
    """Factory function that creates a specialized trace function using the given integrator
    with adaptive step size support"""
    
    @cuda.jit(device=True)
    def trace_disk_redshift_adaptive(y, f, y_temp, point, velocity, k1, k2, k3, k4, h2, base_h, 
                            max_iter, DISKINNERSQR, DISKOUTERSQR, DIST_CAM, SIN_I, 
                            sinalpha, idx, b, sin_psi,
                            min_timestep, max_timestep, 
                            adapt_threshold):
        periastron = 100000
        m = math.sqrt(DISKINNERSQR)/2.0
        color = 0.0
        in_disc_r_x = 0
        in_disc_r_y = 0
        in_disc_r_z = 0
        is_in_disc = False
        was_in_disc = False
        crossed_disk = 0
        alpha = 0.0
        cos_alpha = 0.0
        schwarzschild_radius = 1.0

        current_h = base_h
        r_previous = math.sqrt(point[0]*point[0] + point[1]*point[1] + point[2]*point[2])

        for step in range(max_iter):
            oldx = point[0]
            oldy = point[1]
            oldz = point[2]

            r_current = r_previous
            
            proximity_factor = min(1.0, (r_current - schwarzschild_radius) / (5 * schwarzschild_radius))
            if proximity_factor < adapt_threshold:
                current_h = max(min_timestep, base_h * proximity_factor)
            else:
                current_h = min(max_timestep, current_h * (1.0 + adapt_threshold))
            
            integrator_step(y, f, y_temp, point, velocity, k1, k2, k3, k4, h2, current_h)
            
            pointsqr = point[0]*point[0] + point[1]*point[1] + point[2]*point[2]
            crossed = operator.xor(oldy > 0.0, point[1] > 0.0)
            is_in_disc = operator.and_(pointsqr < DISKOUTERSQR, pointsqr > DISKINNERSQR)
            
            if pointsqr < periastron and pointsqr > DISKINNERSQR:
                periastron = pointsqr

            if pointsqr > 1.4*DIST_CAM:
                color = 0.0
                break

            if crossed and is_in_disc and crossed_disk < 1:
                t = -point[1]/velocity[1]
                in_disc_r_x = point[0] + t*velocity[0]
                in_disc_r_y = point[1] + t*velocity[1]
                in_disc_r_z = point[2] + t*velocity[2]
                v = math.sqrt(velocity[0]**2 + velocity[1]**2 + velocity[2]**2)
                cos_alpha = velocity[2] / v
                alpha_rad = math.acos(cos_alpha)
                sinalpha = math.sin(alpha_rad)
                was_in_disc = True
                crossed_disk += 1
            
            if pointsqr < 1.0 and (oldx*oldx + oldy*oldy + oldz*oldz) > 1.0:
                color = 0.0
                break
                
            r_previous = math.sqrt(pointsqr)

        # if was_in_disc:
        #     p = math.sqrt(periastron)
        #     bcalc = math.sqrt(p*p*p/(p-2.0))
            
        #     rid = math.sqrt(in_disc_r_x**2 + in_disc_r_y**2 + in_disc_r_z**2)
            
        #     phi = math.atan2(in_disc_r_x, in_disc_r_z)
        #     cos_psi = SIN_I*math.cos(phi)
        #     sinpsi = math.sin(math.acos(cos_psi))
            
        #     omega = math.pow(1/rid,3/2)
        #     singphi = math.sin(phi)/math.sqrt(1-math.cos(phi)**2*SIN_I**2)
        #     vs = omega*rid/math.sqrt(1-2/rid)
        #     lz = bcalc*SIN_I*singphi
            
        #     color = math.sqrt((1-vs**2)*(1-2/rid))/(1-omega*lz) - 1.0
            
        #     if math.isnan(color):
        #         color = 255.0
        if was_in_disc:
            rid = math.sqrt(in_disc_r_x**2 + in_disc_r_y**2 + in_disc_r_z**2) 

            if rid <= 3.0000001: # Avoid r <= 3M (ISCO)
                color = 0.0 
            else:
                omega = math.pow(rid, -1.5) # Keplerian omega (M=1)
                
                denominator = 1.0 - b * omega
                
                if abs(denominator) < 1e-9:
                    color = 0.0
                else:
                    numerator = math.sqrt(1.0 - 3.0 / rid)
                    g_factor = numerator / denominator

                    if math.isnan(g_factor) or g_factor < 0:
                        color = 0.0
                    elif g_factor < 1.0: # Redshift (g<1 -> z>0)
                        color = (1.0 / g_factor) - 1.0 
                    else: # Blueshift (g>1 -> z<0)
                        color = -(g_factor - 1.0) 

            if math.isnan(color):
                color = 0.0

        return color
    
    return trace_disk_redshift_adaptive

def make_trace_plane_intersection(integrator_step):
    """Factory function that creates a specialized plane intersection function"""
    
    @cuda.jit(device=True)
    def trace_plane_intersection(y, f, y_temp, point, velocity, k1, k2, k3, k4, h2, h,
                                max_iter, inter_point, checked):
        distance_to_plane = -10.0
        for step in range(max_iter):
            oldx = point[0]
            oldy = point[1]
            oldz = point[2]

            integrator_step(y, f, y_temp, point, velocity, k1, k2, k3, k4, h2, h)

            pointsqr = point[0]*point[0] + point[1]*point[1] + point[2]*point[2]
            crossed = operator.xor(oldz > distance_to_plane, point[2] > distance_to_plane) 

            if crossed:
                if checked:
                    t = -point[2]/velocity[2]
                    inter_point[0] = point[0] + t*velocity[0]
                    inter_point[1] = point[1] + t*velocity[1]
                    inter_point[2] = distance_to_plane
                else:
                    inter_point[0] = point[0] + velocity[0]
                    inter_point[1] = point[1] + velocity[1]
                    inter_point[2] = point[2] + velocity[2]
                break
            
            if pointsqr < 1.0 and (oldx*oldx + oldy*oldy + oldz*oldz) > 1.0:
                inter_point[0] = 0.0
                inter_point[1] = 0.0
                inter_point[2] = 0.0
                break
        
        return
    
    return trace_plane_intersection

def make_trace_plane_intersection_adaptive(integrator_step):
    """Factory function that creates a specialized plane intersection function with adaptive step size"""
    
    @cuda.jit(device=True)
    def trace_plane_intersection_adaptive(y, f, y_temp, point, velocity, k1, k2, k3, k4, h2, base_h,
                                max_iter, inter_point, checked,
                                min_h_factor, max_h_factor, 
                                adapt_threshold):
        distance_to_plane = -10.0
        schwarzschild_radius=1.0
        
        min_timestep = base_h * min_h_factor
        max_timestep = base_h * max_h_factor
        current_h = base_h
        r_previous = math.sqrt(point[0]*point[0] + point[1]*point[1] + point[2]*point[2])
        
        for step in range(max_iter):
            oldx = point[0]
            oldy = point[1]
            oldz = point[2]

            r_current = r_previous
            
            proximity_factor = min(1.0, (r_current - schwarzschild_radius) / (5 * schwarzschild_radius))
            if proximity_factor < adapt_threshold:
                current_h = max(min_timestep, base_h * proximity_factor)
            else:
                current_h = min(max_timestep, current_h * (1.0 + adapt_threshold))

            integrator_step(y, f, y_temp, point, velocity, k1, k2, k3, k4, h2, current_h)

            pointsqr = point[0]*point[0] + point[1]*point[1] + point[2]*point[2]
            crossed = operator.xor(oldz > distance_to_plane, point[2] > distance_to_plane) 

            if crossed:
                if checked:
                    t = -point[2]/velocity[2]
                    inter_point[0] = point[0] + t*velocity[0]
                    inter_point[1] = point[1] + t*velocity[1]
                    inter_point[2] = distance_to_plane
                else:
                    inter_point[0] = point[0] + velocity[0]
                    inter_point[1] = point[1] + velocity[1]
                    inter_point[2] = point[2] + velocity[2]
                break
            
            if pointsqr < 1.0 and (oldx*oldx + oldy*oldy + oldz*oldz) > 1.0:
                inter_point[0] = 0.0
                inter_point[1] = 0.0
                inter_point[2] = 0.0
                break

            r_previous = math.sqrt(pointsqr)
        
        return
    
    return trace_plane_intersection_adaptive

def setup_camera(camera_position, fov, width, height):
    """
    Set up camera and view matrices.
    
    Args:
        camera_position (list): 3D position of the camera [x, y, z]
        fov (float): Field of view in degrees
        width (int): Image width in pixels
        height (int): Image height in pixels
        
    Returns:
        dict: Camera parameters including view matrices and derived values
    """
    camera_pos = np.array(camera_position) 
    DIST_CAM = np.sum(camera_pos**2)
    camera_target = np.array([0, 0, 0]) 
    forward = camera_target - camera_pos
    
    direction_xy = np.array([forward[0], forward[1], 0])
    dot_product = np.dot(forward, direction_xy)
    norms = np.linalg.norm(forward) * np.linalg.norm(direction_xy)
    cos_angle = dot_product / norms if norms > 0 else 0
    inclination = np.arccos(cos_angle)
    SIN_I = math.sin(inclination)
    sin_psi = math.sin(math.pi/2-inclination)
    
    global_up = np.array([0, -1, 0])
    right = np.cross(forward, global_up)
    right = right / np.linalg.norm(right)
    camera_up = np.cross(right, forward)
    camera_up = camera_up / np.linalg.norm(camera_up)
    forward = forward / np.linalg.norm(forward)

    view_matrix = np.array([
        [right[0], camera_up[0], -forward[0], 0],
        [right[1], camera_up[1], -forward[1], 0],
        [right[2], camera_up[2], -forward[2], 0],
        [-np.dot(right, camera_pos), -np.dot(camera_up, camera_pos), np.dot(forward, camera_pos), 1]
    ], dtype=np.float32)
    inv_view_matrix = np.linalg.inv(view_matrix)
    
    aspect_ratio = width/height
    image_plane_width = 2 * np.tan(np.radians(fov) / 2)
    image_plane_height = image_plane_width / aspect_ratio
    
    return {
        'view_matrix': view_matrix,
        'inv_view_matrix': inv_view_matrix,
        'image_plane_width': image_plane_width,
        'image_plane_height': image_plane_height,
        'DIST_CAM': DIST_CAM,
        'SIN_I': SIN_I,
        'sin_psi': sin_psi
    }

def initialize_cuda_arrays(width, height):
    """
    Initialize CUDA arrays for ray tracing.
    
    Args:
        width (int): Image width in pixels
        height (int): Image height in pixels
        
    Returns:
        dict: Dictionary of CUDA arrays
    """
    pixels = width * height
    
    y_t = np.zeros([pixels, 6], dtype="float32")
    y_t_cuda = cuda.to_device(y_t)

    f_t = np.zeros([pixels, 6], dtype="float32")
    f_t_cuda = cuda.to_device(f_t)

    y_temp = np.zeros([pixels, 6], dtype="float32")
    y_temp_cuda = cuda.to_device(y_temp)

    k1 = np.zeros([pixels, 6], dtype="float32")
    k1_cuda = cuda.to_device(k1)
    k2 = np.zeros([pixels, 6], dtype="float32")
    k2_cuda = cuda.to_device(k2)
    k3 = np.zeros([pixels, 6], dtype="float32")
    k3_cuda = cuda.to_device(k3)
    k4 = np.zeros([pixels, 6], dtype="float32")
    k4_cuda = cuda.to_device(k4)

    points = np.zeros([pixels, 3], dtype="float32")
    pos_cuda = cuda.to_device(points)

    velocities = np.zeros([pixels, 3], dtype="float32")
    vel_cuda = cuda.to_device(velocities)

    ray_dir_screen = np.zeros([pixels, 4], dtype="float32")
    ray_dir_screen_cuda = cuda.to_device(ray_dir_screen)
    ray_dir_world = np.zeros([pixels, 4], dtype="float32")
    ray_dir_world_cuda = cuda.to_device(ray_dir_world)
    
    return {
        'y_t_cuda': y_t_cuda,
        'f_t_cuda': f_t_cuda,
        'y_temp_cuda': y_temp_cuda,
        'k1_cuda': k1_cuda,
        'k2_cuda': k2_cuda,
        'k3_cuda': k3_cuda,
        'k4_cuda': k4_cuda,
        'pos_cuda': pos_cuda,
        'vel_cuda': vel_cuda,
        'ray_dir_screen_cuda': ray_dir_screen_cuda,
        'ray_dir_world_cuda': ray_dir_world_cuda
    }