"""
Accretion disk renderer for black hole visualization.
"""
import math
import numpy as np
from numba import cuda
from timeit import default_timer as timer

from ..core.physics import setup_camera, initialize_cuda_arrays
from ..core.cuda_kernels import DISK_IMAGE_KERNELS, PATH_KERNELS
from ..core.cuda_kernels_adaptive import DISK_IMAGE_KERNELS_ADAPTIVE, PATH_KERNELS_ADAPTIVE

def render_disk_adaptive(width, height, inner_radius, outer_radius, camera_position, 
               timestep, maxsteps, fov, integrator_name="Runge-Kutta 4", 
               show_paths=False, samplerate=50,
               min_h_factor=0.1, max_h_factor=2.0, adapt_threshold=0.01):
    """
    Render black hole with accretion disk showing redshift effects, using adaptive step size.
    
    Args:
        width (int): Image width in pixels
        height (int): Image height in pixels
        inner_radius (float): Inner radius of the accretion disk
        outer_radius (float): Outer radius of the accretion disk
        camera_position (list): 3D position of the camera [x, y, z]
        timestep (float): Base time step for integration (will be adjusted if adaptive is enabled)
        maxsteps (int): Maximum number of integration steps
        fov (float): Field of view in degrees
        integrator_name (str): Name of the integration method to use
        show_paths (bool): Whether to show ray path trajectories
        samplerate (int): Sample rate for paths
        min_h_factor (float): Minimum step size factor (relative to timestep)
        max_h_factor (float): Maximum step size factor (relative to timestep)
        adapt_threshold (float): Threshold for adaptation
    
    Returns:
        tuple: (image, history) - the rendered image and ray path history
    """
    width = int(width)
    height = int(height)
    print(f"Rendering with dimensions: {width}x{height}, adaptive stepping")
    image = np.zeros((height, width), dtype=np.float32)
    image_cuda = cuda.to_device(image)
    
    if show_paths:
        history = np.zeros([int(math.ceil(width/samplerate) * math.ceil(height/samplerate)), int(maxsteps), 3], dtype="float32")
        history_cuda = cuda.to_device(history)

    DISKINNERSQR = float(inner_radius**2)
    DISKOUTERSQR = float(outer_radius**2)
    
    camera_params = setup_camera(camera_position, fov, width, height)
    cuda_arrays = initialize_cuda_arrays(width, height)

    if integrator_name not in DISK_IMAGE_KERNELS_ADAPTIVE:
        print(f"Warning: Integration method {integrator_name} not available for adaptive stepping. Using Runge-Kutta 4 instead.")
        integrator_name = "Runge-Kutta 4"

    disk_kernel = DISK_IMAGE_KERNELS_ADAPTIVE[integrator_name]
    path_kernels = PATH_KERNELS_ADAPTIVE[integrator_name]

    if integrator_name in ["Adams-Moulton4", "Adams-Bashforth4", "Bowie", "Obrechkoff"]:
        nthreads = 16
    else:
        nthreads = 32
    nblocksy = (width//nthreads) + 1
    nblocksx = (height//nthreads) + 1
    
    s = timer()
    
    disk_kernel[(nblocksx, nblocksy), (nthreads, nthreads)](
        image_cuda, 
        cuda_arrays['y_t_cuda'], 
        cuda_arrays['f_t_cuda'], 
        cuda_arrays['y_temp_cuda'], 
        cuda_arrays['k1_cuda'], 
        cuda_arrays['k2_cuda'], 
        cuda_arrays['k3_cuda'], 
        cuda_arrays['k4_cuda'], 
        timestep, 
        maxsteps, 
        cuda_arrays['pos_cuda'], 
        cuda_arrays['vel_cuda'], 
        camera_params['inv_view_matrix'], 
        DISKINNERSQR, 
        DISKOUTERSQR, 
        camera_params['DIST_CAM'], 
        camera_params['SIN_I'], 
        cuda_arrays['ray_dir_screen_cuda'], 
        cuda_arrays['ray_dir_world_cuda'], 
        camera_params['image_plane_width'], 
        camera_params['image_plane_height'], 
        camera_position[0],
        camera_position[1],
        camera_position[2], 
        camera_params['sin_psi'],
        min_h_factor,
        max_h_factor,
        adapt_threshold
    )
    
    image = image_cuda.copy_to_host()

    history_modified = None
    
    if show_paths:
        block_dim = (16, 16)
        grid_dim = ((width + block_dim[0] - 1) // block_dim[0], 
                    (height + block_dim[1] - 1) // block_dim[1])
                    
        path_kernels[grid_dim, block_dim](
            cuda_arrays['y_t_cuda'], 
            cuda_arrays['f_t_cuda'], 
            cuda_arrays['y_temp_cuda'], 
            cuda_arrays['k1_cuda'], 
            cuda_arrays['k2_cuda'], 
            cuda_arrays['k3_cuda'], 
            cuda_arrays['k4_cuda'], 
            timestep, 
            maxsteps, 
            cuda_arrays['pos_cuda'], 
            cuda_arrays['vel_cuda'], 
            camera_params['inv_view_matrix'], 
            DISKINNERSQR, 
            DISKOUTERSQR, 
            camera_params['DIST_CAM'], 
            camera_params['SIN_I'], 
            cuda_arrays['ray_dir_screen_cuda'], 
            cuda_arrays['ray_dir_world_cuda'], 
            camera_params['image_plane_width'], 
            camera_params['image_plane_height'], 
            camera_position[0], 
            camera_position[1],
            camera_position[2],
            history_cuda,
            samplerate,
            min_h_factor,
            max_h_factor,
            adapt_threshold
        )
        
        history_modified = history_cuda.copy_to_host()

    e = timer()
    print(f"Rendering time: {e-s:.3f} seconds")
    
    return image, history_modified

def render_disk(width, height, inner_radius, outer_radius, camera_position, 
               timestep, maxsteps, fov, integrator_name="Runge-Kutta 4", show_paths=False, samplerate=50):
    """
    Render black hole with accretion disk showing redshift effects.
    
    Args:
        width (int): Image width in pixels
        height (int): Image height in pixels
        inner_radius (float): Inner radius of the accretion disk
        outer_radius (float): Outer radius of the accretion disk
        camera_position (list): 3D position of the camera [x, y, z]
        timestep (float): Time step for integration
        maxsteps (int): Maximum number of integration steps
        fov (float): Field of view in degrees
        integrator_name (str): Name of the integration method to use
    
    Returns:
        tuple: (image, history) - the rendered image and ray path history
    """
    width = int(width)
    height = int(height)
    print(f"Rendering with dimensions: {width}x{height}")
    image = np.zeros((height, width), dtype=np.float32)
    used_points = np.zeros(height * width, dtype=np.uint16)
    used_points_cuda = cuda.to_device(used_points)
    
    if show_paths:
        # history = np.zeros([height*width, int(maxsteps), 3], dtype="float32")
        history = np.zeros([int(math.ceil(width/samplerate) * math.ceil(height/samplerate)), int(maxsteps), 3], dtype="float32")
        history_cuda = cuda.to_device(history)

    DISKINNERSQR = float(inner_radius**2)
    DISKOUTERSQR = float(outer_radius**2)
    
    camera_params = setup_camera(camera_position, fov, width, height)
    cuda_arrays = initialize_cuda_arrays(width, height)
    
    if integrator_name not in DISK_IMAGE_KERNELS:
        print(f"Warning: Integration method {integrator_name} not available. Using Runge-Kutta 4 instead.")
        integrator_name = "Runge-Kutta 4"
    
    disk_kernel = DISK_IMAGE_KERNELS[integrator_name]
    path_kernel = PATH_KERNELS[integrator_name]

    if integrator_name in ["Adams-Moulton4", "Adams-Bashforth4", "Bowie", "Obrechkoff"]:
        nthreads = 16
    else:
        nthreads = 32
    nblocksy = (width//nthreads) + 1
    nblocksx = (height//nthreads) + 1
    
    s = timer()
    disk_kernel[(nblocksx, nblocksy), (nthreads, nthreads)](
        image, 
        cuda_arrays['y_t_cuda'], 
        cuda_arrays['f_t_cuda'], 
        cuda_arrays['y_temp_cuda'], 
        cuda_arrays['k1_cuda'], 
        cuda_arrays['k2_cuda'], 
        cuda_arrays['k3_cuda'], 
        cuda_arrays['k4_cuda'], 
        timestep, 
        maxsteps, 
        cuda_arrays['pos_cuda'], 
        cuda_arrays['vel_cuda'], 
        camera_params['inv_view_matrix'], 
        DISKINNERSQR, 
        DISKOUTERSQR, 
        camera_params['DIST_CAM'], 
        camera_params['SIN_I'], 
        cuda_arrays['ray_dir_screen_cuda'], 
        cuda_arrays['ray_dir_world_cuda'], 
        camera_params['image_plane_width'], 
        camera_params['image_plane_height'], 
        camera_position[0],
        camera_position[1],
        camera_position[2], 
        camera_params['sin_psi'],
        used_points_cuda
    )
    used_points = used_points_cuda.copy_to_host()

    history_modified = None
    
    if show_paths:
        block_dim = (16, 16)
        grid_dim = ((width + block_dim[0] - 1) // block_dim[0], 
                    (height + block_dim[1] - 1) // block_dim[1])
        path_kernel[grid_dim, block_dim](
            cuda_arrays['y_t_cuda'], 
            cuda_arrays['f_t_cuda'], 
            cuda_arrays['y_temp_cuda'], 
            cuda_arrays['k1_cuda'], 
            cuda_arrays['k2_cuda'], 
            cuda_arrays['k3_cuda'], 
            cuda_arrays['k4_cuda'], 
            timestep, 
            maxsteps, 
            cuda_arrays['pos_cuda'], 
            cuda_arrays['vel_cuda'], 
            camera_params['inv_view_matrix'], 
            DISKINNERSQR, 
            DISKOUTERSQR, 
            camera_params['DIST_CAM'], 
            camera_params['SIN_I'], 
            cuda_arrays['ray_dir_screen_cuda'], 
            cuda_arrays['ray_dir_world_cuda'], 
            camera_params['image_plane_width'], 
            camera_params['image_plane_height'], 
            camera_position[0], 
            camera_position[1],
            camera_position[2],
            history_cuda,
            samplerate
        )
        
        history_modified = history_cuda.copy_to_host()

    e = timer()
    print(f"Rendering time: {e-s:.3f} seconds")
    
    return image, history_modified, np.sum(used_points)