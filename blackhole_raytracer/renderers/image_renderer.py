"""
Image renderer for black hole visualization.
"""
import os
import numpy as np
import cv2
from numba import cuda
from timeit import default_timer as timer
from PIL import Image

from ..core.physics import setup_camera, initialize_cuda_arrays
from ..core.cuda_kernels import IMAGE_COORDS_KERNELS, render_projected_image
from ..core.cuda_kernels_adaptive import IMAGE_COORDS_KERNELS_ADAPTIVE

def render_milky_way_adaptive(width, height, camera_position, timestep, maxsteps, checked, 
                     integrator_name="Runge-Kutta 4",
                     min_h_factor=0.1, max_h_factor=2.0, adapt_threshold=0.01):
    """
    Render the Milky Way image with black hole distortion using adaptive step size.
    
    Args:
        width (int): Image width in pixels
        height (int): Image height in pixels
        camera_position (list): 3D position of the camera [x, y, z]
        timestep (float): Base time step for integration (will be adjusted if adaptive is enabled)
        maxsteps (int): Maximum number of integration steps
        checked (bool): Whether to render the point exactly in the plane (True) or slightly before (False)
        integrator_name (str): Name of the integration method to use
        min_h_factor (float): Minimum step size factor (relative to timestep)
        max_h_factor (float): Maximum step size factor (relative to timestep)
        adapt_threshold (float): Threshold for adaptation
    
    Returns:
        ndarray: The rendered Milky Way with gravitational distortion
    """
    image_path = os.environ.get("MILKY_WAY_IMAGE_PATH", "images/eso0932a.jpg")
    try:
        img = np.array(Image.open(image_path))
    except Exception as e:
        print(f"Error loading Milky Way image: {e}")
        try:
            img = cv2.imread(image_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e2:
            print(f"Fallback loading also failed: {e2}")
            return None
            
    if img is None:
        print(f"Could not load image from {image_path}")
        return None
    
    if integrator_name not in IMAGE_COORDS_KERNELS_ADAPTIVE:
        print(f"Warning: Integration method {integrator_name} not available. Using Runge-Kutta 4 instead.")
        integrator_name = "Runge-Kutta 4"
    
    image_coords_kernel = IMAGE_COORDS_KERNELS_ADAPTIVE[integrator_name]
               
    camera_params = setup_camera(camera_position, 90, width, height)
    cuda_arrays = initialize_cuda_arrays(width, height)
    
    inter_point = np.zeros([width*height, 3], dtype=np.float32)
    inter_point_cuda = cuda.to_device(inter_point)
    pixel_coords = np.zeros((height, width, 2), dtype=np.uint32)
    pixel_coords_cuda = cuda.to_device(pixel_coords)
    
    if integrator_name in ["Adams-Moulton4", "Adams-Bashforth4", "Bowie", "Obrechkoff"]:
        nthreads = 8
    else:
        nthreads = 16
            
    nblocksy = (width//nthreads) + 1
    nblocksx = (height//nthreads) + 1
    
    print(f"Using thread configuration: blocks={nblocksx}x{nblocksy}, threads={nthreads}x{nthreads}")
    
    s = timer()
    
    image_coords_kernel[(nblocksx, nblocksy), (nthreads, nthreads)](
        pixel_coords_cuda, 
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
        camera_params['view_matrix'],
        cuda_arrays['ray_dir_screen_cuda'], 
        cuda_arrays['ray_dir_world_cuda'], 
        camera_params['image_plane_width'], 
        camera_params['image_plane_height'], 
        camera_position[0],
        camera_position[1],
        camera_position[2],
        inter_point_cuda,
        checked,
        min_h_factor,
        max_h_factor,
        adapt_threshold
    )
    
    pixel_buffer = pixel_coords_cuda.copy_to_host()
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    try:
        img_contiguous = np.ascontiguousarray(img)
        background_data = cuda.to_device(img_contiguous)
        image_cuda = cuda.to_device(image)
        
        end = timer()
        print(f"Coordinate calculation time: {end-s:.3f} seconds")
        
        render_projected_image[(nblocksx, nblocksy), (nthreads, nthreads)](
            image_cuda, background_data, pixel_buffer
        )
        
        image = image_cuda.copy_to_host()
        
        end = timer()
        print(f"Total rendering time: {end-s:.3f} seconds")
        
    except Exception as e:
        print(f"Error rendering the image: {e}")
        return None

    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

def render_milky_way(width, height, camera_position, timestep, maxsteps, checked, 
                     integrator_name="Runge-Kutta 4"):
    """
    Render the Milky Way image with black hole distortion.
    
    Args:
        width (int): Image width in pixels
        height (int): Image height in pixels
        camera_position (list): 3D position of the camera [x, y, z]
        timestep (float): Time step for integration
        maxsteps (int): Maximum number of integration steps
        checked (bool): Whether to render the point exactly in the plane (True) or slightly before (False)
        image_path (str): Path to the Milky Way image
        integrator_name (str): Name of the integration method to use
    
    Returns:
        ndarray: The rendered Milky Way with gravitational distortion
    """
    image_path = os.environ.get("MILKY_WAY_IMAGE_PATH", "images/eso0932a.jpg")
    try:
        img = np.array(Image.open(image_path))
    except Exception as e:
        print(f"Error loading Milky Way image: {e}")
        try:
            img = cv2.imread(image_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e2:
            print(f"Fallback loading also failed: {e2}")
            return None
            
    if img is None:
        print(f"Could not load image from {image_path}")
        return None
    
    if integrator_name not in IMAGE_COORDS_KERNELS:
        print(f"Warning: Integration method {integrator_name} not available. Using Runge-Kutta 4 instead.")
        integrator_name = "Runge-Kutta 4"
        
    image_coords_kernel = IMAGE_COORDS_KERNELS[integrator_name]
    
    camera_params = setup_camera(camera_position, 90, width, height)
    cuda_arrays = initialize_cuda_arrays(width, height)
    
    inter_point = np.zeros([width*height, 3], dtype=np.float32)
    inter_point_cuda = cuda.to_device(inter_point)
    pixel_coords = np.zeros((height, width, 2), dtype=np.uint32)
    pixel_coords_cuda = cuda.to_device(pixel_coords)
    
    if integrator_name in ["Adams-Moulton4", "Adams-Bashforth4", "Bowie", "Obrechkoff"]:
        nthreads = 16
    else:
        nthreads = 32
    nblocksy = (width//nthreads) + 1
    nblocksx = (height//nthreads) + 1

    print(f"Using thread configuration: blocks={nblocksx}x{nblocksy}, threads={nthreads}x{nthreads}")
    
    s = timer()

    image_coords_kernel[(nblocksx, nblocksy), (nthreads, nthreads)](
        pixel_coords_cuda, 
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
        camera_params['view_matrix'],
        cuda_arrays['ray_dir_screen_cuda'], 
        cuda_arrays['ray_dir_world_cuda'], 
        camera_params['image_plane_width'], 
        camera_params['image_plane_height'], 
        camera_position[0],
        camera_position[1],
        camera_position[2],
        inter_point_cuda,
        checked
    )
    
    pixel_buffer = pixel_coords_cuda.copy_to_host()
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    try:
        img_contiguous = np.ascontiguousarray(img)
        background_data = cuda.to_device(img_contiguous)
        
        s = timer()
        render_projected_image[(nblocksx, nblocksy), (nthreads, nthreads)](
            image, background_data, pixel_buffer
        )
        e = timer()
        print(f"Rendering time: {e-s:.3f} seconds")
        
    except Exception as e:
        print(f"Error rendering the image: {e}")
        return None

    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)