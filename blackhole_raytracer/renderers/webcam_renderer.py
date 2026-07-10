"""
Webcam renderer for black hole visualization.
"""
import numpy as np
import cv2
from numba import cuda
from timeit import default_timer as timer
import traceback

from ..core.physics import setup_camera, initialize_cuda_arrays
from ..core.cuda_kernels import IMAGE_COORDS_KERNELS, render_projected_image
from ..core.utils import zoom_at
from ..core.cuda_kernels_adaptive import IMAGE_COORDS_KERNELS_ADAPTIVE

def render_webcam_loop_adaptive(width, height, camera_position, timestep, maxsteps, checked, 
                      integrator_name="Runge-Kutta 4",
                      min_h_factor=0.1, max_h_factor=2.0, adapt_threshold=0.01):
    """
    Continuous rendering loop for webcam input with black hole distortion using adaptive step size.
    
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
    """
    camera_params = setup_camera(camera_position, 90, width, height)
    cuda_arrays = initialize_cuda_arrays(width, height)
    
    if integrator_name not in IMAGE_COORDS_KERNELS_ADAPTIVE:
        print(f"Warning: Integration method {integrator_name} not available. Using Runge-Kutta 4 instead.")
        integrator_name = "Runge-Kutta 4"
        
    image_coords_kernel = IMAGE_COORDS_KERNELS_ADAPTIVE[integrator_name]

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
    
    s_init = timer()
    print(f"Calculating ray trajectories with {integrator_name} (adaptive)...")
    
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
    
    e_init = timer()
    print(f"Initialization time: {e_init-s_init:.3f} seconds")
    
    pixel_buffer = pixel_coords_cuda.copy_to_host()
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    print("Try opening webcam")
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        if not cap.isOpened():
            print("Failed to open the webcam with standard methods, trying MSMF...")
            cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
    
        if not cap.isOpened():
            print("Failed to open any webcam")
            return
            
        cap.set(3, width)
        cap.set(4, height)
        
        actual_width = int(cap.get(3))
        actual_height = int(cap.get(4))
        print(f"Requested webcam resolution: {width}x{height}")
        print(f"Actual webcam resolution: {actual_width}x{actual_height}")
    
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                
                if frame.shape[0] != height or frame.shape[1] != width:
                    frame = cv2.resize(frame, (width, height))
                
                s = timer()
                
                if frame is None or frame.size == 0:
                    print("Invalid frame received")
                    continue
                    
                frame = np.ascontiguousarray(frame)
                background_data = cuda.to_device(frame)
                image_cuda = cuda.to_device(image)
                
                render_projected_image[(nblocksx, nblocksy), (nthreads, nthreads)](
                    image_cuda, background_data, pixel_buffer
                )
                
                image = image_cuda.copy_to_host()
                
                e = timer()
                t = round((e-s)*1000, 2)
                fps = 1/(t/1000) if t > 0 else 0
                print(f"Time [ms]: {t}, FPS: {fps}")
                
                cv2.putText(image, f"Adaptive: {integrator_name}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(image, f"Min step: {min_h_factor:.2f} | Max step: {max_h_factor:.2f}", 
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                display_img = zoom_at(image.copy(), 1.3, 180)
                cv2.imshow("Black Hole Webcam", display_img)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
    except Exception as e:
        print(f"Error in webcam loop: {e}")
        traceback.print_exc()
        cv2.destroyAllWindows()

def render_webcam_loop(width, height, camera_position, timestep, maxsteps, checked, 
                      integrator_name="Runge-Kutta 4"):
    """
    Continuous rendering loop for webcam input with black hole distortion.
    
    Args:
        width (int): Image width in pixels
        height (int): Image height in pixels
        camera_position (list): 3D position of the camera [x, y, z]
        timestep (float): Time step for integration
        maxsteps (int): Maximum number of integration steps
        checked (bool): Whether to render the point exactly in the plane (True) or slightly before (False)
        integrator_name (str): Name of the integration method to use
    """
    camera_params = setup_camera(camera_position, 90, width, height)
    cuda_arrays = initialize_cuda_arrays(width, height)

    if integrator_name not in IMAGE_COORDS_KERNELS:
        print(f"Warning: Integration method {integrator_name} not available. Using Runge-Kutta 4 instead.")
        integrator_name = "Runge-Kutta 4"
        
    image_coords_kernel = IMAGE_COORDS_KERNELS[integrator_name]

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
    
    # Only need to do this once
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
    
    print("Try opening webcam")
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        if not cap.isOpened():
            print("Failed to open the webcam with standard methods, trying MSMF...")
            cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
    
        if not cap.isOpened():
            print("Failed to open any webcam")
            return
            
        cap.set(3, width)
        cap.set(4, height)
        
        actual_width = int(cap.get(3))
        actual_height = int(cap.get(4))
        print(f"Requested webcam resolution: {width}x{height}")
        print(f"Actual webcam resolution: {actual_width}x{actual_height}")
    
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                
                if frame.shape[0] != height or frame.shape[1] != width:
                    frame = cv2.resize(frame, (width, height))
                
                s = timer()
                
                if frame is None or frame.size == 0:
                    print("Invalid frame received")
                    continue
                    
                frame = np.ascontiguousarray(frame)               
                print(f"Frame dimensions: {frame.shape[1]}x{frame.shape[0]}, Image dimensions: {width}x{height}")
                background_data = cuda.to_device(frame)
                
                render_projected_image[(nblocksx, nblocksy), (nthreads, nthreads)](
                    image, background_data, pixel_buffer
                )
                
                e = timer()
                t = round((e-s)*1000, 2)
                fps = 1/(t/1000) if t > 0 else 0
                print(f"Time [ms]: {t}, FPS: {fps}")
                
                display_img = zoom_at(image.copy(), 1.3, 180)
                cv2.imshow("Black Hole Webcam", display_img)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
    except Exception as e:
        print(f"Error in webcam loop: {e}")
        traceback.print_exc()
        cv2.destroyAllWindows()