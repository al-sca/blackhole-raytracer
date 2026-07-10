"""
Utility functions for black hole ray tracing.
"""
import cv2
import math
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from PyQt5.QtGui import QImage
from matplotlib.colors import BoundaryNorm, ListedColormap

def zoom_at(img, zoom=1, angle=0, coord=None):
    """
    Zoom an image with specified parameters.
    
    Args:
        img (ndarray): Input image
        zoom (float): Zoom factor
        angle (float): Rotation angle in degrees
        coord (tuple): Coordinates to zoom at (x, y)
        
    Returns:
        ndarray: Zoomed image
    """
    cy, cx = [i/2 for i in img.shape[:-1]] if coord is None else coord[::-1]   
    rot_mat = cv2.getRotationMatrix2D((cx, cy), angle, zoom)
    result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def create_colormap(colormap_name='hot'):
    """
    Create a colormap for rendering.
    
    Args:
        colormap_name (str): Name of the matplotlib colormap
        
    Returns:
        tuple: Grayscale values and colored values arrays
    """
    grayscale_values = np.arange(0, 256, 1, dtype=np.uint8)
    colored_values = plt.get_cmap(colormap_name)(grayscale_values)
    colored_values = (colored_values[:, :3] * 255).astype(np.uint8)
    return grayscale_values, colored_values

def array_to_qimage(image_array, colored_values):
    """
    Convert numpy array to QImage for display.
    
    Args:
        image_array (ndarray): Input image array
        colored_values (ndarray): Colormap values
        
    Returns:
        QImage: Image for display
    """
    
    if image_array is None:
        print("Error: No image data to convert")
        return None
        
    array = np.rot90(image_array, 2)
    height, width = array.shape
    uniques = np.unique(array)
    print(uniques[0], uniques[1], uniques[2], "...", uniques[-3], uniques[-2], uniques[-1])
    
    # Normalize values for display
    array[array < 0] *= -1
    array[array > 1] -= 1
    array[(array > 0) & (array < 1)] *= 255
    array = array.astype(np.uint8)
    
    color_image = colored_values[array]
    return QImage(color_image, width, height, QImage.Format_RGB888)

def map_redshift_to_color_log_g(z_array, b_or_o=False, colormap_name='RdBu_r', vmin_g=None, vmax_g=None, error_val=0.0, perc_low=1.0, perc_high=99.0, debug_mask=True):
    """
    Maps redshift/blueshift values (z) to colors using logarithmic scaling of g.
    Background will be black for invalid values or errors.
    Includes enhanced error checking and debug options.

    Args:
        z_array (np.ndarray): Array of z values (z > 0 redshift, z < 0 blueshift).
        colormap_name (str): Name of the matplotlib colormap to use. 'RdBu_r' recommended.
        vmin_g (float, optional): Minimum g value for scaling. If None, calculated using perc_low.
        vmax_g (float, optional): Maximum g value for scaling. If None, calculated using perc_high.
        error_val (float): Value in z_array indicating an error or no intersection. ***CRITICAL: Ensure this matches simulation output for background***
        perc_low (float): Lower percentile for robust range finding if vmin_g is None.
        perc_high (float): Upper percentile for robust range finding if vmax_g is None.
        debug_mask (bool): If True, prints mask info and optionally displays the final mask.

    Returns:
        QImage: Color image for display with black background.
    """

    height, width = z_array.shape
    uniques = np.unique(z_array)
    print("Redshift values:", uniques[0], uniques[1], uniques[2], "...", uniques[-3], uniques[-2], uniques[-1])
    if b_or_o:
        z_array = np.rot90(z_array, 2)
    else:
        z_array = np.fliplr(z_array)

    image_data = np.zeros((height, width, 3), dtype=np.uint8)
    initial_valid_mask = (z_array != error_val) & np.isfinite(z_array)

    def rtn():
        q_image = QImage(image_data.data, width, height, width * 3, QImage.Format_RGB888)
        q_image._buffer = image_data
        return q_image

    if not np.any(initial_valid_mask):
        print("No valid pixels found after initial masking.")
        return rtn()

    g_array = np.full(z_array.shape, np.nan, dtype=float)
    blueshift_mask = initial_valid_mask & (z_array < 0)
    redshift_mask = initial_valid_mask & (z_array >= 0)
    
    g_array[blueshift_mask] = 1.0 - z_array[blueshift_mask]
    denominator_g = 1.0 + z_array[redshift_mask]
    denominator_g[denominator_g == 0] = np.nan
    g_array[redshift_mask] = 1.0 / denominator_g

    valid_g_mask = initial_valid_mask & np.isfinite(g_array) & (g_array > 1e-9)
    valid_g = g_array[valid_g_mask]

    if valid_g.size == 0:
        return rtn()

    log_g_array = np.full(g_array.shape, np.nan, dtype=float)
    log_g_array[valid_g_mask] = np.log10(valid_g)
    
    valid_log_g_mask = valid_g_mask & np.isfinite(log_g_array)
    valid_log_g = log_g_array[valid_log_g_mask]

    if valid_log_g.size == 0:
        return rtn()

    min_log_g_val = np.percentile(valid_log_g, perc_low) if vmin_g is None else np.log10(max(vmin_g, 1e-9))
    max_log_g_val = np.percentile(valid_log_g, perc_high) if vmax_g is None else np.log10(vmax_g)

    if max_log_g_val <= min_log_g_val:
       max_log_g_val = min_log_g_val + 1e-6
       
    norm_log_g = np.full(log_g_array.shape, np.nan, dtype=float)
    denominator_norm = max_log_g_val - min_log_g_val
    if denominator_norm > 1e-9:
        norm_log_g[valid_log_g_mask] = (log_g_array[valid_log_g_mask] - min_log_g_val) / denominator_norm
    else: 
        norm_log_g[valid_log_g_mask] = 0.5

    norm_log_g = np.clip(norm_log_g, 0.0, 1.0)
    final_color_mask = valid_log_g_mask & np.isfinite(norm_log_g)

    if not np.any(final_color_mask):
        print("No pixels passed final masking stage.")
        return rtn()

    cmap = cm.get_cmap(colormap_name)
    colors = cmap(np.linspace(0, 1, 256))
    inverted_cmap = ListedColormap(colors[::-1])
    inverted_cmap.set_bad(color='black', alpha=1.0) 
    colored_rgba = inverted_cmap(norm_log_g[final_color_mask])
    rgb_values = (colored_rgba[:, :3] * 255).astype(np.uint8)
    image_data[final_color_mask] = rgb_values

    image_data = np.require(image_data, np.uint8, 'C')
    q_image = QImage(image_data.data, width, height, width * 3, QImage.Format_RGB888)
    q_image._buffer = image_data

    return q_image


# def array_to_qimage_bowie(image_array, colored_values):
#     """
#     Convert numpy array to QImage for display.
    
#     Args:
#         image_array (ndarray): Input image array
#         colored_values (ndarray): Colormap values
        
#     Returns:
#         QImage: Image for display
#     """

#     if image_array is None:
#         print("Error: No image data to convert")
#         return None
        
#     array = np.fliplr(image_array)
#     height, width = array.shape
    
#     # Normalize values for display
#     array[array < 0] *= -1
#     array[array > 1] -= 1
#     array[(array > 0) & (array < 1)] *= 255
#     array = array.astype(np.uint8)
    
#     color_image = colored_values[array]
#     return QImage(color_image, width, height, QImage.Format_RGB888)

# def show_redshift_contours(image):
#     """
#     Display contour lines of constant redshift.
    
#     Args:
#         image (ndarray): Redshift image
#     """
#     image = np.fliplr(image)
#     fig, ax = plt.subplots()
#     CS = ax.contour(image, levels=np.linspace(-2, 20, 200))
#     CB = plt.colorbar(CS, shrink=0.8, extend='both')
#     ax.clabel(CS, inline=True, fontsize=10)
    
#     l, b, w, h = plt.gca().get_position().bounds
#     ll, bb, ww, hh = CB.ax.get_position().bounds
#     CB.ax.set_position([ll, b + 0.1*h, ww, h*0.8])
    
#     ax.set_title('Curves of constant z')
#     plt.show()

def show_redshift_contours_log(image, num_levels=20, base_max_z=None, b_or_o=False):
    """
    Display contour lines of constant redshift using log spacing for z > 1.

    Args:
        image (ndarray): Redshift image (z values)
        num_levels (int): Approximate number of contour levels desired.
        base_max_z (float, optional): Use this as the max z for level generation,
                                     otherwise calculated from data.
    """
    if b_or_o:
        image = np.fliplr(np.rot90(image, 2))
    valid_mask = np.isfinite(image) & (image != 0)
    valid_z = image[valid_mask]

    if valid_z.size == 0:
        print("No valid redshift data found for contours.")
        return

    image_display = np.fliplr(image)

    min_z = np.percentile(valid_z[valid_z < 0], 1) if np.any(valid_z < 0) else 0
    max_z = np.percentile(valid_z[valid_z > 0], 99) if np.any(valid_z > 0) else 0
    
    if base_max_z is not None:
         max_z_for_levels = base_max_z
    else:
         max_z_for_levels = max_z
         
    print(f"Contour level range estimated: min_z ~ {min_z:.2f}, max_z ~ {max_z:.2f}")
    print(f"Generating levels up to: {max_z_for_levels:.2f}")

    levels = []
    if min_z < -0.01:
       levels.extend(np.linspace(min_z, -0.05, int(num_levels * 0.2)))

    levels.extend(np.linspace(0, 1, int(num_levels * 0.4)))

    if max_z_for_levels > 1:
       levels.extend(np.geomspace(1.01, max_z_for_levels, int(num_levels * 0.4)))

    levels = sorted(list(set(np.round(levels, 3))))

    if not levels:
         print("Could not generate valid contour levels.")
         return
         
    print(f"Generated {len(levels)} levels: {levels}")

    fig, ax = plt.subplots()
    cmap = plt.get_cmap('RdBu_r') 
    CS = ax.contour(image_display, levels=levels, cmap=cmap, linewidths=0.8)
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    CB = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.8, extend='both')

    label_levels = [lvl for lvl in levels if abs(lvl) < 1 or lvl in np.geomspace(1.01, max_z_for_levels, 5)] # Select fewer labels
    ax.clabel(CS, levels=label_levels, inline=True, fontsize=8, fmt='%.2f')

    l, b, w, h = plt.gca().get_position().bounds
    ll, bb, ww, hh = CB.ax.get_position().bounds
    CB.ax.set_position([ll, b + 0.1*h, ww, h*0.8])

    ax.set_title('Curves of constant z (Log Spacing for z>1)')
    ax.set_xlim(0, image_display.shape[1])
    ax.set_ylim(0, image_display.shape[0])
    plt.show()

def create_redshift_histogram(image):
    """
    Display histogram of redshift values.
    
    Args:
        image (ndarray): Redshift image
    """
    plt.figure()
    image_flat = image.flatten()
    image_flat = image_flat[image_flat != 0]  # Remove zeros
    
    plt.hist(image_flat, alpha=0.7)
    plt.xlabel('Redshift Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Redshift Values')
    plt.grid(True)
    plt.show()

# def display_paths(history, r, angle):
#     """
#     Display photon paths visualization.
    
#     Args:
#         history (ndarray): Path history array
#         r (float): Distance
#         angle (float): Angle in radians
#     """
    
#     plt.figure()
#     ax = plt.gca()
#     ax.set_title(f"Photon paths in xz-plane, r={r}, theta={round(math.degrees(angle))}")
#     ax.hlines(y=0, xmin=-r, xmax=r, linestyles='--', lw=0.5)
    
#     for i, line_data in enumerate(history):
#         x_values = line_data[:, 0]
#         y_values = line_data[:, 1]
#         non_zero_indices = np.where((x_values != 0.0) & (y_values != 0.0))
#         x_values = x_values[non_zero_indices]
#         y_values = y_values[non_zero_indices]
#         if len(x_values) > 0:
#             ax.plot(x_values, y_values, '-', lw=0.5)
    
#     plt.draw()
#     plt.pause(0.001)