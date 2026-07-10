"""
Integration Method Analysis
"""
import os
import math
import time
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from matplotlib.colors import BoundaryNorm, ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

from blackhole_raytracer.renderers.disk_renderer import render_disk
from blackhole_raytracer.core.utils import create_colormap 
from blackhole_raytracer.core.integrators import get_available_methods

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

class IntegrationMethodAnalyzer:
    """Class to analyze and visualize differences between integration methods"""
    
    def __init__(self, inner_radius=3, outer_radius=10, inclination=80, 
                        distance=4.5, resolution=500, timestep=0.08, maxsteps=5000, 
                        fov=90, output_dir="method_comparison"):
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.inclination = inclination
        self.distance = distance
        self.resolution = resolution
        self.timestep = timestep
        self.maxsteps = maxsteps
        self.fov = fov
        self.output_dir = output_dir
        self.methods = [
            "Runge-Kutta 4", 
            "Euler",
            "Adams-Bashforth",
            "Adams-Bashforth4", 
            "Adams-Moulton4", 
            "Bowie",
            "Obrechkoff",
        ]
        # self.colors = plt.cm.viridis(np.linspace(0, 1, 7))
        self.colors = plt.cm.managua(np.linspace(0, 1, 7))
        
    def render_comparison_images(self):
        """
        Render comparison images using different integration methods
        
        Args:
            inner_radius (float): Inner radius of the accretion disk
            outer_radius (float): Outer radius of the accretion disk
            inclination (float): Inclination angle in degrees
            distance (float): Camera distance from black hole
            resolution (int): Image resolution (width and height)
            timestep (float): Time step for integration
            maxsteps (int): Maximum number of integration steps
            fov (float): Field of view in degrees
            output_dir (str): Directory to save rendered images
            
        Returns:
            dict: Dictionary containing rendered images and metadata for each method
        """
        os.makedirs(self.output_dir, exist_ok=True)
        
        angle = math.pi/2 - math.radians(self.inclination)
        x = self.distance * math.cos(angle)
        y = self.distance * math.sin(angle)
        camera_position = [0.0, y, x]
        used_points = 0
        results = {}
        
        for method in self.methods:
            print(f"Rendering accretion disk using: {method}")
            
            try:
                # warmup
                image, history, used_points = render_disk(
                    self.resolution, self.resolution,
                    self.inner_radius, self.outer_radius,
                    camera_position,
                    self.timestep, self.maxsteps, self.fov,
                    method,
                    show_paths=False
                )
                
                render_times = np.zeros(10, dtype=np.float64)
                for idx in range(10):
                    start_time = time.time()
                    image, history, used_points = render_disk(
                        self.resolution, self.resolution,
                        self.inner_radius, self.outer_radius,
                        camera_position,
                        self.timestep, self.maxsteps, self.fov,
                        method,
                        show_paths=False
                    )
                    end_time = time.time()
                    render_times[idx] = end_time - start_time
                    print(idx)
                    
                render_time = np.mean(render_times)
                render_time_std = np.std(render_times)
                print(f"--- Mean Render Time: {render_time:.2f}")
                print(f"--- Std Render Time: {render_time_std:.2f}")

                # render_time = end_time - start_time
                # print(f"--- Total Render Time: {render_time:.2f} seconds")
                # print(f"--- Mean Render Time: {render_time/10:.2f}")
                # render_time = render_time/10
                # array = None
                # if method in ["Bowie", "Obrechkoff"]:
                #     array = np.fliplr(image)
                # else:
                #     array = np.rot90(image, 2)
                # height, width = array.shape
                # array[array < 0] *= -1
                # array[array > 1] -= 1
                # array[(array > 0) & (array < 1)] *= 255
                # array = array.astype(np.uint8)
                # uniques = np.unique(array)
            
                # grayscale_values, colored_values = create_colormap('hot')
                # color_image = colored_values[array]
                colormap_name='hot'
                vmin_g=None
                vmax_g=None
                error_val=0.0
                perc_low=1.0
                perc_high=99.0
                height, width = image.shape

                if method in ["Bowie", "Obrechkoff"]:
                    z_array = np.fliplr(image)
                else:
                    z_array = np.rot90(image, 2)
                
                image = z_array

                image_data = np.zeros((height, width, 3), dtype=np.uint8)
                initial_valid_mask = (z_array != error_val) & np.isfinite(z_array)

                g_array = np.full(z_array.shape, np.nan, dtype=float)
                blueshift_mask = initial_valid_mask & (z_array < 0)
                redshift_mask = initial_valid_mask & (z_array >= 0)
                
                g_array[blueshift_mask] = 1.0 - z_array[blueshift_mask]
                denominator_g = 1.0 + z_array[redshift_mask]
                denominator_g[denominator_g == 0] = np.nan
                g_array[redshift_mask] = 1.0 / denominator_g

                valid_g_mask = initial_valid_mask & np.isfinite(g_array) & (g_array > 1e-9)
                valid_g = g_array[valid_g_mask]

                log_g_array = np.full(g_array.shape, np.nan, dtype=float)
                log_g_array[valid_g_mask] = np.log10(valid_g)
                
                valid_log_g_mask = valid_g_mask & np.isfinite(log_g_array)
                valid_log_g = log_g_array[valid_log_g_mask]

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

                cmap = cm.get_cmap(colormap_name)
                colors = cmap(np.linspace(0, 1, 256))
                inverted_cmap = ListedColormap(colors[::-1])
                inverted_cmap.set_bad(color='black', alpha=1.0) 
                colored_rgba = inverted_cmap(norm_log_g[final_color_mask])
                rgb_values = (colored_rgba[:, :3] * 255).astype(np.uint8)
                image_data[final_color_mask] = rgb_values

                image_data = np.require(image_data, np.uint8, 'C')
                fig, ax = plt.subplots(figsize=(8, 8))
                # plt.figure(figsize=(8, 8))
                plt.imshow(image_data)
                cmap = plt.colormaps['hot']
                imin = np.min(image)
                imax = np.max(image)
                norm = matplotlib.colors.Normalize(vmin=imin, vmax=imax)
                sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                fig.colorbar(sm, ax=ax, label='Redshift')
                # plt.colorbar(label='Redshift')
                ax.set_title(f"Method: {method}")
                plt.savefig(os.path.join(self.output_dir, f"{method.replace(' ', '_')}_disk.png"), dpi=300)
                plt.close()
                
                results[method] = {
                    # 'image': color_image.reshape(-1, 3),
                    'image': image,
                    'image_min': imin,
                    'image_max': imax,
                    'color_image': image_data,
                    #https://www.itu.int/rec/R-REC-BT.601-7-201103-I/en
                    #https://dsp.stackexchange.com/questions/65927/why-do-we-use-difference-rgb-to-grayscale-function
                    'color_image_converted': np.dot(image_data[..., :3], [0.2989, 0.5870, 0.1140]), 
                    'render_time': render_time,
                    'render_time_std': render_time_std,
                    'history': history,
                    'used_points': used_points
                }
                
            except Exception as e:
                print(f"Error rendering with {method}: {e}")
                continue
        
        return results
    
    def create_difference_maps(self, results, ref_method="Runge-Kutta 4"):
        """
        Create difference maps between methods
        
        Args:
            results (dict): Dictionary with rendering results for each method
            ref_method (str): Method to use as reference
        """
        if ref_method not in results:
            print(f"Reference method {ref_method} not found in results")
            return
            
        ref_image = results[ref_method]['image']

        comparison_methods = [m for m in self.methods if m != ref_method and m in results]

        if not comparison_methods:
            print("No methods to compare with reference")
            return

        all_diffs = []
        for method in comparison_methods:
            if method not in results:
                continue
            diff = np.abs(results[method]['color_image_converted'] - results[ref_method]['color_image_converted'])
            all_diffs.append(diff)

        all_diffs_array = np.array(all_diffs)
        global_min = np.min(all_diffs_array)
        global_max = np.max(all_diffs_array)
            
        n_methods = len(comparison_methods)
        n_cols = 2
        n_rows = (n_methods + n_cols - 1) // n_cols

        fig = plt.figure(figsize=(12, 10))

        for i, method in enumerate(comparison_methods):
            if method not in results:
                continue
                
            diff = np.abs(results[method]['image'] - ref_image)
            
            mse = mean_squared_error(ref_image.flatten(), results[method]['image'].flatten())
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            
            diff_visuals = np.abs(results[method]['color_image_converted'] - results[ref_method]['color_image_converted'])

            ax = plt.subplot(n_rows, n_cols, i + 1)
            norm = matplotlib.colors.Normalize(vmin=np.min(diff), vmax=max_diff)
            im = ax.imshow(diff_visuals, cmap='hot')
            
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cmap = plt.colormaps['hot']
            norm = matplotlib.colors.Normalize(vmin=np.min(diff), vmax=max_diff)
            
            sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            plt.colorbar(sm, cax=cax, label='Absolute Difference')
            
            ax.set_title(f"{method} vs {ref_method}\nMSE: {mse:.6f}, Max: {max_diff:.6f}, Mean: {mean_diff:.6f}")

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "method_differences.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        for method in comparison_methods:
            if method not in results:
                continue
                
            diff = np.abs(results[method]['image'] - results[ref_method]['image'])
            diff_visuals = np.abs(results[method]['color_image_converted'] - results[ref_method]['color_image_converted'])

            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(diff_visuals, cmap='hot')
            # im = ax.imshow(diff, cmap='hot', vmin=global_min, vmax=global_max)
            
            cmap = plt.colormaps['hot']
            norm = matplotlib.colors.Normalize(vmin=np.min(diff), vmax=np.max(diff))
            sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, label='Absolute Difference')
            
            mse = mean_squared_error(ref_image.flatten(), results[method]['image'].flatten())
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            
            ax.set_title(f"{method} vs {ref_method}\nMSE: {mse:.6f}, Max: {max_diff:.6f}, Mean: {mean_diff:.6f}")
            plt.savefig(os.path.join(self.output_dir, f"{method.replace(' ', '_')}_diff.png"), dpi=300)
            plt.close()
            
        return

    # def create_difference_maps(self, results, ref_method="Runge-Kutta 4"):
    #     """
    #     Create difference maps between methods
        
    #     Args:
    #         results (dict): Dictionary with rendering results for each method
    #         ref_method (str): Method to use as reference
    #         output_dir (str): Directory to save difference maps
    #     """
    #     if ref_method not in results:
    #         print(f"Reference method {ref_method} not found in results")
    #         return
            
    #     ref_image = results[ref_method]['image']
        
    #     comparison_methods = [m for m in self.methods if m != ref_method and m in results]
        
    #     if not comparison_methods:
    #         print("No methods to compare with reference")
    #         return
            
    #     plt.figure(figsize=(12, 4 * len(comparison_methods)))
        
    #     for i, method in enumerate(comparison_methods):
    #         if method not in results:
    #             continue
                
    #         diff = np.abs(results[method]['color_image'] - ref_image)
            
    #         mse = mean_squared_error(ref_image.flatten(), results[method]['color_image'].flatten())
    #         max_diff = np.max(diff)
    #         mean_diff = np.mean(diff)
    #         # min_diff = np.min(diff)
            
    #         plt.subplot(len(comparison_methods), 1, i + 1)
    #         im = plt.imshow(diff, cmap='hot')
    #         plt.colorbar(label='Absolute Difference')
    #         plt.title(f"{method} vs {ref_method} (MSE: {mse:.6f}, Max: {max_diff:.6f}, Mean: {mean_diff:.6f})")
            
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(self.output_dir, "method_differences.png"), dpi=300)
    #     plt.close()
        
    #     for method in comparison_methods:
    #         if method not in results:
    #             continue
                
    #         diff = np.abs(results[method]['color_image'] - ref_image)
            
    #         plt.figure(figsize=(10, 8))
    #         plt.imshow(diff, cmap='hot')
    #         plt.colorbar(label='Absolute Difference')
            
    #         mse = mean_squared_error(ref_image.flatten(), results[method]['color_image'].flatten())
    #         max_diff = np.max(diff)
    #         mean_diff = np.mean(diff)
            
    #         plt.title(f"{method} vs {ref_method}\nMSE: {mse:.6f}, Max: {max_diff:.6f}, Mean: {mean_diff:.6f}")
    #         plt.savefig(os.path.join(self.output_dir, f"{method.replace(' ', '_')}_diff.png"), dpi=300)
    #         plt.close()
            
    #     return
    
    def analyze_redshift_profiles(self, results, output_dir="method_comparison"):
        """
        Analyze redshift profiles across different methods
        
        Args:
            results (dict): Dictionary with rendering results for each method
            output_dir (str): Directory to save analysis plots
        """
        methods = [m for m in self.methods if m in results]

        sample_image = results[methods[0]]['image']
        height, width = sample_image.shape
        
        plt.figure(figsize=(12, 8))
        for method in methods:
            if method not in results:
                continue
                
            image = results[method]['image']
            horizontal_profile = image[height//2, :]
            plt.plot(np.arange(width), horizontal_profile, label=method)
            
        plt.xlabel('Horizontal Pixel Position')
        plt.ylabel('Redshift Value')
        plt.title('Horizontal Redshift Profile Comparison')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, "horizontal_redshift_profile.png"), dpi=300)
        plt.close()
        
        plt.figure(figsize=(12, 8)) 
        for method in methods:
            if method not in results:
                continue
                
            image = results[method]['image']
            vertical_profile = image[:, width//2]
            plt.plot(np.arange(height), vertical_profile, label=method)
            
        plt.xlabel('Vertical Pixel Position')
        plt.ylabel('Redshift Value')
        plt.title('Vertical Redshift Profile Comparison')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, "vertical_redshift_profile.png"), dpi=300)
        plt.close()

        for method in methods:
            if method not in results:
                continue
                
            image = results[method]['image']
            valid_mask = np.isfinite(image) & (image != 0)
            valid_z = image[valid_mask]

            if valid_z.size == 0:
                print(f"No valid redshift data found for contours in method: {method}")
                continue
            
            min_z = np.percentile(valid_z[valid_z < 0], 1) if np.any(valid_z < 0) else 0
            max_z = np.percentile(valid_z[valid_z > 0], 99) if np.any(valid_z > 0) else 0
            
            levels = []
            num_levels = 20
            
            if min_z < -0.01:
                levels.extend(np.linspace(min_z, -0.05, int(num_levels * 0.2)))

            levels.extend(np.linspace(0, 1, int(num_levels * 0.4)))

            if max_z > 1:
                levels.extend(np.geomspace(1.01, max_z, int(num_levels * 0.4)))

            levels = sorted(list(set(np.round(levels, 3))))
            
            if not levels:
                print(f"Could not generate valid contour levels for method: {method}")
                continue
                
            fig, ax = plt.subplots(figsize=(10, 8))
            cmap = plt.get_cmap('RdBu_r')
            norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
            
            x = np.linspace(-1, 1, width)
            y = np.linspace(1, -1, height)
            X, Y = np.meshgrid(x, y)
            
            CS = ax.contour(X, Y, image, levels=levels, cmap=cmap, linewidths=0.8)
            cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.8, extend='both')
            
            label_levels = [lvl for lvl in levels if abs(lvl) < 1 or lvl in np.geomspace(1.01, max_z, min(5, sum(np.array(levels) > 1)))]
            ax.clabel(CS, levels=label_levels, inline=True, fontsize=8, fmt='%.2f')

            l, b, w, h = ax.get_position().bounds
            ll, bb, ww, hh = cb.ax.get_position().bounds
            cb.ax.set_position([ll, b + 0.1*h, ww, h*0.8])
            
            plt.title(f"Redshift Contours: {method} (Log Spacing for z>1)")
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.savefig(os.path.join(self.output_dir, f"{method.replace(' ', '_')}_contours.png"), dpi=300)
            plt.close()
            
        plt.figure(figsize=(12, 8))
        for method in methods:
            if method not in results:
                continue
                
            # image = results[method]['color_image_converted']
            image = results[method]['image']
            values = image.flatten()
            values = values[~np.isnan(values)]
            
            # values = values[(values > -1) & (values < 1)]
            values = values[(values > 10)]
            
            plt.hist(values, bins=50, alpha=0.5, label=method)
            
        plt.xlabel('Redshift Value')
        plt.ylabel('Frequency')
        plt.title('Redshift Value Distribution Comparison')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, "redshift_histogram.png"), dpi=300)
        plt.close()
        
        return
    

    # def analyze_redshift_profiles(self, results, output_dir="method_comparison"):
    #     """
    #     Analyze redshift profiles across different methods
        
    #     Args:
    #         results (dict): Dictionary with rendering results for each method
    #         output_dir (str): Directory to save analysis plots
    #     """
    #     methods = [m for m in self.methods if m in results]

    #     sample_image = results[methods[0]]['color_image_converted']
    #     height, width = sample_image.shape
        
    #     plt.figure(figsize=(12, 8))
    #     for method in methods:
    #         if method not in results:
    #             continue
                
    #         image = results[method]['color_image_converted']
    #         horizontal_profile = image[height//2, :]
    #         plt.plot(np.arange(width), horizontal_profile, label=method)
            
    #     plt.xlabel('Horizontal Pixel Position')
    #     plt.ylabel('Redshift Value')
    #     plt.title('Horizontal Redshift Profile Comparison')
    #     plt.legend()
    #     plt.grid(True)
    #     plt.savefig(os.path.join(self.output_dir, "horizontal_redshift_profile.png"), dpi=300)
    #     plt.close()
        
    #     plt.figure(figsize=(12, 8)) 
    #     for method in methods:
    #         if method not in results:
    #             continue
                
    #         image = results[method]['color_image_converted']
    #         vertical_profile = image[:, width//2]
    #         plt.plot(np.arange(height), vertical_profile, label=method)
            
    #     plt.xlabel('Vertical Pixel Position')
    #     plt.ylabel('Redshift Value')
    #     plt.title('Vertical Redshift Profile Comparison')
    #     plt.legend()
    #     plt.grid(True)
    #     plt.savefig(os.path.join(self.output_dir, "vertical_redshift_profile.png"), dpi=300)
    #     plt.close()
        
    #     for method in methods:
    #         if method not in results:
    #             continue
                
    #         plt.figure(figsize=(10, 8))
    #         x = np.linspace(-1, 1, width)
    #         y = np.linspace(1, -1, height)
    #         X, Y = np.meshgrid(x, y)
    #         CS = plt.contour(X, Y, results[method]['color_image_converted'], levels=20, colors='white')
    #         plt.clabel(CS, inline=True, fontsize=5)
    #         plt.contourf(X, Y, results[method]['color_image_converted'], levels=50, cmap='hot')
    #         plt.colorbar(label='Redshift')
            
    #         plt.title(f"Redshift Contours: {method}")
    #         plt.xlabel('X')
    #         plt.ylabel('Y')
    #         plt.savefig(os.path.join(self.output_dir, f"{method.replace(' ', '_')}_contours.png"), dpi=300)
    #         plt.close()
            
    #     plt.figure(figsize=(12, 8))
    #     for method in methods:
    #         if method not in results:
    #             continue
                
    #         # image = results[method]['color_image_converted']
    #         image = results[method]['image']
    #         values = image.flatten()
    #         values = values[~np.isnan(values)]
            
    #         # values = values[(values > -1) & (values < 1)]
    #         values = values[(values > 10)]
            
    #         plt.hist(values, bins=50, alpha=0.5, label=method)
            
    #     plt.xlabel('Redshift Value')
    #     plt.ylabel('Frequency')
    #     plt.title('Redshift Value Distribution Comparison')
    #     plt.legend()
    #     plt.grid(True)
    #     plt.savefig(os.path.join(self.output_dir, "redshift_histogram.png"), dpi=300)
    #     plt.close()
        
    #     return
    
    def create_performance_summary(self, results):
        """
        Create performance summary of different methods
        
        Args:
            results (dict): Dictionary with rendering results for each method
            output_dir (str): Directory to save summary plots
        """
        methods = [m for m in self.methods if m in results]
            
        render_times = [results[method]['render_time'] for method in methods]
        render_time_stds = [results[method]['render_time_std'] for method in methods]
        
        plt.figure(figsize=(12, 6))
        # plt.bar(methods, render_times, color='skyblue')
        plt.bar(methods, render_times, color=self.colors[:len(methods)], yerr=render_time_stds, capsize=5)
        plt.xlabel('Integration Method')
        plt.ylabel('Render Time (seconds)')
        plt.title('Rendering Performance by Integration Method')
        plt.xticks(rotation=45)
        
        for i, v in enumerate(render_times):
            plt.text(i, v + max(0.1, render_time_stds[i]), f"{v:.2f}s", ha='center')
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "render_time_comparison.png"), dpi=300)
        plt.close()
        
        data = []
        for method in methods:
            if method not in results:
                continue
                
            image = results[method]['image']
            render_time = results[method]['render_time']
            render_time_std = results[method]['render_time_std']
            
            valid_pixels = np.count_nonzero(~np.isnan(image) & (image != 0))
            total_pixels = image.size
            valid_percentage = (valid_pixels / total_pixels) * 100
            
            processing_rate = total_pixels / render_time
            
            data.append({
                'Method': method,
                'Render Time (s)': render_time,
                'Render Time Std (s)': render_time_std,
                'Processing Rate (px/s)': processing_rate,
                'Valid Pixels (%)': valid_percentage
            })

        df = pd.DataFrame(data)
        df.to_csv(os.path.join(self.output_dir, "performance_summary.csv"), index=False)
        
        print("\nPerformance Summary:")
        print(df.to_string(index=False))
        
        return df
    
    def create_summary_radar_chart(self, results, accuracy_metrics=None):
        """
        Create radar chart summarizing method performance across metrics
        
        Args:
            results (dict): Dictionary with rendering results for each method
            accuracy_metrics (dict): Dictionary with accuracy metrics (optional)
            output_dir (str): Directory to save radar chart
        """
        
        
        methods = [m for m in self.methods if m in results]
            
        categories = [
            'Render Speed',
            'Angular Momentum Conservation',
            'Energy Conservation',
            'Periastron Accuracy',
            'Integration Points',
            'Stability (Photon Sphere)'
        ]
        
        N = len(categories)
        
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        plt.xticks(angles[:-1], categories, fontsize=12)
        ax.set_rlabel_position(0)
        plt.yticks([2, 4, 6, 8, 10], ["2", "4", "6", "8", "10"], fontsize=10)
        plt.ylim(0, 10)
        
        render_times = np.array([results[method]['render_time'] for method in methods])
        min_time = np.min(render_times)
        max_time = np.max(render_times)
        render_speed_scores = 10 - 9 * (render_times - min_time) / (max_time - min_time)

        used_points = np.array([results[method]['used_points'] for method in methods])
        min_points = np.min(used_points)
        max_points = np.max(used_points)
        used_points_scores = 10 - 9 * (used_points - min_points) / (max_points - min_points)

        if accuracy_metrics is None:
            accuracy_metrics = {
                "Runge-Kutta 4": {
                    'Angular Momentum Conservation': 5.37,
                    'Energy Conservation': 7.15,
                    'Periastron Accuracy': 7.15,
                    'Stability (Photon Sphere)': 7.17
                },
                "Adams-Bashforth4": {
                    'Angular Momentum Conservation': 3.84,
                    'Energy Conservation': 5.37,
                    'Periastron Accuracy': 10.00,
                    'Stability (Photon Sphere)': 5.17
                },
                "Adams-Bashforth": {
                    'Angular Momentum Conservation': 1.8,
                    'Energy Conservation': 1.8,
                    'Periastron Accuracy': 1.4,
                    'Stability (Photon Sphere)': 5.04
                },
                "Adams-Moulton4": {
                    'Angular Momentum Conservation': 2.47,
                    'Energy Conservation': 2.47,
                    'Periastron Accuracy': 2.47,
                    'Stability (Photon Sphere)': 5.28
                },
                "Bowie": {
                    'Angular Momentum Conservation': 10.00,
                    'Energy Conservation': 3.84,
                    'Periastron Accuracy': 5.37,
                    'Stability (Photon Sphere)': 10.0
                },
                "Obrechkoff": {
                    'Angular Momentum Conservation': 7.15,
                    'Energy Conservation': 10.00,
                    'Periastron Accuracy': 3.84,
                    'Stability (Photon Sphere)': 8.65
                },
                "Euler": {
                    'Angular Momentum Conservation': 1.2,
                    'Energy Conservation': 1.2,
                    'Periastron Accuracy': 0.4,
                    'Stability (Photon Sphere)': 0.5
                }
            }

        colors = plt.cm.managua(np.linspace(0, 1, len(methods)))
        areas = {}
        for i, method in enumerate(methods):
            if method not in results or method not in accuracy_metrics:
                continue
                
            values = [
                render_speed_scores[i],
                accuracy_metrics[method]['Angular Momentum Conservation'],
                accuracy_metrics[method]['Energy Conservation'],
                accuracy_metrics[method]['Periastron Accuracy'],
                used_points_scores[i],
                accuracy_metrics[method]['Stability (Photon Sphere)']
            ]
            
            values += values[:1]
            areas[method] = calculate_radar_chart_area(angles, values)

            ax.plot(angles, values, 'o-', linewidth=2, color=colors[i], label=method)
            ax.fill(angles, values, color=colors[i], alpha=0.1)
        
        print(areas)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.title('Method Comparison Across All Metrics', size=15, y=1.1)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "method_radar_chart.png"), dpi=300)
        plt.close()
        
        return
    
    def create_image_grid(self, results):
        """
        Create a grid of rendered images for all methods
        
        Args:
            results (dict): Dictionary with rendering results for each method
            output_dir (str): Directory to save image grid
        """
        methods = [m for m in self.methods if m in results]
            
        n_methods = len(methods)
        cols = min(3, n_methods)
        rows = (n_methods + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if n_methods == 1:
            axes = np.array([axes])
            
        axes = axes.flatten()
        # cmap = plt.get_cmap('hot')
        imax = 0
        imin = 10000
        for i, method in enumerate(methods):
            if method not in results:
                continue
            
            if imax < results[method]['image_max']:
                imax = results[method]['image_max']
            if imin > results[method]['image_min']:
                imin = results[method]['image_min']

            im = axes[i].imshow(results[method]['color_image'], cmap='hot')
            # im2 = axes[i].imshow(results[method]['image'], cmap='hot')
            # im = axes[i].imshow(results[method]['color_image_converted'])
            axes[i].set_title(f"{method}\nRender Time: {results[method]['render_time']:.2f}s")
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            
        for i in range(n_methods, len(axes)):
            axes[i].axis('off')
            
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        # cbar = fig.colorbar(im, cax=cbar_ax)
        # cbar.set_label('Redshift')
        cmap = plt.colormaps['hot']
        norm = matplotlib.colors.Normalize(vmin=imin, vmax=imax)
        sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax, label='Redshift')
        
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])
        plt.savefig(os.path.join(self.output_dir, "method_comparison_grid.png"), dpi=300)
        plt.close()
        
        return
    
    def run_full_analysis(self):
        """
        Run full analysis pipeline for all methods
        
        Args:
            inner_radius (float): Inner radius of the accretion disk
            outer_radius (float): Outer radius of the accretion disk
            inclination (float): Inclination angle in degrees
            distance (float): Camera distance from black hole
            resolution (int): Image resolution (width and height)
            timestep (float): Time step for integration
            maxsteps (int): Maximum number of integration steps
            fov (float): Field of view in degrees
            output_dir (str): Directory to save analysis results
            
        Returns:
            dict: Dictionary with analysis results
        """
        os.makedirs(self.output_dir, exist_ok=True)
        
        print("Rendering comparison images...")
        results = self.render_comparison_images()
        
        print("Creating difference maps...")
        self.create_difference_maps(results)
        
        print("Analyzing redshift profiles...")
        self.analyze_redshift_profiles(results)
        
        print("Creating performance summary...")
        performance_df = self.create_performance_summary(results)
        
        print("Creating summary radar chart...")
        self.create_summary_radar_chart(results)
        
        print("Creating image grid...")
        self.create_image_grid(results)
        
        self.create_summary_report(results, performance_df)
        
        print(f"Analysis complete. Results saved to {self.output_dir}/")
        return results
    
    def create_summary_report(self, results, performance_df):
        """
        Create a summary report of the analysis
        
        Args:
            results (dict): Dictionary with rendering results for each method
            performance_df (DataFrame): Performance summary DataFrame
            output_dir (str): Directory to save report
        """
        methods = [m for m in self.methods if m in results]
            
        with open(os.path.join(self.output_dir, "summary_report.txt"), 'w') as f:
            f.write("# Black Hole Raytracer - Integration Method Analysis\n\n")
            
            f.write("## Performance Summary\n\n")
            f.write(performance_df.to_string(index=False))
            f.write("\n\n")
            
            f.write("## Image Analysis\n\n")
            
            for method in methods:
                if method not in results:
                    continue
                    
                image = results[method]['image']
                
                valid_values = image[~np.isnan(image)]
                min_val = np.min(valid_values)
                max_val = np.max(valid_values)
                mean_val = np.mean(valid_values)
                std_val = np.std(valid_values)
                
                f.write(f"### {method}\n")
                f.write(f"- Render Time: {results[method]['render_time']:.2f} seconds\n")
                f.write(f"- Min Redshift: {min_val:.6f}\n")
                f.write(f"- Max Redshift: {max_val:.6f}\n")
                f.write(f"- Mean Redshift: {mean_val:.6f}\n")
                f.write(f"- Std Dev Redshift: {std_val:.6f}\n\n")
            
            f.write("## Comparison with Reference Method\n\n")
            
            ref_method = "Runge-Kutta 4" if "Runge-Kutta 4" in results else methods[0]
            
            f.write(f"Reference Method: {ref_method}\n\n")
            
            for method in methods:
                if method == ref_method or method not in results:
                    continue

                diff = np.abs(results[method]['image'] - results[ref_method]['image'])
                
                mse = mean_squared_error(results[ref_method]['image'].flatten(), 
                                       results[method]['image'].flatten())
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)
                
                f.write(f"### {method} vs {ref_method}\n")
                f.write(f"- Mean Squared Error: {mse:.6f}\n")
                f.write(f"- Maximum Absolute Difference: {max_diff:.6f}\n")
                f.write(f"- Mean Absolute Difference: {mean_diff:.6f}\n\n")
            
            f.write("## Conclusions\n\n")
            
            sorted_methods = sorted(methods, key=lambda m: results[m]['render_time'])
            
            f.write(f"Fastest Method: {sorted_methods[0]} ({results[sorted_methods[0]]['render_time']:.2f}s)\n")
            f.write(f"Slowest Method: {sorted_methods[-1]} ({results[sorted_methods[-1]]['render_time']:.2f}s)\n\n")
            
            # f.write("Method Observations:\n")
            # for method in methods:
            #     if method not in results:
            #         continue
                    
            #     if method == "Runge-Kutta 4":
            #         f.write("- Runge-Kutta 4: Good balance of accuracy and performance. Standard reference method.\n")
            #     elif method == "Adams-Bashforth4":
            #         f.write("- Adams-Bashforth4: Fastest method but lowest accuracy. Good for quick previews.\n")
            #     elif method == "Adams-Moulton4":
            #         f.write("- Adams-Moulton4: Improved stability over Adams-Bashforth, particularly near the event horizon.\n")
            #     elif method == "Bowie":
            #         f.write("- Bowie: Excellent energy conservation with good performance. Optimized for this specific ODE.\n")
            #     elif method == "Obrechkoff":
            #         f.write("- Obrechkoff: Highest accuracy but slowest performance. Best for scientific calculations.\n")
                    
            # f.write("\nSee image comparisons for visual analysis.\n")
            
        return

def mean_squared_error(a, b):
    """
    Calculate mean squared error between two arrays
    
    Args:
        a (ndarray): First array
        b (ndarray): Second array
        
    Returns:
        float: Mean squared error
    """
    return np.mean((a - b) ** 2)

def calculate_radar_chart_area(angles, values):
    """
    Calculate the area of a polygon in radar chart format.
    
    Args:
        angles: Array of angles in radians including the repeated first angle at the end
        values: Array of values including the repeated first value at the end
    
    Returns:
        The area of the polygon
    """
    angles = angles[:-1]
    values = values[:-1]
    
    n = len(angles)

    x_coords = values * np.cos(angles)
    y_coords = values * np.sin(angles)

    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += x_coords[i] * y_coords[j]
        area -= x_coords[j] * y_coords[i]

    area = abs(area) / 2.0
    
    return area

def main():
    # analyzer = IntegrationMethodAnalyzer()
    # analyzer = IntegrationMethodAnalyzer(resolution=900, timestep=0.08, maxsteps=500, output_dir="method_comparison")
    # analyzer = IntegrationMethodAnalyzer(resolution=900, timestep=0.08, maxsteps=500, output_dir="method_comparison")
    analyzer = IntegrationMethodAnalyzer(resolution=900, 
                                         distance=4.5, 
                                         inclination=80, 
                                         timestep=0.005, 
                                         maxsteps=5000,
                                         inner_radius=3, 
                                         output_dir="method_comparison_0.005_5000_80_4.5")
    analyzer.run_full_analysis()

if __name__ == "__main__":
    main()