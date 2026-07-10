"""
Adaptive CUDA kernels for black hole ray tracing.
"""
import math
import numpy as np
import operator
from numba import cuda

from .physics import make_trace_disk_redshift_adaptive, make_trace_plane_intersection_adaptive
from .integrators import (
    pvcalc, rk4_step, euler_step, adams_bashforth_step,
    adams_bashforth4_step, adams_moulton4_step, bowie_integrator, obrechkoff_integrator
)


def make_disk_image_kernel_adaptive(integrator_step):
    """Factory function that creates a specialized adaptive disk image kernel"""
    trace_function = make_trace_disk_redshift_adaptive(integrator_step)

    @cuda.jit(max_registers=64)
    def create_disk_image_adaptive(
        image, y_t, f_t, y_temp, k1, k2, k3, k4, h, max_iter, pos, vel,
        inv_view_matrix, DISKINNERSQR, DISKOUTERSQR, DIST_CAM, SIN_I,
        ray_dir_screen, ray_dir_world, image_plane_width, image_plane_height,
        camera_pos_x, camera_pos_y, camera_pos_z, sin_psi,
        min_h_factor, max_h_factor, adapt_threshold
    ):
        width = image.shape[0]
        height = image.shape[1]
        x, y = cuda.grid(2)
        if x < width and y < height:
            idx = y * width + x
            obs_x = math.floor(x - width / 2)
            obs_y = math.floor(-(y - height / 2))
            sinalpha = math.sin(math.atan2(obs_y, obs_x))

            pos[idx][0] = camera_pos_x
            pos[idx][1] = camera_pos_y
            pos[idx][2] = camera_pos_z

            ray_dir_screen[idx][0] = (x / width - 0.5) * image_plane_width
            ray_dir_screen[idx][1] = (0.5 - y / height) * image_plane_height
            ray_dir_screen[idx][2] = -1
            ray_dir_screen[idx][3] = 1
            ray_dir_world[idx][0] = 0.0
            ray_dir_world[idx][1] = 0.0
            ray_dir_world[idx][2] = 0.0
            ray_dir_world[idx][3] = 0.0

            for i in range(len(inv_view_matrix)):
                for j in range(len(ray_dir_screen[idx])):
                    ray_dir_world[idx][i] += inv_view_matrix[i][j] * ray_dir_screen[idx][j]
            norm = math.sqrt(
                ray_dir_world[idx][0] ** 2
                + ray_dir_world[idx][1] ** 2
                + ray_dir_world[idx][2] ** 2
            )
            vel[idx][0] = ray_dir_world[idx][0] / norm
            vel[idx][1] = ray_dir_world[idx][1] / norm
            vel[idx][2] = ray_dir_world[idx][2] / norm

            b = pos[idx][2] * vel[idx][0] - pos[idx][0] * vel[idx][2]

            h2 = (
                math.pow(pos[idx][1] * vel[idx][2] - pos[idx][2] * vel[idx][1], 2)
                + math.pow(pos[idx][2] * vel[idx][0] - pos[idx][0] * vel[idx][2], 2)
                + math.pow(pos[idx][0] * vel[idx][1] - pos[idx][1] * vel[idx][0], 2)
            )

            min_timestep = h * min_h_factor
            max_timestep = h * max_h_factor

            color = trace_function(
                y_t[idx], f_t[idx], y_temp[idx], pos[idx], vel[idx],
                k1[idx], k2[idx], k3[idx], k4[idx], h2, h, max_iter,
                DISKINNERSQR, DISKOUTERSQR, DIST_CAM, SIN_I, sinalpha, idx, b, sin_psi,
                min_timestep, max_timestep, adapt_threshold
            )

            image[y, x] = color

    return create_disk_image_adaptive


def make_image_coords_kernel_adaptive(integrator_step):
    """Factory function that creates a specialized adaptive image coordinates kernel"""
    trace_function = make_trace_plane_intersection_adaptive(integrator_step)

    @cuda.jit(fastmath=True, max_registers=64)
    def create_image_coords_adaptive(
        pixel_coords, y_t, f_t, y_temp, k1, k2, k3, k4, h, max_iter, pos, vel,
        inv_view_matrix, view_matrix, ray_screen, ray_world,
        image_plane_width, image_plane_height, camera_pos_x, camera_pos_y, camera_pos_z,
        inter_point, checked,
        min_h_factor, max_h_factor, adapt_threshold
    ):
        width = pixel_coords.shape[0]
        height = pixel_coords.shape[1]
        x, y = cuda.grid(2)
        if x < width and y < height:
            idx = y * width + x

            pos[idx][0] = camera_pos_x
            pos[idx][1] = camera_pos_y
            pos[idx][2] = camera_pos_z

            ray_screen[idx][0] = (x / width - 0.5) * image_plane_width
            ray_screen[idx][1] = (0.5 - y / height) * image_plane_height
            ray_screen[idx][2] = -1
            ray_screen[idx][3] = 1
            ray_world[idx][0] = 0.0
            ray_world[idx][1] = 0.0
            ray_world[idx][2] = 0.0
            ray_world[idx][3] = 0.0

            for i in range(len(inv_view_matrix)):
                for j in range(len(ray_screen[idx])):
                    ray_world[idx][i] += inv_view_matrix[i][j] * ray_screen[idx][j]
            norm = math.sqrt(
                ray_world[idx][0] ** 2 + ray_world[idx][1] ** 2 + ray_world[idx][2] ** 2
            )
            vel[idx][0] = ray_world[idx][0] / norm
            vel[idx][1] = ray_world[idx][1] / norm
            vel[idx][2] = ray_world[idx][2] / norm

            h2 = (
                math.pow(pos[idx][1] * vel[idx][2] - pos[idx][2] * vel[idx][1], 2)
                + math.pow(pos[idx][2] * vel[idx][0] - pos[idx][0] * vel[idx][2], 2)
                + math.pow(pos[idx][0] * vel[idx][1] - pos[idx][1] * vel[idx][0], 2)
            )

            trace_function(
                y_t[idx], f_t[idx], y_temp[idx], pos[idx], vel[idx],
                k1[idx], k2[idx], k3[idx], k4[idx], h2, h,
                max_iter, inter_point[idx], checked,
                min_h_factor, max_h_factor, adapt_threshold
            )

            if (
                inter_point[idx][0] == 0.0
                and inter_point[idx][1] == 0.0
                and inter_point[idx][2] == 0.0
            ):
                pixel_coords[y, x][0] = 0
                pixel_coords[y, x][1] = 0
            else:
                ray_world[idx][0] = inter_point[idx][0]
                ray_world[idx][1] = inter_point[idx][1]
                ray_world[idx][2] = inter_point[idx][2]

                ray_world[idx][3] = 1.0
                ray_screen[idx][0] = 0.0
                ray_screen[idx][1] = 0.0
                ray_screen[idx][2] = 0.0
                ray_screen[idx][3] = 0.0

                for i in range(len(view_matrix)):
                    for j in range(len(ray_world[idx])):
                        ray_screen[idx][i] += view_matrix[i][j] * ray_world[idx][j]

                x_2d = ray_screen[idx][0] / ray_screen[idx][2]
                y_2d = ray_screen[idx][1] / ray_screen[idx][2]
                pixel_x = (x_2d / image_plane_width + 0.5) * width
                pixel_y = (0.5 - y_2d / image_plane_height) * height

                pixel_x = int(round(pixel_x))
                pixel_y = int(round(pixel_y))

                if 0 < pixel_x < width and 0 < pixel_y < height:
                    pixel_coords[y, x][0] = pixel_x
                    pixel_coords[y, x][1] = pixel_y

    return create_image_coords_adaptive


def make_path_kernel_adaptive(integrator_step):
    """Factory function that creates a specialized adaptive path kernel"""

    @cuda.jit(max_registers=64)
    def create_paths_adaptive(
        y_t, f_t, y_temp, k1, k2, k3, k4, h, max_iter, pos, vel, inv_view_matrix,
        DISKINNERSQR, DISKOUTERSQR, DIST_CAM, SIN_I, ray_dir_screen, ray_dir_world,
        image_plane_width, image_plane_height, camera_pos_x, camera_pos_y, camera_pos_z,
        history, samplerate,
        min_h_factor, max_h_factor, adapt_threshold
    ):
        width, height = cuda.gridsize(2)
        x, y = cuda.grid(2)
        samplerate = int(samplerate)

        if x < width and y < height and x % samplerate == 0 and y % samplerate == 0:
            idx = int((y // samplerate) * math.ceil(width / samplerate) + (x // samplerate))

            pos[idx][0] = camera_pos_x
            pos[idx][1] = camera_pos_y
            pos[idx][2] = camera_pos_z

            ray_dir_screen[idx][0] = (x / width - 0.5) * image_plane_width
            ray_dir_screen[idx][1] = (0.5 - y / height) * image_plane_height
            ray_dir_screen[idx][2] = -1
            ray_dir_screen[idx][3] = 1
            ray_dir_world[idx][0] = 0.0
            ray_dir_world[idx][1] = 0.0
            ray_dir_world[idx][2] = 0.0
            ray_dir_world[idx][3] = 0.0

            for i in range(len(inv_view_matrix)):
                for j in range(len(ray_dir_screen[idx])):
                    ray_dir_world[idx][i] += inv_view_matrix[i][j] * ray_dir_screen[idx][j]
            norm = math.sqrt(
                ray_dir_world[idx][0] ** 2
                + ray_dir_world[idx][1] ** 2
                + ray_dir_world[idx][2] ** 2
            )
            vel[idx][0] = ray_dir_world[idx][0] / norm
            vel[idx][1] = ray_dir_world[idx][1] / norm
            vel[idx][2] = ray_dir_world[idx][2] / norm

            h2 = (
                math.pow(pos[idx][1] * vel[idx][2] - pos[idx][2] * vel[idx][1], 2)
                + math.pow(pos[idx][2] * vel[idx][0] - pos[idx][0] * vel[idx][2], 2)
                + math.pow(pos[idx][0] * vel[idx][1] - pos[idx][1] * vel[idx][0], 2)
            )

            schwarzschild_radius = 1.0
            min_timestep = h * min_h_factor
            max_timestep = h * max_h_factor
            current_h = h
            r_previous = math.sqrt(
                pos[idx][0] * pos[idx][0]
                + pos[idx][1] * pos[idx][1]
                + pos[idx][2] * pos[idx][2]
            )

            was_in_disc = False
            for step in range(max_iter):
                oldx = y_temp[idx][0]
                oldy = y_temp[idx][1]
                oldz = y_temp[idx][2]

                r_current = r_previous
                proximity_factor = min(
                    1.0,
                    (r_current - schwarzschild_radius) / (5 * schwarzschild_radius)
                )
                if proximity_factor < adapt_threshold:
                    current_h = max(min_timestep, h * proximity_factor)
                else:
                    current_h = min(max_timestep, current_h * (1.0 + adapt_threshold))

                integrator_step(
                    y_t[idx], f_t[idx], y_temp[idx], pos[idx], vel[idx],
                    k1[idx], k2[idx], k3[idx], k4[idx], h2, current_h
                )

                history[idx, step, 0] = pos[idx][0]
                history[idx, step, 1] = -pos[idx][1]
                history[idx, step, 2] = pos[idx][2]
                pointsqr = (
                    pos[idx][0] * pos[idx][0]
                    + pos[idx][1] * pos[idx][1]
                    + pos[idx][2] * pos[idx][2]
                )
                oldx = y_temp[idx][0]
                oldy = y_temp[idx][1]
                oldz = y_temp[idx][2]
                crossed = operator.xor(oldy > 0.0, pos[idx][1] > 0.0)
                is_in_disc = operator.and_(
                    pointsqr < DISKOUTERSQR, pointsqr > DISKINNERSQR
                )

                if pointsqr > 1.4 * DIST_CAM:
                    break
                if crossed and is_in_disc:
                    was_in_disc = True

                r_previous = math.sqrt(pointsqr)

            if not was_in_disc:
                for step in range(max_iter):
                    history[idx, step, 0] = 0.0
                    history[idx, step, 1] = 0.0
                    history[idx, step, 2] = 0.0

    return create_paths_adaptive


INTEGRATOR_FUNCS = {
    "Runge-Kutta 4": rk4_step,
    "Euler": euler_step,
    "Adams-Bashforth": adams_bashforth_step,
    "Adams-Bashforth4": adams_bashforth4_step,
    "Adams-Moulton4": adams_moulton4_step,
    "Bowie": bowie_integrator,
    "Obrechkoff": obrechkoff_integrator,
}

DISK_IMAGE_KERNELS_ADAPTIVE = {
    name: make_disk_image_kernel_adaptive(func)
    for name, func in INTEGRATOR_FUNCS.items()
}

IMAGE_COORDS_KERNELS_ADAPTIVE = {
    name: make_image_coords_kernel_adaptive(func)
    for name, func in INTEGRATOR_FUNCS.items()
}

PATH_KERNELS_ADAPTIVE = {
    name: make_path_kernel_adaptive(func)
    for name, func in INTEGRATOR_FUNCS.items()
}
