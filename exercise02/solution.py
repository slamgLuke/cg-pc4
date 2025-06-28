import math
import numpy as np
from PIL import Image


def read_ply(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    num_vertices = 0
    num_faces = 0
    header_end_index = 0

    for i, line in enumerate(lines):
        if line.startswith('element vertex'):
            num_vertices = int(line.split()[2])
        elif line.startswith('element face'):
            num_faces = int(line.split()[2])
        elif line.startswith('end_header'):
            header_end_index = i + 1
            break

    vertices = []
    for i in range(header_end_index, header_end_index + num_vertices):
        parts = lines[i].split()
        vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])

    faces = []
    for i in range(header_end_index + num_vertices, header_end_index + num_vertices + num_faces):
        parts = lines[i].split()
        faces.append([int(parts[1]), int(parts[2]), int(parts[3])])

    return np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.int32)


def sequence_of_projections(
    full_path_input_mesh,
    optical_center_x_array,
    optical_center_y_array,
    optical_center_z_array,
    optical_axis_x_array,
    optical_axis_y_array,
    optical_axis_z_array,
    up_vector_x_array,
    up_vector_y_array,
    up_vector_z_array,
    focal_distance_array,
    output_width_in_pixels,
    output_height_in_pixels,
    prefix_output_files
):
    world_coords = read_ply(full_path_input_mesh)
    i = 1

    for optical_center_x, optical_center_y, optical_center_z, optical_axis_x, optical_axis_y, optical_axis_z, up_vector_x, up_vector_y, up_vector_z, focal_distance in zip(
        optical_center_x_array,
        optical_center_y_array,
        optical_center_z_array,
        optical_axis_x_array,
        optical_axis_y_array,
        optical_axis_z_array,
        up_vector_x_array,
        up_vector_y_array,
        up_vector_z_array,
        focal_distance_array
    ):
        optical_center = np.array(
            [optical_center_x, optical_center_y, optical_center_z])
        optical_axis = np.array(
            [optical_axis_x, optical_axis_y, optical_axis_z])
        up_vector = np.array([up_vector_x, up_vector_y, up_vector_z])

        z_cam = -optical_axis / np.linalg.norm(optical_axis)
        x_cam = np.cross(up_vector, z_cam)
        x_cam /= np.linalg.norm(x_cam)
        y_cam = np.cross(z_cam, x_cam)

        R = np.stack([x_cam, y_cam, z_cam], axis=0)

        translated_coords = world_coords - optical_center
        camera_coords = translated_coords @ R.T

        in_front_of_camera = camera_coords[:, 2] < 0

        x = camera_coords[in_front_of_camera, 0]
        y = camera_coords[in_front_of_camera, 1]
        z = camera_coords[in_front_of_camera, 2]

        projected_x = -focal_distance * (x / z)
        projected_y = -focal_distance * (y / z)

        pixel_x = projected_x + output_width_in_pixels / 2
        pixel_y = -projected_y + output_height_in_pixels / 2

        is_in_image_x = (pixel_x >= 0) & (pixel_x < output_width_in_pixels)
        is_in_image_y = (pixel_y >= 0) & (pixel_y < output_height_in_pixels)
        is_in_image = is_in_image_x & is_in_image_y

        final_pixel_x = pixel_x[is_in_image].astype(int)
        final_pixel_y = pixel_y[is_in_image].astype(int)

        image_array = np.zeros(
            (output_height_in_pixels, output_width_in_pixels, 3), dtype=np.uint8)
        image_array[final_pixel_y, final_pixel_x] = [255, 255, 255]

        img = Image.fromarray(image_array, 'RGB')
        img.save(prefix_output_files + '-%d.png' % i)

        i += 1
