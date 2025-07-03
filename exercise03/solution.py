import cv2
import numpy as np
import os


def read_ply(filepath):
    """
    Reads a PLY file and extracts vertices and faces.
    Assumes triangular faces.
    """
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
        if len(parts) >= 4:
            faces.append([int(p) for p in parts[1:4]])

    return np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.int32)


def draw_mesh_on_top_of_marker(full_path_input_image,
                               full_path_mesh,
                               full_path_output_image):

    marker_image_path = 'marker.png'

    # --- 1. Load Images and Mesh ---
    marker_image = cv2.imread(marker_image_path, cv2.IMREAD_GRAYSCALE)
    if marker_image is None:
        print(f"Error: Marker image not found at '{marker_image_path}'")
        return False

    frame = cv2.imread(full_path_input_image)
    if frame is None:
        print(f"Error: Input image not found at '{full_path_input_image}'")
        return False

    try:
        mesh_vertices, mesh_faces = read_ply(full_path_mesh)
    except FileNotFoundError:
        print(f"Error: Mesh file not found at '{full_path_mesh}'")
        return False

    # --- 2. Feature Detection Setup ---
    h_marker, w_marker = marker_image.shape[:2]
    orb = cv2.ORB_create(nfeatures=1000)
    kp_marker, des_marker = orb.detectAndCompute(marker_image, None)

    if des_marker is None or len(kp_marker) < 10:
        print(
            "Error: Not enough keypoints found for the marker. Use a more complex marker.")
        return False

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # --- 3. Mesh Pre-processing (Normalization and Scaling) ---
    center = (mesh_vertices.max(axis=0) + mesh_vertices.min(axis=0)) / 2
    centered_vertices = mesh_vertices - center
    max_dim = np.max(centered_vertices.max(axis=0) -
                     centered_vertices.min(axis=0))
    if max_dim == 0:
        max_dim = 1  # Avoid division by zero
    normalized_vertices = centered_vertices / max_dim
    max_z = normalized_vertices[:, 2].max()
    normalized_vertices[:, 2] -= max_z

    mesh_scale_factor = min(w_marker, h_marker) * 0.8
    object_mesh_vertices = normalized_vertices * mesh_scale_factor
    object_mesh_vertices[:, 0] += w_marker / 2
    object_mesh_vertices[:, 1] += h_marker / 2

    # --- 4. Camera Intrinsics Setup ---
    h_frame, w_frame = frame.shape[:2]
    focal_length = w_frame
    camera_center_x = w_frame / 2
    camera_center_y = h_frame / 2
    camera_matrix = np.array([
        [focal_length, 0, camera_center_x],
        [0, focal_length, camera_center_y],
        [0, 0, 1]
    ], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)

    # --- 5. Main Processing Logic ---
    final_display_frame = frame.copy()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_frame, des_frame = orb.detectAndCompute(gray_frame, None)

    if des_frame is None or len(kp_frame) == 0:
        print("No features detected in the input image.")
        cv2.imwrite(full_path_output_image, final_display_frame)
        return True  # It "succeeded" by saving the original image.

    matches = bf.match(des_marker, des_frame)
    DISTANCE_THRESHOLD = 35.0
    good_matches = sorted([m for m in matches if m.distance <
                          DISTANCE_THRESHOLD], key=lambda x: x.distance)

    MIN_GOOD_MATCHES_FOR_PNP_ATTEMPT = 12
    MIN_INLIERS_FOR_STABLE_POSE = 8

    pose_found = False
    if len(good_matches) >= MIN_GOOD_MATCHES_FOR_PNP_ATTEMPT:
        obj_pts_marker_matched = np.float32(
            [kp_marker[m.queryIdx].pt for m in good_matches])
        obj_pts_marker_3d = np.hstack(
            (obj_pts_marker_matched, np.zeros((len(good_matches), 1), dtype=np.float32)))
        img_pts_frame_matched = np.float32(
            [kp_frame[m.trainIdx].pt for m in good_matches])

        try:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                obj_pts_marker_3d, img_pts_frame_matched, camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE, reprojectionError=8.0
            )

            if success and inliers is not None and len(inliers) >= MIN_INLIERS_FOR_STABLE_POSE:
                pose_found = True
                projected_mesh_pts_2d, _ = cv2.projectPoints(
                    object_mesh_vertices, rvec, tvec, camera_matrix, dist_coeffs)
                projected_mesh_pts_2d = np.int32(
                    projected_mesh_pts_2d.reshape(-1, 2))

                R_matrix, _ = cv2.Rodrigues(rvec)
                mesh_vertices_camera_space = (
                    R_matrix @ object_mesh_vertices.T).T + tvec.T

                face_depths = []
                for i, face_indices_in_mesh in enumerate(mesh_faces):
                    vertices_of_face_cam_space = mesh_vertices_camera_space[face_indices_in_mesh]
                    avg_z = np.mean(vertices_of_face_cam_space[:, 2])
                    face_depths.append((avg_z, i))

                face_depths.sort(key=lambda x: x[0], reverse=True)

                for _, original_face_idx in face_depths:
                    points_for_poly = projected_mesh_pts_2d[mesh_faces[original_face_idx]]
                    # A single color for the whole mesh
                    color = (150, 150, 255)
                    cv2.fillPoly(final_display_frame, [points_for_poly], color)
                    cv2.polylines(final_display_frame, [
                                  points_for_poly], True, (30, 30, 30), 1)
        except cv2.error:
            pass  # PnP can sometimes fail with insufficient points

    if not pose_found:
        print("Marker pose could not be estimated. Saving original image.")

    # --- 6. Save the final image ---
    cv2.imwrite(full_path_output_image, final_display_frame)
    print(f"Output image saved to '{full_path_output_image}'")
    return True


if __name__ == '__main__':
    # =================== EXAMPLE USAGE ===================
    # To run this example, you need three files in the same directory:
    # 1. This python script.
    # 2. 'marker.png': A QR code or other complex image to act as the marker.
    # 3. 'input_image.jpg': An image that contains the marker.
    # 4. 'your_mesh.ply': A 3D mesh file in PLY format.

    # Define file paths
    input_image_file = 'input_image.jpg'
    mesh_file = 'your_mesh.ply'
    output_image_file = 'output_with_mesh.jpg'
    marker_file = 'marker.png'

    # Create dummy files if they don't exist for demonstration
    if not os.path.exists(marker_file):
        print(f"Creating a dummy '{marker_file}'...")
        dummy_marker = np.zeros((200, 200), dtype=np.uint8)
        dummy_marker[50:150, 50:150] = 255
        cv2.putText(dummy_marker, "MARKER", (55, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2)
        cv2.imwrite(marker_file, dummy_marker)

    if not os.path.exists(input_image_file):
        print(f"Creating a dummy '{input_image_file}'...")
        dummy_input = np.full((720, 1280, 3), (100, 100, 100), dtype=np.uint8)
        marker = cv2.imread(marker_file)
        h, w, _ = marker.shape
        # Place the marker in the dummy input image
        x_offset, y_offset = 500, 250
        dummy_input[y_offset:y_offset+h, x_offset:x_offset+w] = marker
        cv2.imwrite(input_image_file, dummy_input)

    if not os.path.exists(mesh_file):
        print(f"Creating a dummy '{mesh_file}' (a simple cube)...")
        ply_content = """ply
format ascii 1.0
element vertex 8
property float x
property float y
property float z
element face 6
property list uchar int vertex_indices
end_header
1 1 1
-1 1 1
-1 -1 1
1 -1 1
1 1 -1
-1 1 -1
-1 -1 -1
1 -1 -1
4 0 1 2 3
4 7 6 5 4
4 0 4 5 1
4 1 5 6 2
4 2 6 7 3
4 3 7 4 0
"""
        with open(mesh_file, 'w') as f:
            f.write(ply_content)

    # Call the main function
    success = draw_mesh_on_top_of_marker(
        full_path_input_image=input_image_file,
        full_path_mesh=mesh_file,
        full_path_output_image=output_image_file
    )

    if success:
        print("\nFunction executed successfully.")
        # Optionally open the image
        # if os.name == 'nt': # For Windows
        #     os.startfile(output_image_file)
        # elif os.name == 'posix': # For macOS/Linux
        #     import subprocess
        #     subprocess.call(('open', output_image_file) if sys.platform == 'darwin' else ('xdg-open', output_image_file))
    else:
        print("\nFunction encountered an error.")
