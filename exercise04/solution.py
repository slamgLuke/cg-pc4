import cv2
import os


def stitch_images(
    full_path_input_images,
    blender_type='MultiBandBlender',
    features_finder_type='SIFT',
    features_matcher_type='BestOf2NearestRange',
    warper_type='Mercator',
    full_path_output_image='panorama_output.jpg'
):

    imgs = []
    for img_path in full_path_input_images:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}. Skipping.")
            continue
        imgs.append(img)

    if not imgs:
        print("Error: No images to stitch.")
        return

    # Initialize the stitcher
    stitcher = cv2.Stitcher_create()

    # --- Feature Finder Configuration ---
    # The official documentation and source code reveal that specific feature finders
    # can be set for the stitching pipeline. [8, 10]
    if features_finder_type == 'SIFT':
        features_finder = cv2.SIFT_create()
    elif features_finder_type == 'ORB':
        features_finder = cv2.ORB_create()
    elif features_finder_type == 'AKAZE':
        features_finder = cv2.AKAZE_create()
    # SURF is in opencv-contrib-python, check if available
    elif features_finder_type == 'SURF' and hasattr(cv2, 'xfeatures2d'):
        features_finder = cv2.xfeatures2d.SURF_create()
    else:
        print(
            f"Warning: {features_finder_type} not supported or available. Using default.")
        # Default is ORB in many versions
        features_finder = cv2.ORB_create()

    stitcher.setFeaturesFinder(features_finder)

    # --- Warper Configuration ---
    # The warper can be set to different projection types. [2, 8]
    warper = cv2.PyRotationWarper(warper_type, 1)
    stitcher.setWarper(warper)

    # --- Blender Configuration ---
    # The blending method can be specified to smooth the seams. [10]
    if blender_type == 'MultiBandBlender':
        blender = cv2.detail_MultiBandBlender()
    elif blender_type == 'FeatherBlender':
        blender = cv2.detail_FeatherBlender()
    else:
        # Default to MultiBand for better results
        blender = cv2.detail_MultiBandBlender()

    blender.setNumBands((cv2.getTickCount() / cv2.getTickFrequency()) * 5 - 1)
    stitcher.setBlender(blender)

    # --- Feature Matcher Information ---
    # As per the user request, it is important to note that while the C++ API of OpenCV's
    # stitcher class provides a `setFeaturesMatcher` method, this is not directly exposed
    # or easily configurable in the high-level `cv2.Stitcher` Python class. [2, 8, 12]
    # The stitcher automatically selects a suitable matcher based on the chosen mode (e.g., panorama or scans).
    # The default for panorama mode is typically a variation of `BestOf2NearestMatcher`. [7]
    if features_matcher_type not in ['AffineBestOf2Nearest', 'BestOf2NearestRange']:
        print(
            f"Note: The feature matcher '{features_matcher_type}' is not a standard option and direct setting is ignored in this implementation.")

    # Perform stitching
    status, pano = stitcher.stitch(imgs)

    if status == cv2.Stitcher_OK:
        print("Panorama generated successfully.")
        cv2.imwrite(full_path_output_image, pano)
    else:
        error_messages = {
            1: "ERR_NEED_MORE_IMGS: Not enough images to stitch.",
            2: "ERR_HOMOGRAPHY_EST_FAIL: Homography estimation failed.",
            3: "ERR_CAMERA_PARAMS_ADJUST_FAIL: Camera parameter adjustment failed."
        }
        print(
            f"Error during stitching: {error_messages.get(status, 'Unknown error')}")


# --- Example Usage ---
# Create dummy directories and images for a runnable example.
if not os.path.exists('test_images'):
    os.makedirs('test_images')
# Assume dummy image files exist at these paths for the example to run.
# For a real use case, replace these with actual image paths.
dummy_image_paths = [
    'test_images/panorama_input_1.jpg',
    'test_images/panorama_input_2.jpg',
    'test_images/panorama_input_3.jpg',
]

# Create dummy blank images for demonstration if they don't exist
for path in dummy_image_paths:
    if not os.path.exists(path):
        import numpy as np
        # Create a simple gradient image to simulate distinct features
        img = np.zeros((400, 600, 3), dtype=np.uint8)
        # Add a unique feature to each image
        unique_pos = (dummy_image_paths.index(path) + 1) * 150
        cv2.rectangle(img, (unique_pos - 50, 150),
                      (unique_pos + 50, 250), (255, 255, 255), -1)
        cv2.imwrite(path, img)

print("Starting example stitch...")
stitch_images(
    full_path_input_images=dummy_image_paths,
    blender_type='MultiBandBlender',
    features_finder_type='ORB',  # SIFT or SURF might require opencv-contrib-python
    # Note on this parameter is included in the function
    features_matcher_type='BestOf2NearestRange',
    warper_type='Spherical',
    full_path_output_image='panorama_spherical_output.jpg'
)
