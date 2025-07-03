import os
from PIL import Image, ImageOps
import numpy as np
from ultralytics import YOLO, SAM


def apply_color_scale(image: Image.Image, scale: tuple[int, int, int]) -> Image.Image:
    grayscale_image = image.convert('L')
    return ImageOps.colorize(grayscale_image, black=(0, 0, 0), white=scale)


def highlight_people_cars_and_bikes(
    full_path_input_image: str,
    color_scale_image: tuple[int, int, int],
    color_scale_people: tuple[int, int, int],
    color_scale_cars: tuple[int, int, int],
    color_scale_bikes: tuple[int, int, int],
    full_path_output_image: str
):
    yolo_model = YOLO('yolov8n.pt')
    sam_model = SAM('sam_b.pt')

    original_image = Image.open(full_path_input_image).convert('RGB')

    target_classes = [0, 1, 2]
    yolo_results = yolo_model(
        original_image, classes=target_classes, verbose=False)

    if not yolo_results or not yolo_results[0].boxes.xyxy.numel():
        print("No people, cars, or bikes found. Applying background color scale only.")
        output_image = apply_color_scale(original_image, color_scale_image)
        output_image.save(full_path_output_image)
        return

    sam_results = sam_model(
        original_image, bboxes=yolo_results[0].boxes, verbose=False)

    background_img = apply_color_scale(original_image, color_scale_image)
    people_img = apply_color_scale(original_image, color_scale_people)
    cars_img = apply_color_scale(original_image, color_scale_cars)
    bikes_img = apply_color_scale(original_image, color_scale_bikes)

    output_image = background_img.copy()
    masks = sam_results[0].masks.data
    class_ids = yolo_results[0].boxes.cls.int().cpu().tolist()

    for i, mask_tensor in enumerate(masks):
        class_id = class_ids[i]

        if class_id == 0:  # person
            source_color_img = people_img
        elif class_id == 1:  # bicycle
            source_color_img = bikes_img
        elif class_id == 2:  # car
            source_color_img = cars_img
        else:
            continue

        mask_pil = Image.fromarray(
            (mask_tensor.cpu().numpy() * 255).astype(np.uint8), mode='L')
        output_image.paste(source_color_img, (0, 0), mask_pil)

    output_image.save(full_path_output_image)
    print(
        f"Successfully created highlighted image at: {full_path_output_image}")


# --- Example Usage ---
if __name__ == '__main__':
    input_image_path = "zidane.jpg"

    if not os.path.exists(input_image_path):
        print(f"'{input_image_path}' not found.")
        print("Please replace this path with the path to your own image.")
    else:
        output_image_path = "highlighted_rgb_output.jpg"

        print("Processing image with RGB color tints...")
        highlight_people_cars_and_bikes(
            full_path_input_image=input_image_path,
            # Background in grayscale
            color_scale_image=(255, 255, 255),
            # People tinted with a bright blue
            color_scale_people=(0, 150, 255),
            # Cars tinted with a strong red
            color_scale_cars=(255, 50, 50),
            # Bikes tinted with a vibrant green
            color_scale_bikes=(50, 255, 50),
            full_path_output_image=output_image_path
        )

        # To view the result, open 'highlighted_rgb_output.jpg'
        try:
            Image.open(output_image_path).show()
        except Exception:
            print(
                f"Image saved to {output_image_path}. Open it to see the result.")
