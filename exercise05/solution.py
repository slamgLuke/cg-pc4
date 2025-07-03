from ultralytics import YOLO
import os


def count_people_cars_and_bikes(full_path_input_video: str) -> list[int]:
    model = YOLO('yolov8n.pt')

    target_classes = [0, 1, 2]  # person, bicycle, car

    # persist=True maintains tracking between frames
    results = model.track(
        source=full_path_input_video,
        persist=True,
        verbose=False,
        imgsz=640,
        classes=target_classes
    )

    person_ids = set()
    bike_ids = set()
    car_ids = set()

    for r in results:
        if r.boxes.id is not None:
            boxes_cls = r.boxes.cls.int().cpu().tolist()
            boxes_id = r.boxes.id.int().cpu().tolist()

            for cls, track_id in zip(boxes_cls, boxes_id):
                if cls == 0:
                    person_ids.add(track_id)
                elif cls == 1:
                    bike_ids.add(track_id)
                elif cls == 2:
                    car_ids.add(track_id)

    total_people = len(person_ids)
    total_bikes = len(bike_ids)
    total_cars = len(car_ids)

    return [total_people, total_bikes, total_cars]


# --- Example Usage ---
if __name__ == '__main__':
    # Create a dummy video file for demonstration if it doesn't exist.
    # In a real scenario, you would replace this with your actual video path.
    video_file_name = "sample_video.mp4"
    if not os.path.exists(video_file_name):
        print(f"'{video_file_name}' not found.")
        print("Please replace this with the path to your video file.")
        # As a placeholder, we create an empty file.
        # The function will gracefully handle it (and return [0, 0, 0]).
        open(video_file_name, 'a').close()
        video_path = video_file_name
    else:
        video_path = video_file_name

    print(f"Processing video: {video_path}")

    # Ensure the video path exists before processing
    if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
        try:
            counts = count_people_cars_and_bikes(video_path)
            print("\n--- Detection Results ---")
            print(f"Total Unique People: {counts[0]}")
            print(f"Total Unique Bikes:  {counts[1]}")
            print(f"Total Unique Cars:   {counts[2]}")
            print(f"\nReturned list: {counts}")
        except Exception as e:
            print(f"An error occurred during video processing: {e}")
            print(
                "Please ensure you have a valid video file and all dependencies are installed.")
    else:
        print("The specified video path is either invalid or the file is empty.")
