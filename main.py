# webcam_detection.py - Live object detection from webcam!

import cv2
from ultralytics import YOLO

# Load model
model = YOLO("yolo11n.pt")

print("📹 Starting detection...")
print("Press 'q' to quit")

# Setup window allows resizing
window_name = "YOLO Webcam Detection"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1280, 720)

# Initialize log file settings
log_file = "detection_log.txt"
target_file = "target_classes.txt"

# Load target classes
target_classes = set()
try:
    with open(target_file, "r") as f:
        # Read lines, strip whitespace, and ignore empty lines
        target_classes = {line.strip() for line in f if line.strip()}
    print(f"🎯 Target classes loaded: {', '.join(target_classes)}")
except FileNotFoundError:
    print(f"⚠️ Warning: {target_file} not found. All objects will be detected.")

# Clear the log file at the start of the program
with open(log_file, "w") as f:
    f.write(f"{'Object':<15} {'Max Conf':<10} {'Min Conf':<10}\n")
    f.write("-" * 35 + "\n")

# Dictionary to store confidence stats: {class_name: {'max': float, 'min': float}}
object_stats = {}

while True:
    # Use source="example.mp4" for video file
    results = model.predict(
        source="example.mp4",
        show=False,  # Disable built-in display to handle key events manually
        stream=True,  # Real-time streaming
        verbose=False,  # Less terminal output
    )

    stop_program = False
    # Process live stream
    if results is None:
        break
    for r in results:
        # Filter detections based on target_classes
        if target_classes:
            # Create a new boxes object with only filtered detections
            # We iterate through original boxes and keep indices of matching classes
            keep_indices = []
            for i, box in enumerate(r.boxes):
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                if class_name in target_classes:
                    keep_indices.append(i)

            # Update the result's boxes to only contain target objects
            if keep_indices:
                r.boxes = r.boxes[keep_indices]
            else:
                r.boxes = r.boxes[[]]  # Empty boxes if no matches

        # Log detected objects
        for box in r.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])

            # Update stats if object seen before, else initialize
            if class_name in object_stats:
                stats = object_stats[class_name]
                if confidence > stats["max"]:
                    stats["max"] = confidence
                if confidence < stats["min"]:
                    stats["min"] = confidence
            else:
                object_stats[class_name] = {"max": confidence, "min": confidence}

            # Rewrite the entire log file with updated stats
            # (Note: For very high performance requirements, writing to file every frame
            # might be slow, but for typical webcam usage it's fine and ensures data safety)
            with open(log_file, "w") as f:
                f.write(f"{'Object':<15} {'Max Conf':<10} {'Min Conf':<10}\n")
                f.write("-" * 35 + "\n")
                for name, stats in object_stats.items():
                    f.write(f"{name:<15} {stats['max']:.2f}       {stats['min']:.2f}\n")

        # Display the frame
        im_array = r.plot()  # Get the image with bounding boxes
        cv2.imshow(window_name, im_array)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop_program = True
            break

    # If user pressed 'q', break the outer loop (stop repeating)
    if stop_program:
        break

    # If user pressed 'q', break the outer loop (stop repeating)
    if stop_program:
        break

print("👋 Detection stopped")
cv2.destroyAllWindows()