# webcam_detection.py - Live object detection from webcam!

import cv2
from ultralytics import YOLO

# Load model
model = YOLO("yolo11n.pt")

print("📹 Starting detection...")
print("Press 'q' to quit")

# Setup window allows resizing
window_name = "YOLO Webcam Detection"

# Global state for object tracking
tracking_state = {"boxes": [], "tracked_id": None}


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if param["tracked_id"] is not None:
            # Cancel tracking if clicked anywhere while tracking
            param["tracked_id"] = None
            print("🚫 Stopped tracking")
        else:
            # Check if clicked on a box to start tracking
            for box in param["boxes"]:
                x1, y1, x2, y2, obj_id = box
                if x1 <= x <= x2 and y1 <= y <= y2:
                    param["tracked_id"] = obj_id
                    print(f"🎯 Started tracking Object ID: {obj_id}")
                    break


cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(window_name, mouse_callback, tracking_state)
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
    results = model.track(
        source="example.mp4",
        show=False,  # Disable built-in display to handle key events manually
        stream=True,  # Real-time streaming
        verbose=False,  # Less terminal output
        persist=True,  # Persist tracking IDs between frames
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

        # If tracking a specific object, filter out everything else
        if tracking_state["tracked_id"] is not None:
            if r.boxes.id is not None:
                # Get all IDs currently detected
                detected_ids = r.boxes.id.cpu().numpy().astype(int)
                # Find the index of the tracked object
                match_indices = [
                    i
                    for i, mid in enumerate(detected_ids)
                    if mid == tracking_state["tracked_id"]
                ]

                if match_indices:
                    r.boxes = r.boxes[match_indices]
                else:
                    r.boxes = r.boxes[[]]  # Tracked object lost
            else:
                r.boxes = r.boxes[[]]

        # Update tracking state with current frame's boxes
        current_boxes = []
        if r.boxes.id is not None:
            boxes_coord = r.boxes.xyxy.cpu().numpy()
            boxes_ids = r.boxes.id.cpu().numpy()
            for box, box_id in zip(boxes_coord, boxes_ids):
                x1, y1, x2, y2 = box
                current_boxes.append((x1, y1, x2, y2, int(box_id)))
        tracking_state["boxes"] = current_boxes

        # Output tracked object position
        if tracking_state["tracked_id"] is not None and current_boxes:
            # Since we filtered r.boxes to only contain the tracked object (if found),
            # and current_boxes is built from r.boxes, current_boxes should only contain
            # the tracked object if it was successfully tracked.
            # We can also double check the ID to be safe.
            tracked_box = next(
                (b for b in current_boxes if b[4] == tracking_state["tracked_id"]), None
            )

            if tracked_box:
                x1, y1, x2, y2, _ = tracked_box
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                h, w = r.orig_img.shape[:2]
                dx = cx - (w / 2)
                dy = cy - (h / 2)
                print(f"📍 Tracked Object Pos: ({dx:.1f}, {dy:.1f})")

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

        # Draw UI overlay
        height, width = im_array.shape[:2]
        center_x, center_y = width // 2, height // 2

        # Draw center crosshair (concentric circles + cross)
        # Outer circle (Radius 20 -> 100, Thickness 1 -> 2)
        cv2.circle(im_array, (center_x, center_y), 100, (0, 255, 0), 2)
        # Inner circle (Radius 5 -> 25)
        cv2.circle(im_array, (center_x, center_y), 25, (0, 255, 0), -1)
        # Cross lines (Length 30 -> 150, Thickness 1 -> 2)
        cv2.line(
            im_array,
            (center_x - 150, center_y),
            (center_x + 150, center_y),
            (0, 255, 0),
            2,
        )
        cv2.line(
            im_array,
            (center_x, center_y - 150),
            (center_x, center_y + 150),
            (0, 255, 0),
            2,
        )

        if tracking_state["tracked_id"] is not None and current_boxes:
            tracked_box = next(
                (b for b in current_boxes if b[4] == tracking_state["tracked_id"]), None
            )
            if tracked_box:
                x1, y1, x2, y2, _ = tracked_box
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                # Draw line from center to object (Thickness 2 -> 4)
                cv2.line(im_array, (center_x, center_y), (cx, cy), (0, 0, 255), 4)

                # Calculate relative position for display
                dx = cx - center_x
                dy = cy - center_y

                # Draw text near the object
                text = f"Pos: ({dx}, {dy})"
                cv2.putText(
                    im_array,
                    text,
                    (cx + 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

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
