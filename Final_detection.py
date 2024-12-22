import cv2 as cv
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict
import time
import datetime

def processVideo():
    """
    Main function to process the video, detect objects, track them, and save the output with annotated information.
    """
    count = 0  # Counter for the total number of unique objects detected
    detection_classes = []  # List to store class names of detected objects
    counter_cache = []  # Cache to store track IDs of detected objects to avoid recounting
    trails = defaultdict(list)  # Dictionary to store the trails of tracked objects
    object_times = defaultdict(float)  # Dictionary to store time each object spends in the video
    
    path = r"C:\Users\dharu\Downloads\Object Tracking using Yolo and DEEPSORT\output_upscaled.mp4"
    # Read video
    vs = cv.VideoCapture(path)
    # Load the model
    model = YOLO('yolov8x.pt')

    # Initialize DeepSort object tracker
    object_tracker = DeepSort(max_iou_distance=0.7,
                              max_age=5,
                              n_init=3,
                              nms_max_overlap=1.0,
                              max_cosine_distance=0.2,
                              nn_budget=None,
                              gating_only_position=False,
                              override_track_class=None,
                              embedder="mobilenet",
                              half=True,
                              bgr=True,
                              embedder_gpu=True,
                              embedder_model_name=None,
                              embedder_wts=None,
                              polygon=False,
                              today=None
                              )

    # Get the frame width and height for video writer
    frame_width = int(vs.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vs.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = int(vs.get(cv.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
    out = cv.VideoWriter('output_tracked.webm', cv.VideoWriter_fourcc(*'VP80'), fps, (frame_width, frame_height))
    
    start_time = time.time()
    while True:
        (grabbed, frame) = vs.read()
        if not grabbed:
            break

        # Predict objects in the current frame
        results = model.predict(frame, stream=False, conf=0.5)
        detection_classes = results[0].names
        
        for result in results:
            for data in result.boxes.data.tolist():
                id = int(data[5])
                drawBox(data, frame, detection_classes[id])
                x1, y1, x2, y2, _, _ = data
                cx, cy = int((x1+x2) / 2), int((y1+y2) / 2)
                cv.circle(frame, (cx, cy), 5, (0, 255, 0), -1)  # Draw a circle at the center of the detected object

            # Get details of detected objects and update tracks
            details = get_details(result, frame)
            tracks = object_tracker.update_tracks(details, frame=frame)
            current_time = time.time()
            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                bbox = track.to_ltrb()
                x1, y1, x2, y2 = bbox
                cx, cy = int((x1+x2) / 2), int((y1+y2) / 2)
                trails[track_id].append((cx, cy))
                
                # Draw trails of tracked objects
                for i in range(1, len(trails[track_id])):
                    cv.line(frame, trails[track_id][i - 1], trails[track_id][i], (0, 255, 255), 2)
                
                if track_id not in counter_cache:
                    counter_cache.append(track_id)
                    count += 1
                    object_times[track_id] = 0  # Initialize time for new object

                # Update object time
                object_times[track_id] += (current_time - start_time) / fps

                # Display ID and count
                cv.putText(frame, "ID: " + str(track_id), (int(x1), int(y1) - 10), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                cv.putText(frame, "Object Count: " + str(count), (100, 100), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 9)
                cv.putText(frame, "Time: {:.2f}s".format(object_times[track_id]), (int(x1), int(y1) - 30), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        
        start_time = current_time

        # Write the frame into the output file
        out.write(frame)

        # Show frames
        cv.imshow('image', frame)
        if cv.waitKey(24) & 0xFF == ord('q'):
            break

    vs.release()
    out.release()
    cv.destroyAllWindows()
    
    # Export metrics to a file
    export_metrics(count, object_times)

def drawBox(data, image, name):
    """
    Function to draw bounding boxes around detected objects.
    """
    x1, y1, x2, y2, _, _ = data
    p1 = (int(x1), int(y1))
    p2 = (int(x2), int(y2))
    cv.rectangle(image, p1, p2, (0, 0, 255), 3)  # Draw rectangle around the object
    cv.putText(image, name, (int(x1), int(y1) - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # Put class name above the rectangle
    return image

def get_details(result, image):
    """
    Function to extract details of detected objects required for tracking.
    """
    classes = result.boxes.cls.numpy()  # Class IDs
    conf = result.boxes.conf.numpy()  # Confidence scores
    xywh = result.boxes.xywh.numpy()  # Bounding box coordinates in xywh format

    detections = []
    for i, item in enumerate(xywh):
        sample = (item, conf[i], classes[i])
        detections.append(sample)

    return detections

def export_metrics(total_count, object_times):
    """
    Function to export metrics to an HTML file.
    """
    with open("output_metrics.html", "w") as f:
        f.write("<html><head><title>Object Tracking Metrics</title></head><body>")
        f.write("<h1>Object Tracking Metrics</h1>")
        f.write("<p>Total unique objects detected: {}</p>".format(total_count))
        f.write("<table border='1'><tr><th>Object ID</th><th>Time in video (s)</th></tr>")
        for track_id, time_spent in object_times.items():
            f.write("<tr><td>{}</td><td>{:.2f}</td></tr>".format(track_id, time_spent))
        f.write("</table>")
        f.write("<video width='640' height='480' controls><source src='output_tracked.webm' type='video/webm'>Your browser does not support the video tag.</video>")
        f.write("</body></html>")

processVideo()
