import cv2 as cv
import numpy as np

# Open video capture
vs = cv.VideoCapture(r"C:\Users\dharu\Downloads\Object Tracking using Yolo and DEEPSORT\macv-obj-tracking-video.mp4")

# Get video parameters (width, height, fps)
frame_width = int(vs.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(vs.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = vs.get(cv.CAP_PROP_FPS)

# Define new resolution (e.g., upscale to 1280x720)
new_resolution = (1280, 720)

# Create video writer
out = cv.VideoWriter('output_upscaled.mp4', cv.VideoWriter_fourcc(*'mp4v'), fps, new_resolution)

while True:
    ret, frame = vs.read()
    if not ret:
        break

    # Resize each frame
    frame_resized = cv.resize(frame, new_resolution)

    # Slightly sharpen the frame to enhance clarity
    kernel = np.array([[0, -1, 0], 
                       [-1, 5,-1], 
                       [0, -1, 0]])
    frame_sharpened = cv.filter2D(frame_resized, -1, kernel)

    # Write the resized and sharpened frame to the output video
    out.write(frame_sharpened)

# Release resources
vs.release()
out.release()
cv.destroyAllWindows()
