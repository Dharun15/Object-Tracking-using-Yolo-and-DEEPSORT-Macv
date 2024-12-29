from ultralytics import YOLO
from ultralytics.solutions import heatmap
import cv2 as cv
import imutils


def track_createHeatmap():
    path = r"C:\Users\dharu\Downloads\output_tracked.mp4"
    vs = cv.VideoCapture(path)
    model = YOLO("yolov8l.pt")
    frame_width = int(vs.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vs.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = int(vs.get(cv.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
    out = cv.VideoWriter('added_heatmap.webm', cv.VideoWriter_fourcc(*'VP80'), fps, (frame_width, frame_height))
    #classes_for_heatmap = [0, 2]
    heatmap_obj = heatmap.Heatmap()
    (grabbed, frame) = vs.read()
    h, w, c = frame.shape
    heatmap_obj.set_args(colormap=cv.COLORMAP_PARULA,
                         imw=w,
                         imh=h,
                         view_img=True,
                         shape="circle")
    while True:
        # read the next frame from the file
        (grabbed, frame) = vs.read()
        if not grabbed:
            break
        tracks = model.track(frame )
        for item in tracks:
            print("got it:-",item[0].boxes)
        print("result##:-",heatmap_obj.extract_results(tracks))
        frame = heatmap_obj.generate_heatmap(frame, tracks)
        out.write(frame)       

        cv.imshow('image', frame)

        # waitKey() waits for a key press to close the window and 0 specifies indefinite loop
        cv.waitKey(24)

track_createHeatmap()