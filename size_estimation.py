import cv2 as cv
def processVideo():
    path = r"C:\Users\dharu\Downloads\Object Tracking using Yolo and DEEPSORT\macv-obj-tracking-video.mp4"
    #read video
    vs = cv.VideoCapture(path)
    while True:
        (grabbed,frame) = vs.read()
        if not grabbed:
            break
#show frames
        cv.imshow('image', frame)

        height, width, channels = frame.shape

# Print dimensions
        print(f"Width: {width}")
        print(f"Height: {height}")
        print(f"Channels: {channels}")  # 3 for a color image (BGR), 1 for grayscale


        cv.waitKey(24)   

processVideo()