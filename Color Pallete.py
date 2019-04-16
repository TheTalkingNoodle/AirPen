import cv2
import numpy as np

def set_cam(vino):
    video_path=vino
    cap=cv2.VideoCapture(video_path)
    #cap.set(3, 1920)
    #cap.set(4, 1080)
    set_cam.cap=cap

vino = 1
set_cam(vino)
cap = set_cam.cap

############################## Choose Pixel
def click_and_crop(event, x, y, flags, param):

    global refPt, cropping

    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True
        #print "Top Left", refPt[0]
        if refPt:
            print "Pixel Color: HSV & RGB"
            print hsv[refPt[0][1],refPt[0][0]]
            print frame[refPt[0][1],refPt[0][0]]
            print lab[refPt[0][1],refPt[0][0]]
            print "--------------"

cv2.namedWindow("Original Image")
cv2.setMouseCallback("Original Image", click_and_crop)
##############################

# Create a pipeline
def nothing(x):
    pass

hsv_min=[133, 0, 173]
hsv_max=[255, 156, 255]

cv2.namedWindow("image")
cv2.resizeWindow("image", 640,480)

# create trackbars for color change
cv2.createTrackbar('lowH', 'image', hsv_min[0], 255,nothing)
cv2.createTrackbar('highH', 'image', hsv_max[0], 255, nothing)
cv2.createTrackbar('lowS', 'image', hsv_min[1], 255, nothing)
cv2.createTrackbar('highS', 'image', hsv_max[1], 255, nothing)
cv2.createTrackbar('lowV', 'image', hsv_min[2], 255, nothing)
cv2.createTrackbar('highV', 'image', hsv_max[2], 255, nothing)

frame_no=0
n=0
wb_flag =0 #white balance flag
calibrate_0=0

while(True):
    if n==0 and vino!=10:
        ret, frame = cap.read()
        frame = cv2.resize(frame,(1280,720))
        image=frame.copy()
        lab= cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    # grab the frame
    key = cv2.waitKey(1) & 0xFF
    # if the 'c' key is pressed, break from the loop

    #frame=cv2.resize(frame,(960,720))
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    ilowH = cv2.getTrackbarPos('lowH', 'image')
    ihighH = cv2.getTrackbarPos('highH', 'image')
    ilowS = cv2.getTrackbarPos('lowS', 'image')
    ihighS = cv2.getTrackbarPos('highS', 'image')
    ilowV = cv2.getTrackbarPos('lowV', 'image')
    ihighV = cv2.getTrackbarPos('highV', 'image')

    lower_HSV = np.array([ilowH, ilowS, ilowV])
    higher_HSV = np.array([ihighH, ihighS, ihighV])
    mask_HSV = cv2.inRange(frame, lower_HSV, higher_HSV)
    frame_HSV = cv2.bitwise_and(frame, frame, mask=mask_HSV)

    frame_total=cv2.bitwise_and(frame, frame, mask=mask_HSV)

    cv2.imshow('frame_total', frame_total)
    cv2.imshow('Original Image', frame)

    if key == ord("q"):
        break

cv2.destroyAllWindows()
print list(lower_HSV)
print list(higher_HSV)