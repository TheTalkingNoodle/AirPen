import numpy as np
import cv2
import fnmatch, os
from matplotlib.colors import rgb_to_hsv
from PIL import Image
from Model import Model, DecoderType
from DataLoader import DataLoader, Batch
from SamplePreprocessor import preprocess
from gtts import gTTS


################################## Set Webcam
def set_cam(vino):
    video_path=vino
    cap=cv2.VideoCapture(video_path)
    #cap.set(3, 1920)
    #cap.set(4, 1080)
    set_cam.cap=cap

############################## Choose Pixel
def click_and_crop(event, x, y, flags, param):

    global refPt, cropping

    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True
        #print "Top Left", refPt[0]
        if refPt:
            print ("HSV")
            print (hsv[refPt[0][1],refPt[0][0]])
            
            print ("BGR")
            print (frame[refPt[0][1],refPt[0][0]])

cv2.namedWindow("frame")
cv2.setMouseCallback("frame", click_and_crop)
##############################

def imfill(im_in):
    th, im_th = cv2.threshold(im_in, 220, 255, cv2.THRESH_BINARY)

    # Copy the thresholded image.
    im_floodfill = im_th.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255);

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    im_out = im_th | im_floodfill_inv

    return im_out

def tracker_func(frame):

    timer = cv2.getTickCount()

    # Update tracker
    ok, bbox = tracker.update(frame)

    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

    # Draw bounding box
    if ok:
        fail_flag = 0
        # Tracking success
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
    else:
        # Tracking failure
        cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        fail_flag = 1

    # Display FPS on frame
    cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

    return frame, bbox, fail_flag

vino = 1
set_cam(vino)
cap = set_cam.cap
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

class FilePaths:
    fnCharList = '/home/asad/Documents/Air-Pen/SimpleHTR/model/charList.txt'
decoderType = DecoderType.BeamSearch
#decoderType = DecoderType.BestPath
model = Model(open(FilePaths.fnCharList).read(), decoderType, mustRestore=True)

lower_HSV=      [133, 0, 173]
higher_HSV=     [255, 156, 255]
width = int(480*2)
height = int(360*2)
white_board = np.ones((height, width, 3), np.uint8) * 255
tracker_flag = 0
fail_flag = 0
pen = 0
bbox_prev = []
final = []
pen_color =[0,0,255]

while(True):

    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)
    frame=cv2.resize(frame,(width,height))

    if ret == True:
        final_frame=frame.copy()

        if tracker_flag == 1:
            final_frame, bbox_n, fail_flag = tracker_func(final_frame)
            p1 = (int(bbox_n[0]), int(bbox_n[1]))
            p2 = (int(bbox_n[0] + bbox_n[2]), int(bbox_n[1] + bbox_n[3]))

            cx = int((p1[0] + p2[0] )/ 2)
            cy = int((p1[1] + p2[1]) / 2)

            cv2.circle(final_frame, (cx, cy), radius=5, color=[0, 0, 0], thickness=-1)

            if pen == 1:
                if bbox_prev == []:
                    cv2.circle(white_board, (cx, cy), radius=5, color=[0, 0, 0], thickness=-1)
                    bbox_prev = bbox_n
                else:
                    p1_pre = (int(bbox_prev[0]), int(bbox_prev[1]))
                    p2_pre = (int(bbox_prev[0] + bbox_prev[2]), int(bbox_prev[1] + bbox_prev[3]))

                    cx_pre = int((p1_pre[0] + p2_pre[0]) / 2)
                    cy_pre = int((p1_pre[1] + p2_pre[1]) / 2)

                    cv2.circle(white_board, (cx, cy), radius=5, color=[0, 0, 0], thickness=-1)
                    cv2.line(white_board,(cx,cy),(cx_pre,cy_pre),color = [0,0,0],thickness=7)

                    if all(v == 0 for v in bbox_n) == False:
                        bbox_prev = bbox_n

            final = white_board.copy()
            if pen_color!=[]:
                cv2.circle(final, (cx, cy), radius=5, color=pen_color, thickness=-1)

        blank_image = np.zeros((frame.shape[0], frame.shape[1], 3), np.uint8)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask_HSV = cv2.inRange(frame, np.array(lower_HSV), np.array(higher_HSV))
        #frame_HSV = cv2.bitwise_and(frame, frame, mask=mask_HSV)
        #green_mask = cv2.dilate(mask_HSV, np.ones((5, 5), np.uint8), iterations=2)
        mask_HSV = cv2.erode(mask_HSV, np.ones((3, 3), np.uint8), iterations=1)
        #green_mask[np.where((green_mask == [255, 255, 255]).all(axis=2))] = [0, 255, 0]

        contours, hierarchy = cv2.findContours(mask_HSV, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt_n, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            if area > 300:
                cv2.drawContours(blank_image, contours, cnt_n, (255, 255, 255), 2)
                bbox = (x,y,w,h)
                cv2.rectangle(final_frame,(x,y),(x+w,y+h),color = [0,255,255])

        frame_HSV = cv2.bitwise_and(final_frame, final_frame, mask=mask_HSV)
        blank_image = cv2.cvtColor(blank_image,cv2.COLOR_BGR2GRAY)
        blank_image = imfill(blank_image)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord('s'):
            if tracker_flag ==0:
                print "Tracking Started."
                tracker = cv2.TrackerCSRT_create()
                ok = tracker.init(frame, bbox)
                tracker_flag  = 1

        elif key == ord('w'):
            pen = 1
            pen_color = [0,255,0]
        elif key == ord('e'):
            pen = 0
            pen_color = [0,0,255]
            bbox_prev= []
        elif key == ord('r'):
            tracker_flag = 0
        elif key == ord('c'):
            white_board = np.ones((height, width, 3), np.uint8) * 255
        elif key == ord('a'):

            img = white_board[int(white_board.shape[0]/2)-64:int(white_board.shape[0]/2)+64,int(white_board.shape[1]/2)-256:int(white_board.shape[1]/2)+256]
            img = cv2.resize(img,(128,32))
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            img = preprocess(img, Model.imgSize)

            batch = Batch(None, [img])
            (recognized, probability) = model.inferBatch(batch, True)
            print('Recognized:', '"' + recognized[0] + '"')
            print('Probability:', probability[0])

            if recognized[0].isalpha():
                tts = gTTS(text=recognized[0], lang='en')
                tts.save("result.mp3")
                os.system("mpg123 result.mp3")

        if fail_flag == 1:
            print "Tracking Started."
            tracker = cv2.TrackerCSRT_create()
            ok = tracker.init(frame, bbox)
            tracker_flag  = 1

        # Display the resulting frame
        cv2.imshow("frame",final_frame)

        if final != []:
            f = final[int(final.shape[0]/2)-64:int(final.shape[0]/2)+64,int(final.shape[1]/2)-256:int(final.shape[1]/2)+256]

            cv2.imshow("white board",f)

        #cv2.imshow("mask",mask_HSV)

cap.release()
cv2.destroyAllWindows()