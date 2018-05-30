import sys
import cv2
import numpy as np

def detectmotion(input, kernel, area):
    cap = cv2.VideoCapture()
    if(not cap.open(input)):
        print "No video found"
        return 
    # create window by name (as resizable)
    cv2.namedWindow("Image Input", cv2.WINDOW_FULLSCREEN);
    cv2.namedWindow("Foreground Objects", cv2.WINDOW_FULLSCREEN);
    #Subtract foreground and background
    mog = cv2.createBackgroundSubtractorMOG2();
    while(True):
    # if video file successfully open then read frame from video
        if(cap.isOpened):
            ret, frame = cap.read()
        # when we reach the end of the video (file) exit cleanly
        if (ret == 0):
            return
        # add current frame to background model and retrieve current foreground objects
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gaussianGray = cv2.GaussianBlur(gray,(21,21),0)
        fgmask = mog.apply(gaussianGray);
        # get current background image (representative of current GMM model)
        bgmodel = mog.getBackgroundImage();
        threshBinary = cv2.threshold(fgmask,25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(threshBinary, None, iterations=2)

        startCoords = []
        endCoords = []
        image, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            a = cv2.contourArea(c)
            if a < area:
                continue

            (x,y,w,h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x,y), (x + w, y + h), (0,0,255), 2)
            # for small boxes. Commented out on purpose


            # if(not len(startCoords) and not len(endCoords)):
            #     startCoords.extend((x, y))
            #     endCoords.extend((x+w, y+h))
            # else:
            #     if(x < startCoords[0]):
            #         startCoords[0] = x
            #     if(y < startCoords[1]):
            #         startCoords[1] = y
            #     if(x + w > endCoords[0]):
            #         endCoords[0] = x + w
            #     if(y + h > endCoords[1]):
            #         endCoords[1] = y + h

        # display images - input, background and original
        cv2.imshow("Foreground Objects", thresh)
        # if(startCoords and endCoords):
        #     cv2.rectangle(frame, tuple(startCoords), tuple(endCoords), (0,0,255), 2)
        cv2.imshow("Image Input", frame)
        key = cv2.waitKey(40) & 0xFF; 

        if (key == ord('x')):
            return

    cv2.destroyAllWindows()

if __name__ == "__main__":
    data_dir = "dataset/"
    camera = 0
    # filename = "trooper.gif"
    # filename = "slow-motion2.gif"
    # filename = "slow-motion1.gif"
    # filename = "people-walking.gif"
    # filename = "people-cctv.mp4"
    # filename = "just-do-it.mp4"
    # filename = "airport.mp4"
    # filename = "cctv-fall.mp4"

    # detectmotion(camera)
    kernel = (2, 2)
    area = 1000                 #adjust depending on video situation


    # Uncomment if webcam will be used
    # detectmotion(0, kernel, area)
    detectmotion(data_dir + filename, kernel, area)