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
    cv2.namedWindow("Background Model", cv2.WINDOW_FULLSCREEN);

    # mog = cv2.createBackgroundSubtractorMOG2();
    firstFrame = None
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

        # if firstFrame is None:
        #     firstFrame = gaussianGray
        #     continue

    # fgmask = mog.apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY));
    # get current background image (representative of current GMM model)
    # bgmodel = mog.getBackgroundImage();
        frameDelta = mog.apply(gaussianGray);
# get current background image (representative of current GMM model)
        bgmodel = mog.getBackgroundImage();
        thresh = cv2.threshold(frameDelta,25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        # (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		# cv2.CHAIN_APPROX_SIMPLE)

    # fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, np.ones(kernel, np.uint8))

    # least to greatest
        startCoords = []
        endCoords = []
        # (im2, contours, hierarchy) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            a = cv2.contourArea(c)
            if a < area:
                continue


        #         if cv2.contourArea(c) < args["min_area"]:
		# 	continue
 
		# # compute the bounding box for the contour, draw it on the frame,
		# # and update the text
		# (x, y, w, h) = cv2.boundingRect(c)
		# cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		# text = "Occupied"



            (x,y,w,h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x,y), (x + w, y + h), (0,0,255), 2)
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
    filename = "cctv-fall.mp4"

    # detectmotion(camera)
    kernel = (2, 2)
    area = 1000
    # detectmotion(0, kernel, area)
    detectmotion(data_dir + filename, kernel, area)