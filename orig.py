import sys
import cv2
import numpy as np

def detectmotion(camera_to_use,video):

    cap = cv2.VideoCapture();

    if (cap.open(str(video)) or cap.open(camera_to_use)):

    # create window by name (as resizable)

        cv2.namedWindow("Live Camera Input", cv2.WINDOW_NORMAL);
        cv2.namedWindow("Background Model", cv2.WINDOW_NORMAL);
        cv2.namedWindow("Foreground Objects", cv2.WINDOW_NORMAL);
        cv2.namedWindow("Foreground Probabiity", cv2.WINDOW_NORMAL);

        # create GMM background subtraction object (using default parameters - see manual)

        mog = cv2.createBackgroundSubtractorMOG2(history=2000, varThreshold=16, detectShadows=True);

        while(1):

        # if video file successfully open then read frame from video

            if (cap.isOpened):
                ret, frame = cap.read();

        # when we reach the end of the video (file) exit cleanly

                if (ret == 0):
                    break

    # add current frame to background model and retrieve current foreground objects

            fgmask = mog.apply(frame);

    # threshold this and clean it up using dilation with a elliptical mask

            fgthres = cv2.threshold(fgmask.copy(), 200, 255, cv2.THRESH_BINARY)[1];
            fgdilated = cv2.dilate(fgthres, kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations = 3);

            # get current background image (representative of current GMM model)

            bgmodel = mog.getBackgroundImage();

            # display images - input, background and original

            cv2.imshow("Live Camera Input",frame);
            cv2.imshow("Foreground Objects",fgdilated);
            cv2.imshow("Foreground Probabiity",fgmask);
            cv2.imshow("Background Model", bgmodel);

            key = cv2.waitKey(40) & 0xFF; 

            if (key == ord('x')):
                break

    # close all windows

        cv2.destroyAllWindows()

    else:

        print("No video file specified or camera connected.")

if __name__ == "__main__":

    if len(sys.argv) == 1:
        detectmotion(0,"")

    detectmotion(1,sys.argv[1])