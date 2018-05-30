import sys
import cv2
import numpy as np

def detectmotion(camera_to_use,video):

    cap = cv2.VideoCapture();

    if(cap.open(str(video)) or cap.open(camera_to_use)):

        cv2.namedWindow("Live Camera Input", cv2.WINDOW_FULLSCREEN);
        cv2.namedWindow("Foreground Objects", cv2.WINDOW_FULLSCREEN);

        mog = cv2.createBackgroundSubtractorMOG2(100, 50, detectShadows = True);
    
        while(1):

            if (cap.isOpened):
                ret, frame = cap.read();

                if (ret == 0):
                    break

            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fgmask = mog.apply(frame)
            
            retval, fgGrey = cv2.threshold(gray.copy(), 150, 255, cv2.THRESH_BINARY);
            # bgmodel = mog.getBackgroundImage();
            # fgmask = cv2.morphologyEx(fgmask.copy(), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
            # grayscaled = cv2.cvtColor(fgmask,cv2.COLOR_BGR2GRAY)
            
            
            (im2, contours, hierarchy) = cv2.findContours(fgGrey.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                if cv2.contourArea(c) < 500:
                    continue

                (x,y,w,h) = cv2.boundingRect(c)

                cv2.rectangle(frame, (x,y), (x + w, y+h), (0,255,0), 2)

                cv2.imshow("Live Camera Input",frame);
                cv2.imshow("Foreground Objects",fgGrey);


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