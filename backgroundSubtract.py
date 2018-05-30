import numpy as np
import cv2
import sys

def detectmotion(camera_to_use,video):
    cap = cv2.VideoCapture()
    cv2.namedWindow("Foreground Objects", cv2.WINDOW_FULLSCREEN);
    cv2.namedWindow("Live", cv2.WINDOW_FULLSCREEN);
    
    fgbg = cv2.createBackgroundSubtractorMOG2()
    if(cap.open(str(video)) or cap.open(camera_to_use)):
        while(1):
            ret, frame = cap.read()
            bilateral_filtered_image = cv2.bilateralFilter(frame, 7, 150, 150)
            gray = cv2.cvtColor(bilateral_filtered_image, cv2.COLOR_BGR2GRAY)
            fgmask = fgbg.apply(gray)
            # fgBlur = cv2.GaussianBlur(frame,(15,15),0)
            retval, fgthres = cv2.threshold(fgmask.copy(), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # retval, fgthres = cv2.threshold(fgmask.copy(), 30, 255, cv2.THRESH_BINARY);
            fgdilated = cv2.dilate(fgthres, kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations = 3);
            # fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel = np.ones((5,5),np.uint8))
            # close_operated_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
            # retval, fgthres = cv2.threshold(fgmask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # fgBlur = cv2.GaussianBlur(fgthres,(15,15),0)
            # median = cv2.medianBlur(fgthres, 5)
            
            (im2, contours, hierarchy) = cv2.findContours(fgdilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                if cv2.contourArea(c) < 500:
                    continue

                (x,y,w,h) = cv2.boundingRect(c)

                cv2.rectangle(frame, (x,y), (x + w, y+h), (0,255,0), 2)

            cv2.imshow("Foreground Objects",fgdilated)
            cv2.imshow("Live", frame);
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
                
        cap.release()
        cv2.destroyAllWindows()
    else:
        print 'Error'

if __name__ == "__main__":

    if len(sys.argv) == 1:
        detectmotion(0,"")

    detectmotion(1,sys.argv[1])