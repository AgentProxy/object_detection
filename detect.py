import sys
import cv2
import numpy as np

def detectmotion(input):

	cap = cv2.VideoCapture()
	if(not cap.open(input)):
		print "No video found"
		return 
	# create window by name (as resizable)
	cv2.namedWindow("Image Input", cv2.WINDOW_FULLSCREEN);
	cv2.namedWindow("Foreground Objects", cv2.WINDOW_FULLSCREEN);
	cv2.namedWindow("Background Model", cv2.WINDOW_FULLSCREEN);
	# cv2.namedWindow("B Model", cv2.WINDOW_FULLSCREEN);
	# cv2.namedWindow("Foreground Probabiity", cv2.WINDOW_FULLSCREEN);

	# create GMM background subtraction object (using default parameters - see manual)

	# mog = cv2.createBackgroundSubtractorMOG2(history=3000, varThreshold=16, detectShadows=True);
	# mog = cv2.createBackgroundSubtractorMOG2(history=2000, varThreshold=16);
	mog = cv2.createBackgroundSubtractorMOG2();
	while(True):
		# if video file successfully open then read frame from video
		ret = None
		frame = None
		if (cap.isOpened):
			ret, frame = cap.read()
			# when we reach the end of the video (file) exit cleanly
			if (ret == 0):
				return
			grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			# grayFrame = cv2.equalizeHist(grayFrame)

		# add current frame to background model and retrieve current foreground objects
		# sample = mog.apply(grayFrame)
		# cv2.imshow("equalize", sample)
		fgmask = mog.apply(grayFrame);
		# get current background image (representative of current GMM model)
		bgmodel = mog.getBackgroundImage();

		kernel = np.ones((2,2),np.uint8)
		fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
		# thresh, fgmask = cv2.threshold(fgmask, 0, 255, cv2.THRESH_OTSU)
		
		# kernel = np.ones((3,3),np.uint8)
		# fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
		# fgmask = cv2.erode(fgmask, kernel, iterations = 5);
	
		# least to greatest
		startCoords = []
		endCoords = []
		(im2, contours, hierarchy) = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		for c in contours:
			area = cv2.contourArea(c)
			if area < 80:
				continue
			
			(x,y,w,h) = cv2.boundingRect(c)
			cv2.rectangle(frame, (x,y), (x + w, y + h), (0,255,0), 2)
			if(not len(startCoords) and not len(endCoords)):
				startCoords.extend((x, y))
				endCoords.extend((x+w, y+h))
			else:
				if(x < startCoords[0]):
					startCoords[0] = x
				if(y < startCoords[1]):
					startCoords[1] = y
				if(x + w > endCoords[0]):
					endCoords[0] = x + w
				if(y + h > endCoords[1]):
					endCoords[1] = y + h

		# display images - input, background and original

		cv2.imshow("Foreground Objects", fgmask)
		if(startCoords and endCoords):
			cv2.rectangle(frame, tuple(startCoords), tuple(endCoords), (0,0,255), 2)
		
		cv2.imshow("Image Input", frame)
		cv2.imshow("Background Model", bgmodel);
		# cv2.imshow("Foreground Objects",fgdilated);
		# cv2.imshow("Foreground Probabiity",fgmask);
	


		key = cv2.waitKey(40) & 0xFF; 

		if (key == ord('x')):
			return

	# close all windows
	cv2.destroyAllWindows()

if __name__ == "__main__":
	camera = 0
	# filename = "trooper.gif"
	# filename = "slow-motion2.gif"
	filename = "slow-motion1.gif"
	filename = "people-walking.gif"
	filename = "people-cctv.mp4"
	filename = "just-do-it.mp4"
	filename = "airport.mp4"
	filename = "cctv-fall.mp4"
	# detectmotion(1, "slow-motion1.gif")
	# detectmotion(1, filename)
	# detectmotion(camera)
	data_dir = "dataset/"
	detectmotion(data_dir + filename)