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

		# add current frame to background model and retrieve current foreground objects
		bgmodel = mog.getBackgroundImage();
		fgmask = mog.apply(grayFrame);
		# get current background image (representative of current GMM model)

		kernel = np.ones((5,5),np.uint8)
		# kernel1 = np.ones((1,1),np.uint8)
		# kernel2 = np.ones((3,3),np.uint8)
		# kernel3 = np.ones((5,5),np.uint8)

		# sample1 = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel1)
		# sample2 = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel2)
		# sample3 = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel3)

		# cv2.imshow('Varying kernels', np.hstack([sample1, sample2, sample3]))
		fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
	
		# least to greatest
		startCoords = []
		endCoords = []
		(im2, contours, hierarchy) = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		for c in contours:
			area = cv2.contourArea(c)
			if area < 30:
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
		key = cv2.waitKey(40) & 0xFF; 

		if (key == ord('x')):
			return

	cv2.destroyAllWindows()

if __name__ == "__main__":
	data_dir = "dataset/"
	camera = 0
	# filename = "trooper.gif"
	filename = "slow-motion2.gif"
	# filename = "slow-motion1.gif"
	# filename = "people-walking.gif"
	# filename = "people-cctv.mp4"
	# filename = "just-do-it.mp4"
	# filename = "airport.mp4"
	# filename = "cctv-fall.mp4"

	# detectmotion(camera)
	detectmotion(data_dir + filename)