import cv2

import csv
import numpy as np
import pandas as pd
# Create a video capture object, in this case we are reading the video from a file
vid_capture = cv2.VideoCapture('rgb_CAM7_20220203_092129_raw_input_Tail.avi')
ret0, frame0 = vid_capture.read()

background = cv2.resize(cv2.rotate(frame0, cv2.ROTATE_90_CLOCKWISE), (480, 720))
height, width = (frame0.shape[1], frame0.shape[0])
rate_H = float(height/720)
rate_W = float(width/480)

#Data of tail
data = pd.read_csv('CAM7_20220203_092129_output_Tail_unique.csv')
position = data.iloc[:, [6, 7, 8, 9]]
position = np.array(position)

kernel = np.array([[0, 0, 1, 0, 0],
				   [0, 1, 1, 1, 0],
				   [1, 1, 1, 1, 1],
				   [0, 1, 1, 1, 0],
				   [0, 0, 1, 0, 0]], np.uint8)

kernel2 = np.array([[0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0],
                   [0, 0, -2, 8, -2, 0, 0],
                   [0, 0, -2, -25, -2, 0, 0],
                   [0, 0, -2, 8, -2, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0]])
gamma = 0.6
count = 0


while(vid_capture.isOpened()):
	# vid_capture.read() methods returns a tuple, first element is a bool 
	# and the second is frame
	ret, frame = vid_capture.read()	
	frame = cv2.resize(cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE), (480, 720))
	
	if(count<position.shape[0]):
		x1 = int(np.round(position[count][0]/rate_W))
		y1 = int(np.round(position[count][1]/rate_H))
		x2 = int(np.round(position[count][2]/rate_W))
		y2 = int(np.round(position[count][3]/rate_H))
	frame = cv2.rectangle(frame, pt1=(x1, y1), pt2=(x2, y2), color=(255, 0, 0), thickness=1)
	
	if ret == True:				
		frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		bg_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
		#Chech lech giua frame trc va sau = nhan biet chuyen dong
		diff = cv2.absdiff(frame_gray, bg_gray)
		
		#OPen xong tang do day
		diff = cv2.GaussianBlur(diff, (3, 3), 2)
		diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel)
		_, diff = cv2.threshold(diff, 45, 255, cv2.THRESH_BINARY)
		
		#Add Weight 2 image
		frame_change = cv2.addWeighted(frame_gray, 0.3, diff, 0.8, 0.0)
		
		frame_filter = cv2.filter2D(frame_change, ddepth=-1, kernel=kernel2)

		diff2 = cv2.absdiff(frame_filter, frame_gray)
		diff2 = 255-diff2
		diff2 = cv2.GaussianBlur(diff2, (3, 3), 0)
		
		_, result = cv2.threshold(diff2, 40, 255, cv2.THRESH_BINARY_INV)
		
		#result = cv2.rectangle(result, pt1=(x1, y1), pt2=(x2, y2), color=(255, 0, 0), thickness=3)

		frame_gray = frame_gray.reshape((frame_gray.shape[0], frame_gray.shape[1], 1))
		frame_gray = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
		result = result.reshape((result.shape[0], result.shape[1], 1))
		result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

		if x1 > 0:
			tail1 = frame_gray[y1:y2, x1:x2]
			tail2 = result[y1:y2, x1:x2]
			tail3 = frame[y1:y2, x1:x2]

		# Draw Contours
		tail2 = cv2.Canny(tail2, 50, 75)
		contours, _ = cv2.findContours(tail2,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		tail2 = cv2.cvtColor(tail2, cv2.COLOR_GRAY2BGR)
		for ct in contours:
			cv2.drawContours(tail3, [ct], -1, (255, 0, 0), 2)
		
		print(tail1.shape, tail3.shape)
		final = cv2.hconcat([frame_gray, result, frame])
		tail = cv2.hconcat([tail1, tail2, tail3])
		cv2.imshow('Frame', final)
		cv2.imshow('Start', tail)

		count += 1
		# 10 is in milliseconds, try to increase the value, say 50 and observe
		key = cv2.waitKey(1)
		
		if key == ord('q'):
			break
	else:
		break

	background = frame
	background = cv2.GaussianBlur(background, (5, 5), 0)
	
# Release the video capture object
vid_capture.release()
cv2.destroyAllWindows()

