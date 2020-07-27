#Before you run this program, you need to install scipy, imutils, pygame, dlib and cv2 modules in your device..
#after successful installation, please check all the files of this program are in the same folder only
from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
from pygame import mixer

def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear

mixer.init()
sound = mixer.Sound('alarm.wav') #plays the sound of this alarm

thresh = 0.25 # this is the threshold value where you should specify how long we have to wait before beeping alarm
frame_check = 10#checking the frames
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")# Dat file is the crux of the code, it contains all the eye_aspect_ratio

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"] #imports left eye ratio
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]#imports right eye ratio
cap=cv2.VideoCapture(0) #capturing video from the camera
flag=0
while True:#every time,
	ret, frame=cap.read()#reads the captured frame from the video
	frame = imutils.resize(frame, width=1000,height=800)#resizes the frame for size purposes
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	subjects = detect(gray, 0)
	for subject in subjects:
		shape = predict(gray, subject)
		shape = face_utils.shape_to_np(shape)#converting to NumPy Array
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		ear = (leftEAR + rightEAR) / 2.0
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		if ear < thresh:
			flag += 1
			print (flag)
			if flag >= frame_check:#literally, if the algorithm meets the persons eyes as sleepy, he would do the following..
				cv2.putText(frame, "****************ALERT!****************", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				cv2.putText(frame, "****************ALERT!****************", (10,325),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				sound.play()
				print ("Sleepy")
		else:
			flag = 0
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):#please enter q to destroy the windows
		break
cv2.destroyAllWindows()#destroy all windows
#cap.stop()#stop the program
