# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import imutils
import time
import dlib
import cv2


def getheadpose(frame,shape,size):
    #2D image points. If you change the image, you need to change vector
    image_points = np.array([
                                (shape[33, :]),     # Nose tip
                                (shape[8,  :]),     # Chin
                                (shape[36, :]),     # Left eye left corner
                                (shape[45, :]),     # Right eye right corne
                                (shape[48, :]),     # Left Mouth corner
                                (shape[54, :])      # Right mouth corner
                            ], dtype="double")
     
    # 3D model points.
    model_points = np.array([
                                (0.0, 0.0, 0.0),             # Nose tip
                                (0.0, -330.0, -65.0),        # Chin
                                (-225.0, 170.0, -135.0),     # Left eye left corner
                                (225.0, 170.0, -135.0),      # Right eye right corne
                                (-150.0, -150.0, -125.0),    # Left Mouth corner
                                (150.0, -150.0, -125.0)      # Right mouth corner                     
                            ])

    # Camera internals
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
                             [[focal_length, 0, center[0]],
                             [0, focal_length, center[1]],
                             [0, 0, 1]], dtype = "double"
                             )
    #print ("Camera Matrix :\n {0}".format(camera_matrix))

    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
     
    # Project a 3D point (0, 0, 1000.0) onto the image plane.
    # We use this to draw a line sticking out of the nose
    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
     
    for p in image_points:
        cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0,0,255), -1)

    p1 = ( int(image_points[0][0]), int(image_points[0][1]))
    p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
    cv2.line(frame, p1, p2, (255,0,0), 2)
    return p1,p2
     


def notification(value):
    if value:
        print("sending notification")
        url = 'http://blynk-cloud.com/_eTX4XVZPY8CIglkW18kKzaA1kokX47n/notify'   # paste ur token here and add notification widget in the blynk app
        body = {'body': 'Alert Driver is sleepy'}    #msg u want as notification
        headers = {'content-type': 'application/json'}
        r = requests.post(url, data=json.dumps(body), headers=headers)

def mouth_aspect_ratio(mouth):
    # Compute the euclidean distances between the three sets
    # of vertical mouth landmarks (x, y)-coordinates
    A = np.linalg.norm(mouth[13] - mouth[19])
    B = np.linalg.norm(mouth[14] - mouth[18])
    C = np.linalg.norm(mouth[15] - mouth[17])

    # Compute the euclidean distance between the horizontal
    # mouth landmarks (x, y)-coordinates
    D = np.linalg.norm(mouth[12] - mouth[16])

    # Compute the mouth aspect ratio
    mar = (A + B + C) / (2 * D)

    # Return the mouth aspect ratio
    return mar


def sound_alarm(path):
        # play an alarm sound
        playsound.playsound(path)
        
def eye_aspect_ratio(eye):
        # compute the euclidean distances between the two sets of
        # vertical eye landmarks (x, y)-coordinates
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])

        # compute the euclidean distance between the horizontal
        # eye landmark (x, y)-coordinates
        C = dist.euclidean(eye[0], eye[3])

        # compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)

        # return the eye aspect ratio
        return ear
 
# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold for to set off the
# alarm
EYE_AR_THRESH = 0.27
EYE_AR_CONSEC_FRAMES = 48
MOUTH_AR_THRESH = 0.2
MOUTH_AR_CONSECUTIVE_FRAMES = 15
YAWN_COUNT = 0
# initialize the frame counter as well as a boolean used to
# indicate if the alarm is going off
COUNTER = 0
ALARM_ON = False
get_pose = False
MOUTH_COUNTER = 0
YELLOW_COLOR = (0, 255, 255)

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
# start the video stream thread
print("[INFO] Starting Video")
vs = VideoStream(0).start()
time.sleep(1.0)

def audioalert():
    global ALARM_ON
    # if the alarm is not on, turn it on
    if not ALARM_ON:
            ALARM_ON = True
            #notification(True)
            
            # check to see if an alarm file was supplied,
            # and if so, start a thread to have the alarm
            # sound played in the background
            if "alarm.wav" != "":
                    t = Thread(target=sound_alarm,
                            args=("alarm.wav",))
                    t.deamon = True
                    t.start()

    # draw an alarm on the frame
    #notification(True)
    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# loop over frames from the video stream
while True:
        # grab the frame from the threaded video file stream, resize
        # it, and convert it to grayscale
        # channels)
        frame = vs.read()
        size = frame.shape
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale frame
        rects = detector(gray, 0)
        if(len(rects) < 1):
            cv2.putText(frame, "Alert! Look at Camera", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # loop over the face detections
        for rect in rects:
                # determine the facial landmarks for the face region, then
                # convert the facial landmark (x, y)-coordinates to a NumPy
                # array
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                # extract the left and right eye coordinates, then use the
                # coordinates to compute the eye aspect ratio for both eyes
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                mouth = shape[mStart:mEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                mar = mouth_aspect_ratio(mouth)
                

                # average the eye aspect ratio together for both eyes
                ear = (leftEAR + rightEAR) / 2.0

                # compute the convex hull for the left and right eye, then
                # visualize each of the eyes and mouth
                mouthHull = cv2.convexHull(mouth)
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [mouthHull], -1, YELLOW_COLOR, 1)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                # check to see if the eye and mouth aspect ratio is below the 
                # threshold, and if so, increment the blink frame counter
                if mar > MOUTH_AR_THRESH:
                        MOUTH_COUNTER += 1
                        #print(MOUTH_COUNTER)
                        if MOUTH_COUNTER >= MOUTH_AR_CONSECUTIVE_FRAMES:
                                YAWN_COUNT += 1
                                MOUTH_COUNTER = 0
                                if YAWN_COUNT > 5:
                                    audioalert()
                                    
                else:
                        MOUTH_COUNTER = 0
                        notification(False)
                        
                if ear < EYE_AR_THRESH:
                        COUNTER += 1

                        # if the eyes were closed for a sufficient number of frames
                        # then sound the alarm
                        if COUNTER >= EYE_AR_CONSEC_FRAMES:
                            audioalert()
            
                # otherwise, the eye aspect ratio is not below the blink
                # threshold, so reset the counter and alarm
                else:
                        COUNTER = 0
                        ALARM_ON = False
                        notification(False)
                if get_pose == True:
                    p1,p2 = getheadpose(frame,shape,size)
                    #print("pitch" + str(p1[0]) + " yaw" +  str(p2[0]))
                    pitch = p1[0]
                    look = p2[1]
                    if pitch >150 and pitch<200:
                        cv2.putText(frame, "looking right".format(ear), (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    elif pitch >270 and pitch <300:
                        cv2.putText(frame, "looking left".format(ear), (30, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if look > 270 and look < 300:
                        cv2.putText(frame, "looking down".format(ear), (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    elif look < 100:
                        cv2.putText(frame, "looking up".format(ear), (30, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)                    
                        
                   
                    
                # draw the computed eye aspect ratio on the frame to help
                # with debugging and setting the correct eye aspect ratio
                # thresholds and frame counters
                cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "YAWN COUNT:{}".format(YAWN_COUNT), (270, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "Press 'h' to Start Head Pose", (20, 280),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "Press 'q' to Quit", (20, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
                break
        if key == ord("h"):
                get_pose = not get_pose

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
print("[INFO] Cleaning all")
print("[INFO] Closed")
