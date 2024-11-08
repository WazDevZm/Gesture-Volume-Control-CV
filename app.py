import cv2
import mediapipe as mp
import numpy as np
import time
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Webcam setup
wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

# Initialize MediaPipe Hands and Drawing utils
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize the audio interface
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Get volume range for mapping
volRange = volume.GetVolumeRange()
minVol, maxVol = volRange[0], volRange[1]

# Volume Control variables
vol = 0
volBar = 400
volPer = 0

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while True:
        success, img = cap.read()
        if not success:
            break

        # Convert image color for MediaPipe
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        # Process the hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks and connections
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Get the coordinates of the thumb and index finger
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                
                # Convert to pixel coordinates
                x1, y1 = int(thumb_tip.x * wCam), int(thumb_tip.y * hCam)
                x2, y2 = int(index_tip.x * wCam), int(index_tip.y * hCam)

                # Draw circles on the thumb and index finger tips
                cv2.circle(img, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, (x2, y2), 10, (255, 0, 0), cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

                # Calculate the distance between the thumb and index finger
                length = np.hypot(x2 - x1, y2 - y1)

                # Map the distance to volume range
                vol = np.interp(length, [30, 300], [minVol, maxVol])
                volBar = np.interp(length, [30, 300], [400, 150])
                volPer = np.interp(length, [30, 300], [0, 100])

                # Set system volume
                volume.SetMasterVolumeLevel(vol, None)

                # Display volume bar and percentage
                cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
                cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)

        # Calculate and display FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

        # Display the output
        cv2.imshow("Volume Control", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
