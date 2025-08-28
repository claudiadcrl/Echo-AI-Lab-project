import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd
import soundfile as sf
from pedalboard import Pedalboard, Chorus, Gain, Compressor, Bitcrush, Distortion
import threading
import gui

#Global variables
effects_lock = threading.Lock()
active_effects = {}

#to initialize new effects in board when added
default_effects={
    "gain": Gain(),
    "bitcrush": Bitcrush(),
    "distortion": Distortion(),
    "chorus": Chorus(depth=0.5),
    "compressor": Compressor(ratio=2.0)
}

board = Pedalboard([]) #board starts empty, to avoid noise

# ----------------------------------------------------------------- AUDIO ------------------------------------------------------------------------------------------

AUDIO_FILES = [
    'Tracks\\Alesis-Fusion-Bass-Loop.wav', #default at start
    'Tracks\\-1000-Handz-36-Chambers.wav',
    'Tracks\\Pawel-Spychala-Crash-in-space.wav',
    'Tracks\\Pawel-Spychala-Conscience.wav',
    'Tracks\\Tebo-Steele-The-Skies.wav',
    'Tracks\\Alesis-Fusion-Bass-Loop.wav'
]
loaded_tracks = [sf.read(file, always_2d=True) for file in AUDIO_FILES]

data, samplerate = loaded_tracks[0]
loop_position = 0
blocksize = 1024
current_track_index = 0
track_lock = threading.Lock()

#for GUI
root = gui.tk.Tk()
gui = gui.AudioVisualizerGUI(root, active_effects, AUDIO_FILES)

def switch_track(index):
    global data, samplerate, loop_position, current_track_index
    with track_lock:
        #when track changes, according to left hand, load data of new audio file
        if index!=current_track_index:
            if index!=0:
                current_track_index = index
                data, samplerate = loaded_tracks[index]
                loop_position = 0
            else:
                # stop playback → tell thread to exit
                current_track_index = 0
                data = None

def audio_callback(outdata, frames, time, status):
    global loop_position, board

    if data is None:
        #Output silence when stopped
        outdata[:] = np.zeros((frames, 2), dtype=np.float32)
        return

    #Select chunk to process (even in case needs to loop)
    end_position = loop_position + frames
    if end_position >= len(data):
        chunk = np.vstack([
            data[loop_position:],
            data[:end_position % len(data)]
        ])
        loop_position = end_position % len(data)
    else:
        chunk = data[loop_position:end_position]
        loop_position = end_position

    #update the effects in the board, but only "activated" ones
    with effects_lock:
        board = Pedalboard([])

        for effect in active_effects:
            board.append(active_effects[effect])

    outdata[:] = board(chunk, samplerate)

def start_audio():
    with sd.OutputStream(channels=2, callback=audio_callback,
                         blocksize=blocksize, samplerate=samplerate):
        threading.Event().wait()
        

audio_thread = threading.Thread(target=start_audio, daemon=True)
audio_thread.start()
# ----------------------------------------------------------------- VISUAL ------------------------------------------------------------------------------------------

joint_list = [[4,2,1], [8,6,5], [12,10,9], [16,14,13], [20,18,17]]
up = [0,0,0,0,0]
prev_counts=[]

#FUNCTIONS

def update_effects():
    with effects_lock:
        if rot_deg > 180: return
        #checks right hand fingers movement, activates effect and updates it
        match up:
            case [1,0,0,0,0]:
                if "gain" not in active_effects:
                    active_effects["gain"] = Gain()
                active_effects["gain"].gain_db = rot_norm * 20

            case [1,1,0,0,0]:
                if "bitcrush" not in active_effects:
                    active_effects["bitcrush"] = Bitcrush()
                active_effects["bitcrush"].bit_depth  = 24 - ((rot_norm + 1) / 2) * 20

            case [1,1,1,0,0]:
                if "distortion" not in active_effects:
                    active_effects["distortion"] = Distortion()
                active_effects["distortion"].drive_db = max(0, rot_norm) * 40

            case [1,1,1,1,0]:
                if "chorus" not in active_effects:
                    active_effects["chorus"] = Chorus(depth=0.5)
                active_effects["chorus"].rate_hz = ((rot_norm + 1) / 2) * 5

            case [1,1,1,1,1]:
                if "compressor" not in active_effects:
                    active_effects["compressor"] = Compressor(ratio=2.0)
                active_effects["compressor"].threshold_db = -((rot_norm + 1) / 2) * 40

def majority(lista):
    for x in lista:
        if lista.count(x) >= 4:
            return True
    return False

def draw_finger_angles(image, landmrk, label, joint_list):
    #Loop through joint sets 
    count = 0
    for i,joint in enumerate(joint_list):
            a = np.array([landmrk[joint[0]].x, landmrk[joint[0]].y]) # First coord
            b = np.array([landmrk[joint[1]].x, landmrk[joint[1]].y]) # Second coord
            c = np.array([landmrk[joint[2]].x, landmrk[joint[2]].y]) # Third coord
            
            radians = np.arctan2(c[1] - b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0]) #computes angle of finger to see if open
            angle = np.abs(radians*180.0/np.pi)
            cv2.putText(image, str(round(angle, 2)), tuple(np.multiply(b, [640, 480]).astype(int)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            if angle > 180.0:
                angle = 360-angle
            if label=="Right":
                if angle<160:
                    up[i]=0 #finger is closed
                else:
                    up[i]=1 #finger is open
            if label=="Left":
                if angle>160:
                    count+=1 #for left hand we have more freedom
    #switch tracks only if for last 4 out of 5 frames the count was the same --> stabilizes change
    if label=="Left":
        prev_counts.append(count)
        if len(prev_counts)>5:
            prev_counts.pop(0)
        if majority(prev_counts)==True:
            switch_track(count)
    return image

#Function to calculate the angle between two 2D vectors in radians
def get_angle(v1,v2):
    angle1=np.arctan2(v1[1],v1[0])
    angle2=np.arctan2(v2[1],v2[0])
    return angle2-angle1

#Function to get labels of hands and coords
def get_label(index, hand, results):
    output = None
    for idx, classification in enumerate(results.multi_handedness):
        if classification.classification[0].index == index:
            
            # Process results
            label = classification.classification[0].label
            score = classification.classification[0].score
            text = '{} {}'.format(label, round(score, 2))
            
            # Extract Coordinates
            coords = tuple(np.multiply(
                np.array((hand.landmark[mp_hands.HandLandmark.WRIST].x, hand.landmark[mp_hands.HandLandmark.WRIST].y)),
            [640,480]).astype(int))
            
            output = text, coords
    return output

#initialise hands and drawing tools
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

#gets camera and initialises vector and final rotation angle
cap=cv2.VideoCapture(0)
prev_vectors = [None, None]
rotation_sums = [0.0, 0.0]

with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.8,
                    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        #Rendering results
        if results.multi_hand_landmarks:
            for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                #draw the landmarks on the hand
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS, 
                                       mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=4),
                                       mp_drawing.DrawingSpec(color=(216, 255, 87), thickness=2, circle_radius=2),
                )
                num=hand_handedness.classification[0].index
                if len(results.multi_hand_landmarks)>1:
                    #Set anchor point and compute rotation angle
                    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                    anchor = landmarks[0][:2] #Wrist
                    index_tip = landmarks[8][:2]  #Index finger tip
                    curr_vector = index_tip - anchor
                    
                    if len(prev_vectors)>=num+1:
                        if prev_vectors[num] is not None:
                            angle = get_angle(prev_vectors[num], curr_vector)
                            rotation_sums[num] += angle

                            total_deg = np.degrees(rotation_sums[1])

                        prev_vectors[num] = curr_vector
                    # Render left or right detection
                    if get_label(num, hand_landmarks, results):
                        text, coord = get_label(num, hand_landmarks, results)
                        cv2.putText(image, text, coord, cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2, cv2.LINE_AA)
                    # Draw angles to image from joint list
                    label = hand_handedness.classification[0].label
                    draw_finger_angles(image, hand_landmarks.landmark, label, joint_list)

            rot_deg = np.degrees(rotation_sums[1])
            rot_clamped = np.clip(rot_deg, -60, 60)  #to ease movements
            rot_norm = rot_clamped / 60.0

            update_effects()

            gui.update(rot_clamped, current_track_index)
            root.update_idletasks()
            root.update()

        cv2.imshow('Hand Tracking', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()