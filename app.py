import pickle
import mediapipe as mp
import numpy as np
import cv2

# import gradio as gr
# import streamlit as st
import pandas as pd
import cv2
import streamlit as st
with open('emotion_detector.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Emotion Detection")
st.text("select the checkbox to run the app. \n\n")
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])
cap = cv2.VideoCapture(0)
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic





while run:
  with mp_holistic.Holistic(min_detection_confidence = 0.5, min_tracking_confidence= 0.5) as holistic:

    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # FRAME_WINDOW.image(image)
    #ret , frame = cap.read()

    #recolor - cv2 captures image in blue green red bgr format but mediapipe need image in rgb format
    image = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    results = holistic.process(image)
    #print(results.face_landmarks)

    #recolor image
    image = cv2.cvtColor(image , cv2.COLOR_RGB2BGR)
    image.flags.writeable = True
    
    #1. draw face landmark
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                                mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                )
        
    # 2. Right hand
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                )

    # 3. Left Hand
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                              mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                              )

    # 4. Pose Detections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                              mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                              )
    try:
          # Extract Pose landmarks
          pose = results.pose_landmarks.landmark
          pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
          
          # Extract Face landmarks
          face = results.face_landmarks.landmark
          face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())
          
          # Concate rows
          row = pose_row+face_row
          
          # # Append class name 
          # row.insert(0, class_name)
          
          # # Export to CSV
          # with open('coords.csv', mode='a', newline='') as f:
          #     csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
          #     csv_writer.writerow(row) 
          
          
          # make detection
          X = pd.DataFrame([row])
          body_language_class = model.predict(X)[0]
          body_language_prob = model.predict_proba(X)[0]
          print(body_language_class, body_language_prob)
          
          coords = tuple(np.multiply(
                            np.array(
                                (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
                                 results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
                        , [640,480]).astype(int))
          
          cv2.rectangle(image, 
                          (coords[0], coords[1]+5), 
                          (coords[0]+len(body_language_class)*20, coords[1]-30), 
                          (245, 117, 16), -1)
          cv2.putText(image, body_language_class, coords, 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
            # Get status box
          cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)
            
            # Display Class
          # Display Class
          cv2.putText(image, 'CLASS'
                        , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
          cv2.putText(image, body_language_class.split(' ')[0]
                        , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Display Probability
          cv2.putText(image, 'PROB'
                        , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
          cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)],2))
                        , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
          
    except:
          pass                
                      
                      
                      
                      
    
    FRAME_WINDOW.image(image)
else:
  cap.release()
  cv2.destroyAllWindows()
  st.write('Stopped')


