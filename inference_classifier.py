# import pickle

# import cv2
# import mediapipe as mp 
# import numpy as np

# model_dict = pickle.load(open('./model.p', 'rb'))
# model = model_dict['model']

# cap = cv2.VideoCapture(0)

# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles

# hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# while True:
  
#   data_aux = []

#   ret, frame = cap.read()

#   frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#   results = hands.process(frame_rgb)
#   if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#           mp_drawing.draw_landmarks(
#                 frame,  # image to draw
#                 hand_landmarks,  # model output
#                 mp_hands.HAND_CONNECTIONS,  # hand connections
#                 mp_drawing_styles.get_default_hand_landmarks_style(),
#                 mp_drawing_styles.get_default_hand_connections_style())
          
#         for hand_landmarks in results.multi_hand_landmarks:
#             for i in range(len(hand_landmarks.landmark)):
#                 x = hand_landmarks.landmark[i].x
#                 y = hand_landmarks.landmark[i].y

#                 data_aux.append(x)
#                 data_aux.append(y)

#         model.predict([np.asarray(data_aux)])

#   cv2.imshow('frame',frame)
#   cv2.waitKey(25)



# cap.release()
# cv2.destroyAllWindows()



# import cv2


# cap = cv2.VideoCapture(0)

# while True:
#   ret, frame = cap.read()

#   cv2.imshow('frame',frame)
#   cv2.waitKey(25)




# video
# import pickle

# import cv2
# import mediapipe as mp
# import numpy as np

# model_dict = pickle.load(open('./model.p', 'rb'))
# # model_dict = pickle.load(open('./data/model.p', 'rb'))

# model = model_dict['model']

# cap = cv2.VideoCapture(2)

# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles

# hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# labels_dict = {0: 'A', 1: 'B', 2: 'L'}
# while True:

#     data_aux = []
#     x_ = []
#     y_ = []

#     ret, frame = cap.read()

#     H, W, _ = frame.shape

#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     results = hands.process(frame_rgb)
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(
#                 frame,  # image to draw
#                 hand_landmarks,  # model output
#                 mp_hands.HAND_CONNECTIONS,  # hand connections
#                 mp_drawing_styles.get_default_hand_landmarks_style(),
#                 mp_drawing_styles.get_default_hand_connections_style())

#         for hand_landmarks in results.multi_hand_landmarks:
#             for i in range(len(hand_landmarks.landmark)):
#                 x = hand_landmarks.landmark[i].x
#                 y = hand_landmarks.landmark[i].y

#                 x_.append(x)
#                 y_.append(y)

#             for i in range(len(hand_landmarks.landmark)):
#                 x = hand_landmarks.landmark[i].x
#                 y = hand_landmarks.landmark[i].y
#                 data_aux.append(x - min(x_))
#                 data_aux.append(y - min(y_))

#         x1 = int(min(x_) * W) - 10
#         y1 = int(min(y_) * H) - 10

#         x2 = int(max(x_) * W) - 10
#         y2 = int(max(y_) * H) - 10

#         prediction = model.predict([np.asarray(data_aux)])

#         predicted_character = labels_dict[int(prediction[0])]

#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
#         cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
#                     cv2.LINE_AA)

#     cv2.imshow('frame', frame)
#     cv2.waitKey(1)


# cap.release()
# cv2.destroyAllWindows()



# also show
# import pickle
# import cv2
# import mediapipe as mp
# import numpy as np


# model_dict = pickle.load(open('./model.p', 'rb'))
# model = model_dict['model']

# # Initialize MediaPipe Hands module
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles

# # Set up the video capture
# cap = cv2.VideoCapture(0)

# # Check if the camera opened successfully
# if not cap.isOpened():
#     print("Error: Could not open video.")
#     exit()

# labels_dict = {0: 'A', 1: 'B', 2: 'L'}
# # Initialize the hands detection model
# with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
#     while True:
#         # Read a frame from the camera
#         data_aux = []
#         ret, frame = cap.read()

#         # Check if the frame was captured correctly
#         if not ret:
#             print("Error: Could not read frame.")
#             break

#         # Flip the frame horizontally for a later selfie-view display
#         frame = cv2.flip(frame, 1)

#         # Convert the BGR frame to RGB
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
#         # Process the frame and find hands
#         results = hands.process(frame_rgb)

#         # Draw hand landmarks if any hands are found
#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 mp_drawing.draw_landmarks(
#                     frame,  # image to draw on
#                     hand_landmarks,  # hand landmarks output from the model
#                     mp_hands.HAND_CONNECTIONS,  # hand connections
#                     mp_drawing_styles.get_default_hand_landmarks_style(),
#                     mp_drawing_styles.get_default_hand_connections_style()
#                 )

#             for hand_landmarks in results.multi_hand_landmarks:
#               for i in range(len(hand_landmarks.landmark)):
#                 x = hand_landmarks.landmark[i].x
#                 y = hand_landmarks.landmark[i].y

#                 data_aux.append(x)
#                 data_aux.append(y)

#             prediction = model.predict([np.asarray(data_aux)])

#             predicted_character = labels_dict[int(prediction[0])]

#             print(predicted_character)

#         # Display the frame with landmarks
#         cv2.imshow('Hand Tracking', frame)

#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# # Release the camera and close all OpenCV windows
# cap.release()
# cv2.destroyAllWindows()




















# show letter in cmd
# import pickle
# import cv2
# import mediapipe as mp
# import numpy as np
# import warnings

# # Suppress warnings from the protobuf library
# warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

# # Load the trained model
# model_dict = pickle.load(open('./model.p', 'rb'))
# model = model_dict['model']

# # Initialize MediaPipe Hands module
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles

# # Set up the video capture
# cap = cv2.VideoCapture(0)

# # Check if the camera opened successfully
# if not cap.isOpened():
#     print("Error: Could not open video.")
#     exit()

# labels_dict = {0: 'A', 1: 'B', 2: 'L'}  # Ensure this matches your model's classes
# # Initialize the hands detection model
# with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
#     while True:
#         # Read a frame from the camera
#         data_aux = []
#         x_ = []
#         y_ = []

#         ret, frame = cap.read()

#         H, W, _ = frame.shape
#         # Check if the frame was captured correctly
#         if not ret:
#             print("Error: Could not read frame.")
#             break

#         # Flip the frame horizontally for a later selfie-view display
#         frame = cv2.flip(frame, 1)

#         # Convert the BGR frame to RGB
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
#         # Process the frame and find hands
#         results = hands.process(frame_rgb)

#         # Draw hand landmarks if any hands are found
#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 mp_drawing.draw_landmarks(
#                     frame,  # image to draw on
#                     hand_landmarks,  # hand landmarks output from the model
#                     mp_hands.HAND_CONNECTIONS,  # hand connections
#                     mp_drawing_styles.get_default_hand_landmarks_style(),
#                     mp_drawing_styles.get_default_hand_connections_style()
#                 )

#                 # Collect landmarks data (only one hand is processed)
#                 for i in range(len(hand_landmarks.landmark)):
#                     x = hand_landmarks.landmark[i].x
#                     y = hand_landmarks.landmark[i].y

#                     data_aux.append(x)
#                     data_aux.append(y)
#                     x_.append(x)
#                     y_.append(y)

#                 # Ensure the data_aux contains exactly 84 features (42 landmarks x 2)
#                 if len(data_aux) < 84:  # Adjust based on your model input shape
#                     data_aux.extend([0] * (84 - len(data_aux)))  # Fill with zeros if not enough
#                 else:
#                     data_aux = data_aux[:84]  # Truncate if too many

#                 # Predict the character using only the required features
#                 prediction = model.predict([np.asarray(data_aux[:42])])  # Adjust as per your model

#                 predicted_character = labels_dict[int(prediction[0])]

#                 print(predicted_character)  # Print the predicted character

#         # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
#         # cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
        
#         # Display the frame with landmarks
#         cv2.imshow('Hand Tracking', frame)

#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# # Release the camera and close all OpenCV windows
# cap.release()
# cv2.destroyAllWindows()






# frame
import pickle
import cv2
import mediapipe as mp
# video analysis hand detection
import numpy as np  
#calculation and to handle landmarks
import warnings

# Suppress warnings from the protobuf library
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands

# Set up the video capture
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Label dictionary for your model's classes
labels_dict = {0: 'A', 1: 'B', 2: 'L'}  # Adjust according to your trained model

# Initialize the hands detection model
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while True:
        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()

        # Check if the frame was captured correctly
        if not ret:
            print("Error: Could not read frame.")
            break

        # Flip the frame horizontally for selfie-view display
        frame = cv2.flip(frame, 1)

        # Convert the BGR frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and find hands
        results = hands.process(frame_rgb)

        # Initialize the variable to store the predicted character
        predicted_character = None

        # If any hands are found
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Collect landmarks data (only one hand is processed)
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)
                    x_.append(x)
                    y_.append(y)

                # Ensure the data_aux contains exactly 84 features (42 landmarks x 2)
                if len(data_aux) < 84:  # Adjust based on your model input shape
                    data_aux.extend([0] * (84 - len(data_aux)))  # Fill with zeros if not enough
                else:
                    data_aux = data_aux[:84]  # Truncate if too many

                # Predict the character using only the required features
                prediction = model.predict([np.asarray(data_aux[:42])])  # Adjust as per your model
                predicted_character = labels_dict[int(prediction[0])]  # Predicted character

                # Display the predicted character on the frame
                cv2.putText(
                    frame, 
                    f'Prediction: {predicted_character}', 
                    (10, 40),  # Position of the text
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1.3,  # Font scale
                    (0, 255, 0),  # Color: Green
                    3,  # Thickness
                    cv2.LINE_AA
                )

        # Display the frame with predicted character only
        cv2.imshow('Sign Language Prediction', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
