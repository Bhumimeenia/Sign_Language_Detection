import os
import pickle
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Updated DATA_DIR with absolute path
DATA_DIR = 'C:\\Users\\SMILE\\OneDrive\\Desktop\\aiml\\32_sign_language_detection\\code\\data'

data = []
labels = []

print("Current Working Directory:", os.getcwd())

# Iterate over directories in DATA_DIR
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    
    # Check if the current path is a directory
    if os.path.isdir(dir_path):
        for img_path in os.listdir(dir_path):
            img_full_path = os.path.join(dir_path, img_path)
            print(f"Processing image: {img_full_path}")
            
            # Check if image is loaded properly
            img = cv2.imread(img_full_path)
            if img is None:
                print(f"Failed to load image: {img_full_path}")
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
                data_aux = []
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x)
                        data_aux.append(y)

                data.append(data_aux)
                labels.append(dir_)

# Save the data to a pickle file in the same folder
pickle_file_path = os.path.join(DATA_DIR, '..', 'data.pickle')
print("Saving data to:", pickle_file_path)

with open(pickle_file_path, 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Data saved successfully.")


# import os
# import pickle

# import mediapipe as mp
# import cv2
# import matplotlib.pyplot as plt

# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles

# hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# DATA_DIR = 'C:\\Users\\SMILE\\OneDrive\\Desktop\\aiml\\32_sign_language_detection\\code\\data'

# data = []
# labels = []

# print("Current Working Directory:", os.getcwd())

# for dir_ in os.listdir(DATA_DIR):
#     for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
#         data_aux = []
#         img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#         results = hands.process(img_rgb)
#         if results.multi_hand_landmarks:
#           for hand_landmarks in results.multi_hand_landmarks:
#               for i in range(len(hand_landmarks.landmark)):
#                   x = hand_landmarks.landmark[i].x
#                   y = hand_landmarks.landmark[i].y
#                   data_aux.append(x)
#                   data_aux.append(y)

#           data.append(data_aux)   
#           labels.append(dir_) 

# pickle_file_path = os.path.join(os.getcwd(), 'data.pickle')
# print("Saving data to:", pickle_file_path)

# f = open('data.pickle', 'wb')
# pickle.dump({'data': data, 'labels': labels}, f)
# f.close()

#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:



#                     x_.append(x)
#                     y_.append(y)

#                 for i in range(len(hand_landmarks.landmark)):
#                     x = hand_landmarks.landmark[i].x
#                     y = hand_landmarks.landmark[i].y
#                     data_aux.append(x - min(x_))
#                     data_aux.append(y - min(y_))






# landmarks drawing
# mp_drawing.draw_landmarks(
#             img_rgb,
#             hand_landmarks,
#             mp_hands.HAND_CONNECTIONS,
#             mp_drawing_styles.get_default_hand_landmarks_style(),
#             mp_drawing_styles.get_default_hand_connections_style() 
#         )