import copy
import itertools
import tensorflow as tf

import cv2 as cv
import numpy as np
import mediapipe as mp

class HandPointsRecognizer(object):
    def __init__(self, model_path='hand_points.tflite', num_threads=1):
        # Initialize the HandPointsRecognizer object
        self.interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def classify_gesture(self, landmark_list):
        # Classify hand gesture using TensorFlow Lite model
        input_details_tensor_index = self.input_details[0]['index']
        self.interpreter.set_tensor(input_details_tensor_index, np.array([landmark_list], dtype=np.float32))
        self.interpreter.invoke()
        output_details_tensor_index = self.output_details[0]['index']
        result = self.interpreter.get_tensor(output_details_tensor_index)
        result_index = np.argmax(np.squeeze(result))
        return result_index


def preprocess_landmarks(landmark_list):
    # Deep copy the landmark list
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Normalize and flatten the landmark list
    temp_landmark_list = [[x - temp_landmark_list[0][0], y - temp_landmark_list[0][1]] for x, y in temp_landmark_list]
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(map(abs, temp_landmark_list))
    temp_landmark_list = [n / max_value for n in temp_landmark_list]

    return temp_landmark_list


def calculate_relative_coordinates(landmark_list):
    # Calculate relative coordinates of landmarks based on the first landmark
    base_x, base_y = landmark_list[0][0], landmark_list[0][1]
    for index, (x, y) in enumerate(landmark_list):
        landmark_list[index] = [x - base_x, y - base_y]
    return landmark_list


def flatten_list(landmark_list):
    # Flatten the 2D landmark list to a 1D list
    return list(itertools.chain.from_iterable(landmark_list))


def normalize_landmarks(landmark_list):
    # Normalize landmark coordinates
    max_value = max(map(abs, landmark_list))
    return [n / max_value for n in landmark_list]


def calculate_landmark_points(image, landmarks):
    # Calculate landmark points from detected landmarks on an image
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_points = [[min(int(landmark.x * image_width), image_width - 1),
                        min(int(landmark.y * image_height), image_height - 1)]
                       for landmark in landmarks.landmark]
    return landmark_points

def draw_detected_landmarks(image, landmark_points):
    # Draw hand landmarks on the image, including fingers, palm, and hand points
    if len(landmark_points) > 0:
        fingers = [[2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]
        draw_fingers(image, landmark_points, fingers)
        
        palm_points = [0, 1, 2, 5, 9, 13, 17]
        draw_palm(image, landmark_points, palm_points)

    draw_hand_points(image, landmark_points)
    return image


def draw_fingers(image, landmark_points, fingers):
    # Draw fingers on the image
    for finger_points in fingers:
        for i in range(len(finger_points) - 1):
            # Draw lines for fingers
            cv.line(image, (landmark_points[finger_points[i]]), (landmark_points[finger_points[i + 1]]), (0, 0, 0), 7)
            cv.line(image, (landmark_points[finger_points[i]]), (landmark_points[finger_points[i + 1]]), (255, 255, 255), 2)


def draw_palm(image, landmark_points, palm_points):
    # Draw palm on the image
    for i in range(len(palm_points) - 1):
        # Draw lines for palm
        cv.line(image, (landmark_points[palm_points[i]]), (landmark_points[palm_points[i + 1]]), (0, 0, 0), 7)
        cv.line(image, (landmark_points[palm_points[i]]), (landmark_points[palm_points[i + 1]]), (255, 255, 255), 2)
    
    # Connect last and first palm points
    cv.line(image, (landmark_points[palm_points[-1]]), (landmark_points[palm_points[0]]), (0, 0, 0), 7)
    cv.line(image, (landmark_points[palm_points[-1]]), (landmark_points[palm_points[0]]), (255, 255, 255), 2)


def draw_hand_points(image, landmark_points):
    # Draw hand points on the image
    for index, landmark in enumerate(landmark_points):
        radius = 8 if index in [4, 8, 12, 16, 20] else 5
        cv.circle(image, (landmark[0], landmark[1]), radius, (255, 255, 255), -1)
        cv.circle(image, (landmark[0], landmark[1]), radius, (0, 0, 0), 1)


def calculate_bounding_rect(image, landmarks):
    # Calculate the bounding rectangle around hand landmarks
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point = [np.array((landmark_x, landmark_y))]
        landmark_array = np.append(landmark_array, landmark_point, axis=0)
    x, y, w, h = cv.boundingRect(landmark_array)
    return [x, y, x + w, y + h]


def draw_information_text(image, bounding_rect, handedness, gesture_text):
    # Draw information text on the image, including hand classification and additional sign text
    cv.rectangle(image, (bounding_rect[0], bounding_rect[1]), (bounding_rect[2], bounding_rect[1] - 22), (0, 0, 0), -1)
    info_text = handedness.classification[0].label[0:]
    
    if gesture_text != "":
        info_text = info_text + ':' + gesture_text

    cv.putText(image, info_text, (bounding_rect[0] + 5, bounding_rect[1] - 4), cv.FONT_ITALIC, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    
    return image



def main():
    camera_index = 0
    screen_width = 960
    screen_height = 540

    cap = cv.VideoCapture(camera_index)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, screen_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, screen_height)

    mp_hands = mp.solutions.hands
    hands_detector = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.6)
    hand_points_recognizer = HandPointsRecognizer()
    gesture_labels = ["Open", "Close", "Pointer", "OK"]

    while True:
        key = cv.waitKey(10)
        if key == 113:  # 'q' key
            break
        if key == 27:  # 'Esc' key
            break

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv.flip(frame, 1)
        copied_frame = copy.deepcopy(frame)
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame.flags.writeable = False
        results = hands_detector.process(frame)
        frame.flags.writeable = True

        if results.multi_hand_landmarks != None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                bounding_rectangle = calculate_bounding_rect(copied_frame, hand_landmarks)
                
                landmark_points = calculate_landmark_points(copied_frame, hand_landmarks)
                
                preprocessed_landmark_points = preprocess_landmarks(landmark_points)
                
                gesture_id = hand_points_recognizer.classify_gesture(preprocessed_landmark_points)
                
                copied_frame = draw_detected_landmarks(copied_frame, landmark_points)
                
                copied_frame = draw_information_text(copied_frame, bounding_rectangle, handedness, gesture_labels[gesture_id])

        cv.imshow('Hand Gesture Recognition', copied_frame)
    cap.release()
    cv.destroyAllWindows()

main()
