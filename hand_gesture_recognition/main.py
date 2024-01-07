import copy
import itertools
import tensorflow as tf

import cv2 as cv
import numpy as np
import mediapipe as mp


class HandPoints(object):
    def __init__(
        self,
        model_path='hand_points.tflite',
        num_threads=1):
        self.interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(self, landmark_list):
        input_details_tensor_index = self.input_details[0]['index']
        self.interpreter.set_tensor(input_details_tensor_index, np.array([landmark_list], dtype=np.float32))
        self.interpreter.invoke()
        output_details_tensor_index = self.output_details[0]['index']
        result = self.interpreter.get_tensor(output_details_tensor_index)
        result_index = np.argmax(np.squeeze(result))
        return result_index




def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    #Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    #Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    #Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    #hand point
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        #landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        #Define finger landmarks
        fingers = [[2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]

        #Draw fingers
        for finger_points in fingers:
            for i in range(len(finger_points) - 1):
                cv.line(image, tuple(landmark_point[finger_points[i]]), tuple(landmark_point[finger_points[i + 1]]), (0, 0, 0), 7)
                cv.line(image, tuple(landmark_point[finger_points[i]]), tuple(landmark_point[finger_points[i + 1]]), (255, 255, 255), 2)

        #Draw palm
        palm_points = [0, 1, 2, 5, 9, 13, 17]
        for i in range(len(palm_points) - 1):
            cv.line(image, tuple(landmark_point[palm_points[i]]), tuple(landmark_point[palm_points[i + 1]]), (0, 0, 0), 7)
            cv.line(image, tuple(landmark_point[palm_points[i]]), tuple(landmark_point[palm_points[i + 1]]), (255, 255, 255), 2)
        
        #Connect last and first palm points
        cv.line(image, tuple(landmark_point[palm_points[-1]]), tuple(landmark_point[palm_points[0]]), (0, 0, 0), 7)
        cv.line(image, tuple(landmark_point[palm_points[-1]]), tuple(landmark_point[palm_points[0]]), (255, 255, 255), 2)


    #hand points
    for index, landmark in enumerate(landmark_point):
        radius = 8 if index in [4, 8, 12, 16, 20] else 5
        cv.circle(image, (landmark[0], landmark[1]), radius, (255, 255, 255), -1)
        cv.circle(image, (landmark[0], landmark[1]), radius, (0, 0, 0), 1)

    return image


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def draw_info_text(image, brect, handedness, hand_sign_text):

    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)
    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_ITALIC, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    return image




def main():

    camera = 0
    screen_width = 960
    screen_height = 540

    #camra prep 
    cap = cv.VideoCapture(camera)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, screen_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, screen_height)

    #load netwrok
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.6)
    hand_points = HandPoints()
    hand_points_labels = ["Open", "Close", "Pointer", "OK"]



    while True:
        #end program
        key = cv.waitKey(10)
        if key == 113:
            break
        if key == 27: 
            break

        #camera setup 
        ret, image = cap.read()
        if not ret:
            break

        image = cv.flip(image, 1)
        copy_image = copy.deepcopy(image)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                #Bounding box calculation
                brect = calc_bounding_rect(copy_image, hand_landmarks)
                #Landmark calculation
                landmark_list = calc_landmark_list(copy_image, hand_landmarks)
                #Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                #Hand sign classification
                hand_sign_id = hand_points(pre_processed_landmark_list)
                #Drawing part
                copy_image = draw_landmarks(copy_image, landmark_list)
                copy_image = draw_info_text(copy_image, brect, handedness, hand_points_labels[hand_sign_id])

        cv.imshow('Hand Gesture Recognition', copy_image)
    cap.release()
    cv.destroyAllWindows()


main()