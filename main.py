import cv2
import numpy as np
import time
import os

def get_image_value(image, grid_scale) -> int:
    image = image/255

    h_sections = np.split(image, grid_scale, axis=0)
    sections = []
    for h_section in h_sections:
        sections.extend(np.split(h_section, grid_scale, axis=1))
    
    section_values = []
    for section in sections:
        section_values.append(np.sum(section))
    return section_values


def detect_movement(value1, value2, threshold):
    for val1, val2 in zip(value1, value2):
        if np.abs(val1 - val2) > threshold:
            return True
    return False


def draw_X(image):
    height = image.shape[0]
    width = image.shape[1]

    top_left = (0, height)
    top_right = (width, height)
    bottom_left = (0, 0)
    bottom_right = (width, 0)

    red = (0, 0, 255)
    thickness = 10

    cv2.line(image, top_left, bottom_right, red, thickness)
    cv2.line(image, bottom_left, top_right, red, thickness)

    return image


def save_image(image, idx):
    dir = 'detected_movement'
    if not os.path.isdir(dir):
        os.mkdir(dir)
    img_path = os.path.join(dir, str(idx) + '.png')
    cv2.imwrite(img_path, image)


def main():
    camera = cv2.VideoCapture(0)
    _, last_frame = camera.read()

    grid_scale = 16
    last_value = get_image_value(last_frame, grid_scale)

    threshold = 0.08 * last_frame.shape[0] * last_frame.shape[1] / (grid_scale ** 2)
    
    image_index = 0
    while(True):
        _, frame = camera.read()
        value = get_image_value(frame, grid_scale)

        if detect_movement(value, last_value, threshold):
            save_image(frame, image_index)
            image_index += 1
            print('MOVEMENT')
        else:
            frame = draw_X(frame)
            print('STILL')


        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        last_frame = frame
        last_value = value
        time.sleep(0.1)
    
    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()