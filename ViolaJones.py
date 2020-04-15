import cv2
import os

def vj_matcher(image_path, i):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    line_width = 3
    face_color = (0, 0, 255)
    eyes_color = (0, 255, 0)
    scale_factor = 1.3
    min_neighbors = 5
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scale_factor, min_neighbors)
    for x, y, w, h in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), face_color, line_width)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for ex, ey, ew, eh in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), eyes_color, line_width)

    cv2.imshow('img', img)
    cv2.imwrite(os.path.join('results', f'{i}.jpg'), img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    file_directory = 'photos'
    images = [os.path.join(file_directory, f'{i}.jpg') for i in range(1, 11)]
    print(f'detecting {images[0]}')
    vj_matcher(images[0], 0)