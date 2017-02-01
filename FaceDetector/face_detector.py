import cv2
import optparse
import os


def detect_faces(classifier, img_frame):
    """
    Function that retreives face rectangles from image.

    Input
    ====
    classifier: obj.
    img_frame: numpy array.

    Output
    ======
    faces_coords: List of tuples.

    """
    faces_coords = classifier.detectMultiScale(
        img_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    return faces_coords


def draw_rectangle(img_frame, coords):
    """
    Renders rectangles given coordinates and image frame.

    Input
    =====
    img_frame: numpy array
    coords: Tuple with x1, y1, x2, y2 coordinates

    """
    x_point, y_point, width, height = coords
    cv2.rectangle(
        img_frame,
        (x_point, y_point),
        (x_point + width, y_point + height),
        (0, 255, 0),
        2
    )


def main():
    # Read input options.
    options = parse_options()
    img = cv2.imread(options.path)

    # Create Haar-cascade classifier
    classifier_path = 'Classifiers/haarcascade_frontalface_alt.xml'
    face_cascade = cv2.CascadeClassifier(classifier_path)

    # Find faces using Haar cascade classifier.
    faces = detect_faces(face_cascade, img)

    # Draw a rectangle around the faces
    for face in faces:
        draw_rectangle(img, face)

    # Display image.
    cv2.imshow('Image', img)
    cv2.waitKey(0)

    print 'Done'


def parse_options():
    """
    Parses parameters from command line.
    """
    parser = optparse.OptionParser()
    parser.add_option('-p',
                      '--path',
                      dest='path')

    options, _ = parser.parse_args()

    return options


if __name__ == '__main__':
    main()

