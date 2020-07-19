import cv2
import numpy


def contour(image):
    image_afrer_contour = image.copy()
    shapes = {
        0: "Circle",
        1: "Dot",
        2: "line",
        3: "Triangle",
        4: "Square",
        5: "Pentagon",
        6: "Hexagon",
        7: "Heptagon",
        8: "Octagon",
        9: "Nonagon",
        10: "Decagon",
        11: "Hendecagon",
        12: "Dodecagon",
        13: "Tridecagon",
        14: "Tetradecagon",
        15: "Pentadecagon",
        16: "Hexadecagon",
        17: "Heptadecagon",
        18: "Octadecagon",
        19: "Enneadecagon",
        20: "Icosagon"
    }
    contours, hierarchy_param_NotInUze = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        if cv2.contourArea(contour) > 1000:
            cv2.drawContours(image_afrer_contour, contour, -1, (255, 200, 250), 6)

            contour_length = cv2.arcLength(contour, True)
            points_approximation = cv2.approxPolyDP(contour, 0.02 * contour_length, True)

            if shapes[len(points_approximation)]:
                print(shapes[len(points_approximation)] + '      ' + str(len(points_approximation)))

            a, b, c, h = cv2.boundingRect(points_approximation)

            cv2.putText(image_afrer_contour, "Points: " + str(len(points_approximation)), (a + c + 20, b + 20),
                        cv2.FONT_HERSHEY_COMPLEX, .7, (240, 255, 0), 2)
            cv2.putText(image_afrer_contour, "shape: " + shapes[len(points_approximation)], (a + c + 20, b + 50),
                        cv2.FONT_HERSHEY_COMPLEX, .7, (240, 255, 0), 2)

    return image_afrer_contour

def img_to_blur_and_grayscale(original_capture):
    gray_scale_capture = cv2.cvtColor(original_capture, cv2.COLOR_BGR2GRAY)
    blur_image = cv2.GaussianBlur(gray_scale_capture, (7, 7), 1)

    return blur_image, gray_scale_capture

def main():
    capture = cv2.VideoCapture(0)
    while True:
        ok, img_for_show = capture.read()
        blur_image, gray_image = img_to_blur_and_grayscale(img_for_show)

        #Identify edges in a grayscale image using a canny algorithm
        canny_image = cv2.Canny(blur_image, 23 , 20)

        # create 5X5 array of 1's
        array_for_dilation = numpy.ones((3, 3))

        #We will use the cv2.dilate function to emphasize the shape
        dilation_image = cv2.dilate(canny_image, array_for_dilation, 1)

        image_after_contour_filter = contour(dilation_image)

        #show image
        cv2.imshow('Shape Recognition', image_after_contour_filter)

        key = cv2.waitKey(1)
        #close window on ESC
        if key == 27:
            break


main()