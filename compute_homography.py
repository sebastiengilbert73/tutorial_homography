import logging
import cv2
import numpy as np
import copy

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s \t%(message)s')

def main():
    logging.info("compute_homography.py main()")
    image_filepath = "./images/checkerboard.png"
    features_mm_to_pixels_dict =  {(0, 0): (192, 801, ),
                                  (214.2, 0): (937, 652),
                                  (214.2, 178.5): (798, 67),
                                  (0, 178.5): (101, 197)}
    A = np.zeros((2 * len(features_mm_to_pixels_dict), 6), dtype=float)
    b = np.zeros((2 * len(features_mm_to_pixels_dict), 1), dtype=float)
    index = 0
    for XY, xy in features_mm_to_pixels_dict.items():
        X = XY[0]
        Y = XY[1]
        x = xy[0]
        y = xy[1]
        A[2 * index, 0] = x
        A[2 * index, 1] = y
        A[2 * index, 2] = 1
        A[2 * index + 1, 3] = x
        A[2 * index + 1, 4] = y
        A[2 * index + 1, 5] = 1
        b[2 * index, 0] = X
        b[2 * index + 1, 0] = Y
        index += 1
    # A @ x = b
    x, residuals, rank, singular_values = np.linalg.lstsq(A, b, rcond=None)

    pixels_to_mm_transformation_mtx = np.array([[x[0, 0], x[1, 0], x[2, 0]], [x[3, 0], x[4, 0], x[5, 0]], [0, 0, 1]])
    logging.debug("main(): pixels_to_mm_transformation_mtx = \n{}".format(pixels_to_mm_transformation_mtx))

    test_xy_1 = (399, 500, 1)
    test_XY_1 = pixels_to_mm_transformation_mtx @ test_xy_1
    logging.info("main(): test_xy_1 = {}; test_XY_1 = {}".format(test_xy_1, test_XY_1))

    mm_to_pixels_transformation_mtx = np.linalg.inv(pixels_to_mm_transformation_mtx)
    test_XY_2 = (107.1, 107.1, 1)
    test_xy_2 = mm_to_pixels_transformation_mtx @ test_XY_2
    image = cv2.imread(image_filepath)
    annotated_img = copy.deepcopy(image)
    draw_crosshair(annotated_img, (round(test_xy_2[0]), round(test_xy_2[1])), 20, (0, 255, 0))
    cv2.imwrite("./outputs/annotated.png", annotated_img)


def draw_crosshair(image, center, width, color):
    cv2.line(image, (center[0] - width//2, center[1]), (center[0] + width//2, center[1]), color)
    cv2.line(image, (center[0], center[1] - width//2), (center[0], center[1] + width//2), color)

if __name__ == '__main__':
    main()