import logging
import cv2
import numpy as np
import copy

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s \t%(message)s')

def main():
    logging.info("compute_homography.py main()")
    image_filepath = "./images/white_square.png"
    corners_cm_to_pixels_dict =  {(4, 2): (1626, 2310, ),
                                  (4, 11): (2101, 973),
                                  (11, 11): (3134, 1274),
                                  (11, 4): (2896, 2326)}
    A = np.zeros((2 * len(corners_cm_to_pixels_dict), 6), dtype=float)
    b = np.zeros((2 * len(corners_cm_to_pixels_dict), 1), dtype=float)
    index = 0
    for XY, xy in corners_cm_to_pixels_dict.items():
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
    #logging.debug("main(): x = {}".format(x))
    #logging.debug("main(): x.shape = {}".format(x.shape))
    pixels_to_cm_transformation_mtx = np.array([[x[0, 0], x[1, 0], x[2, 0]], [x[3, 0], x[4, 0], x[5, 0]], [0, 0, 1]])
    logging.debug("main(): pixels_to_cm_transformation_mtx = \n{}".format(pixels_to_cm_transformation_mtx))

    test_xy = (2888, 1650, 1)
    test_XY = pixels_to_cm_transformation_mtx @ test_xy
    logging.info("main(): test_xy = {}; test_XY = {}".format(test_xy, test_XY))

    cm_to_pixels_transformation_mtx = np.linalg.inv(pixels_to_cm_transformation_mtx)
    XY_5_9 = (5, 9, 1)
    xy_5_9 = cm_to_pixels_transformation_mtx @ XY_5_9
    image = cv2.imread(image_filepath)
    annotated_img = copy.deepcopy(image)
    draw_crosshair(annotated_img, xy_5_9, 20, (0, 255, 0))
    

def draw_crosshair(image, center, width, color):
    cv2.line(image, (center[0] - width//2, center[1]), (center[0] + width//2, center[1]), color)
    cv2.line(image, (center[0], center[1] - width//2), (center[0], center[1] + width//2), color)

if __name__ == '__main__':
    main()