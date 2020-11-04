"""
    date:       2020/10/13 8:13 下午
    written by: neonleexiang
"""
import cv2 as cv
import numpy as np
import os


def get_img_dir(file_dir, labels):
    path = []
    for c in os.listdir(file_dir+'/'+labels):
        if c != '.DS_Store':
            path.append([file_dir, labels, c])
    return path


def img_process(img_path):
    _, labels, name = tuple(img_path)
    # print('now we are preprocessing the -----------------'+name)
    img_bgr = cv.imread('/'.join(img_path))
    img_bgr = cv.copyMakeBorder(img_bgr, 10, 10, 10, 10,
                                cv.BORDER_CONSTANT,
                                value=[255, 255, 255])
    img_gray = cv.imread('/'.join(img_path), cv.IMREAD_GRAYSCALE)
    img_gray = cv.copyMakeBorder(img_gray, 10, 10, 10, 10,
                                 cv.BORDER_CONSTANT,
                                 value=[255, 255, 255])

    img_gray_copy = img_gray.copy()
    img_gray_copy = cv.GaussianBlur(img_gray_copy, (9, 9), 0)
    _, img_gray_copy = cv.threshold(img_gray_copy, 220, 255, cv.THRESH_TOZERO_INV)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (10, 10))
    # kernel = np.ones((30, 30), np.uint8)
    img_gray_copy = cv.erode(img_gray_copy, kernel, iterations=10)
    img_gray_copy = cv.dilate(img_gray_copy, kernel, iterations=10)

    _, img_gray_copy = cv.threshold(img_gray_copy, 15, 255, cv.THRESH_BINARY)
    # img_gray_copy = cv.Canny(img_gray_copy, 0, 50, apertureSize=3)
    # img_gray_copy = cv.dilate(img_gray_copy, kernel, iterations=5)

    (contours, _) = cv.findContours(img_gray_copy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    area = []
    for i in range(len(contours)):
        area.append(cv.contourArea(contours[i]))
    c = np.argmax(area)
    rect = cv.minAreaRect(contours[c])

    # c = sorted(contours, key=cv.contourArea, reverse=True)[0]
    # rect = cv.minAreaRect(c)

    box = np.int0(cv.boxPoints(rect))

    src_h, src_w = img_bgr.shape[:2]
    mask = np.full((src_h, src_w), 255, dtype=np.uint8)

    mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    cv.fillConvexPoly(mask, box, (0, 0, 0))

    mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)

    result_img = cv.add(img_gray, mask)
    cv.imwrite('new_datasets/' + labels + '/' + name, result_img)


def preprocess(img_dir_lst):
    for path in img_dir_lst:
        print('we are preprocessing ---------', path)
        img_process(path)

    print('finished')

# FILE_DIR ='datasets/normal/'
#
# print(get_img_dir(FILE_DIR))


if __name__ == '__main__':
    datasets_dir = 'datasets'
    normal_img_path = get_img_dir(datasets_dir, 'normal')
    # print('/'.join(normal_img_path[0]))
    preprocess(normal_img_path)

    unnormal_img_path = get_img_dir(datasets_dir, 'unnormal')
    # print(unnormal_img_path)
    preprocess(unnormal_img_path)
