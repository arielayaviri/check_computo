# -*- coding: UTF-8 -*-
import numpy as np
import os, time
import json
import requests
import cv2
import pandas as pd
from sklearn.externals import joblib
from skimage.feature import hog

# METHOD #1: OpenCV, NumPy, and urllib
def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = requests.get(url)

    image = np.asarray(bytearray(resp.content), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # return the image
    return image

def get_test_image():
    return cv2.imread('./test_image.jpg', cv2.IMREAD_COLOR)

def get_mesas_list(path):
    xls = pd.ExcelFile(path)
    sheet1 = xls.parse(0)
    return sheet1[u'Número Mesa']

def url_to_json(url):
    response = requests.get(url)
    print(response.content)
    data = json.loads(response.content)
    return data

def evaluate_mesa(mesa_num):
    presidentResultsUrl = 'https://trep.oep.org.bo/resul/resulActa/' + str(mesa_num) + '/1'
    diputadoResultsUrl = 'https://trep.oep.org.bo/resul/resulActa/' + str(mesa_num) + '/2'
    actaPhotoUrl = 'https://trep.oep.org.bo/resul/imgActa/' + str(mesa_num) + '1.jpg'

    debugFolderPath = './output'
    # print url_to_json(presidentResultsUrl)
    if not os.path.exists(debugFolderPath):
        os.makedirs(debugFolderPath)

    # actaImg = url_to_image(actaPhotoUrl) # to get image from url
    actaImg = get_test_image()
    roi = [(700, 750), (2500, 2100)]
    cv2.rectangle(actaImg, roi[0], roi[1], [255,0,0], 20)
    debugFileName = str(int(time.time()))+'.jpg'
    cv2.imwrite(debugFolderPath+'/'+debugFileName, actaImg, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    cv2.imwrite(debugFolderPath+'/'+'roi_'+debugFileName, actaImg[roi[0][1]:roi[1][1], roi[0][0]:roi[1][0]], [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    # TODO: Implement digits recognition to detect the data from the acta photo
    # recognize_digits(actaImg[roi[0][1]:roi[1][1], roi[0][0]:roi[1][0]])

def recognize_digits(im):
    clf = joblib.load("digits_cls.pkl")
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

    # Threshold the image
    ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the image
    ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get rectangles contains each contour
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]

    # For each rectangular region, calculate HOG features and predict
    # the digit using Linear SVM.
    for rect in rects:
        # Draw the rectangles
        cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
        # Make the rectangular region around the digit
        leng = int(rect[3] * 1.6)
        pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
        pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
        roi = im_th[pt1:pt1 + leng, pt2:pt2 + leng]
        # Resize the image
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        roi = cv2.dilate(roi, (3, 3))
        # Calculate the HOG features
        roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
        nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
        cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)

    debugFolderPath = './output'
    debugFileName = str(int(time.time())) + '.jpg'
    cv2.imwrite(debugFolderPath + '/' + 'digits_'+debugFileName, im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    return rects

def main():
    mesa_list = get_mesas_list('./testActaList.xlsx')
    # print mesa_list[0]
    evaluate_mesa(mesa_list[0])
    # iterate over all mesas
    # for mesa_num in mesa_list:
    #     print 'Checking mesa: ' + mesa_num
    #     evaluate_mesa(mesa_num)

main()
