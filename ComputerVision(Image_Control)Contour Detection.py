import cv2
import numpy as np
import sys
import Dir
import pandas as pd
import matplotlib.pyplot as plt

# 2.4 윤곽 감지(Contour Detection)
img = cv2.imread(Dir.dir+'card.png')
target_img = img.copy()  # 사본이미지
# 이진 이미지로 변경
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 최적의 임계치를 찾아 이진화
ret, otsu = cv2.threshold(grey, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# THRESH_OTSU 원래는 기준 값을 적음 하지만 THRESH_OTSU을 사용하면 최적의 기준값 찾아줌

# 윤곽선 검출 -구조. 계층구조
contours , hierarchy = cv2.findContours(otsu, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
# 윤곽선 정보, 구조 = 대상이미지, 윤곽선 찾는 모드(mode), 윤곽선 찾을때 사용하는 근사치(method)
# 윤곽선 그리기
# cv2.RETR_LIST - 있는 외곽선을 다 찾아라
# cv2.RETR_EXTERNAL - 외곽에 았는 외곽선만 찾아라
# cv2.

COLOR = (0, 200, 2) # 녹색
cv2.drawContours(target_img, contours, -1, COLOR, 2)
                # 대상이미지, 윤곽선 정보, 인덱스(-1 이면 전체), 색깔, 두께
cv2.imshow('img', img)
cv2.imshow('gray', grey)
cv2.imshow('otsu', otsu)
cv2.imshow('contour', target_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


greytest = cv2.imread(Dir.dir+"[Dataset] Module 20 images/image001_3contours.png",0)    # 해당 이미지 읽기
contouroutlines = np.zeros(greytest.shape,dtype="uint8")    # 감지된 윤곽을 그리기 위한 빈 캔버스 만들기

# 윤곽선을 찾아봅시다!
# https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga17ed9f5d79ae97bd4c7cf18403e1689a
(cnts,_) = cv2.findContours(greytest, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(cnts)
for (i, c) in enumerate(cnts):
    cv2.drawContours(contouroutlines, [c], -1, 255, 1)  # 각 윤곽선에 대해 윤곽 영역의 바깥선만 그립니다.
                                                        # https://docs.opencv.org/3.4/d6/d6e/group__imgproc__draw.html#ga746c0625f1781f1ffc9056259103edbc
cv2.imshow("Contour Outlines",contouroutlines)          # 결과 표시
cv2.waitKey(0)                                          # 아무 키나 누른 후 창 종료
cv2.destroyAllWindows()

print("There are "+str(len(cnts))+" contours!")

img = cv2.imread(Dir.dir+"[Dataset] Module 20 images/image001.png")
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 임계값을 적용합니다
(T, thresholded) = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)# 0이 들어가냐 -1이 들어가냐 OTSU를 사용하기 때문에 상관 X
#파이썬에서는 -1을 사용하면 기준값이 뭔지 몰라도 알아서 전체로 해서 찾아줌

cv2.imshow("Thresholded",thresholded)

cv2.waitKey(0)                           # 아무 키나 누른 후 창 종료
cv2.destroyAllWindows()

# 윤곽선을 찾아보자!
(cnts, _) = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

mask = np.zeros(img.shape, dtype="uint8")  # 감지된 윤곽을 그리기 위한 빈 캔버스 만들기
for (i, c) in enumerate(cnts):
    cv2.drawContours(mask, [c], -1, (0, 255, 0), 1)

cv2.imshow("Mask", mask)
cv2.waitKey(0)  # 아무 키나 누른 후 창 종료
cv2.destroyAllWindows()

print("There are " + str(len(cnts)) + " contours!")

grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 임계값을 적용합니다
(T, thresholded) = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
cv2.imshow("Thresholded", thresholded)

# 윤곽선을 찾아보자!
(cnts, _) = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=lambda cnts: cv2.boundingRect(cnts)[1])  # 윤곽선을 위에서 아래로 정렬합니다.

mask = cv2.merge([thresholded, thresholded, thresholded])  # 감지된 윤곽을 그리기 위한 캔버스 만들기
for (i, c) in enumerate(cnts):  # https://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html
    # cv2.drawContours(mask, [c], -1, (255,255,255), -1)
    (x, y, w, h) = cv2.boundingRect(c)  # 윤곽 경계 상자의 x,y 좌표를 가져옵니다.
    cv2.rectangle(mask, (x, y), (x + w, y + h), (0, 0, 255))  # 경계 상자를 빨간색으로 그립니다.

    cv2.putText(mask, "" + str(i + 1), (x, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 0), 1)

cv2.imshow("Mask", mask)
cv2.waitKey(0)  # 아무 키나 누른 후 창 종료
cv2.destroyAllWindows()

print("There are " + str(len(cnts)) + " contours!")

(T, thresholded) = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
thresholded[410:, :] = 0  # 이미지 하단에 있는 텍스트를 제거합니다.
# cv2.imshow("Thresholded",thresholded)

# 윤곽선이 몇개인지 확인하기.
(cnts, _) = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

mask = np.zeros(thresholded.shape, dtype="uint8")
for (i, c) in enumerate(cnts):
    cv2.drawContours(mask, [c], -1, 255, -1)  # 마지막 매개변수는 윤곽선 두께를 정의합니다. -1은 윤곽선 내부를 채웁니다.

cv2.imshow("Mask", mask)
cv2.imshow("Masked Image", cv2.bitwise_and(img, img, mask=mask))
cv2.waitKey(0)  # 아무 키나 누른 후 창 종료
cv2.destroyAllWindows()

print("There are " + str(len(cnts)) + " contours!")

# 2.5 선 그리기 및 텍스트 쓰기