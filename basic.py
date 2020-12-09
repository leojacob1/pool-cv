import cv2
from matplotlib import pyplot as plt
import numpy as np

table_empty = cv2.imread("table6_empty.jpg", 1)
# table_empty = cv2.cvtColor(table_empty, cv2.COLOR_BGR2HSV)
table = cv2.imread("table6.jpg", 1)
# table = cv2.cvtColor(table, cv2.COLOR_BGR2HSV)


cv2.namedWindow('img', cv2.WINDOW_NORMAL)

def getSides(pts):
    c1 = { "x": pts[0][0], "y": pts[0][1] }
    c2 = { "x": pts[1][0], "y": pts[1][1] }
    c3 = { "x": pts[2][0], "y": pts[2][1] }
    c4 = { "x": pts[3][0], "y": pts[3][1] }

    d12 = np.sqrt((c1.get('x') - c2.get('x'))**2 + (c1.get('y') - c2.get('y'))**2)
    d13 = np.sqrt((c1.get('x') - c3.get('x'))**2 + (c1.get('y') - c3.get('y'))**2)
    d14 = np.sqrt((c1.get('x') - c4.get('x'))**2 + (c1.get('y') - c4.get('y'))**2)

    lengths = [d12, d13, d14]
    lengths.sort()

    height = lengths[0]
    width = lengths[1]

    return { "height": height, "width": width }

def getCenter(img, scale):
    h, w, c = img.shape
    croppedEdge = (1 - scale) / 2
    x1 = int(croppedEdge*w)
    x2 = int(x1 + scale*w)
    y1 = int(croppedEdge*h)
    y2 = int(y1 + scale*h)

    croppedImg = img[y1:y2, x1:x2]
    resizedImg = cv2.resize(croppedImg, None, fx=1/scale, fy=1/scale, interpolation=cv2.INTER_LINEAR)
    return resizedImg


def unique_count_app(a):
    colors, count = np.unique(a.reshape(-1,a.shape[-1]), axis=0, return_counts=True)
    return colors[count.argmax()]

def getWarpMatrix(image):
    tableColorBgr = unique_count_app(getCenter(image, 0.5))
    lower = []
    upper = []
    plusMinus = 40
    for color in tableColorBgr:
        if color >= plusMinus:
            lower.append(color - plusMinus)
        else:
            lower.append(0)
        if color <= 256 - plusMinus:
            upper.append(color + plusMinus)
        else:
            upper.append(256)

    lower = np.array(lower)
    upper = np.array(upper)

    mask = cv2.inRange(image, lower, upper)
    output = cv2.bitwise_and(image, image, mask=mask)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key = cv2.contourArea)
    rect = cv2.minAreaRect(c)
    box = np.int0(cv2.boxPoints(rect))
    boxWidth = int(getSides(box)['width'])
    boxHeight = int(getSides(box)['height'])

    # rows, cols = mask.shape
    pts1 = np.float32([box[0], box[1], box[2]])
    pts2 = np.float32([[0, 0], [0, boxHeight], [boxWidth, boxHeight]])

    M = cv2.getAffineTransform(pts1, pts2)
    return M, boxWidth, boxHeight

M, boxWidth, boxHeight = getWarpMatrix(table_empty)
background = cv2.warpAffine(table_empty, M, (boxWidth, boxHeight))#[0:int(boxHeight), 0:int(boxWidth)]
table = cv2.warpAffine(table, M, (boxWidth, boxHeight))#[0:int(boxHeight), 0:int(boxWidth)]

difference = cv2.absdiff(table, background)

lower = np.array([3, 3, 3])
upper = np.array([255, 255, 255])

mask = cv2.inRange(difference, lower, upper)
output = cv2.bitwise_and(table, table, mask=mask)

contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

balls = []
for con in contours:
    perimeter = cv2.arcLength(con, True)
    area = cv2.contourArea(con)
    if perimeter == 0:
        continue
    circularity = 4*np.pi*(area/(perimeter*perimeter))
    if area > 1000 and area < 11000 and circularity > 0.5 and circularity < 1.2:
        balls.append(con)

cv2.drawContours(output, balls, -1, (0, 255, 0), 6)
cv2.drawContours(mask, balls, -1, (0, 255, 0), 6)

cv2.imshow('img', mask)
cv2.waitKey(0)
cv2.imshow('img', output)
cv2.waitKey(0)

cv2.destroyAllWindows()
