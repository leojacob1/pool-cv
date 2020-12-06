import cv2
from matplotlib import pyplot as plt
import numpy as np

image = cv2.imread("table5.jpg", 1)
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

tableColorBgr = unique_count_app(getCenter(image, 0.25))
lower = []
upper = []
plusMinus = 80
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

cv2.imshow('img', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
c = max(contours, key = cv2.contourArea)
rect = cv2.minAreaRect(c)
box = np.int0(cv2.boxPoints(rect))
boxWidth = getSides(box)['width']
boxHeight = getSides(box)['height']

# cv2.rectangle(output,(x,y),(x+w,y+h),(0,255,0),2)
# cv2.drawContours(output, [c], 0, (0, 255, 0), 3)
rows, cols = mask.shape
pts1 = np.float32([box[0], box[1], box[2]])
pts2 = np.float32([[0, 0], [0, boxHeight], [boxWidth, boxHeight]])

M = cv2.getAffineTransform(pts1, pts2)
output = cv2.warpAffine(image, M, (cols, rows))[0:int(boxHeight), 0:int(boxWidth)]


cv2.imshow('img', output)
cv2.waitKey(0)
cv2.destroyAllWindows()


# plt.subplot(121),plt.imshow(mask),plt.title('Input')
# plt.subplot(122),plt.imshow(dst),plt.title('Output')
# plt.show()


# rect = cv2.minAreaRect(c)
# box = np.int0(cv2.boxPoints(rect))
# cv2.drawContours(output, [box], 0, (0, 255, 0), 2)

# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow("img", gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# grayHist = cv2.calcHist([gray], [0], None, [256], [0, 256])

# chans = cv2.split(image)
# colors = ('b', 'g', 'r')
# plt.figure()
# plt.title("'Flattened' Color Histogram")
# plt.xlabel("Bins")
# plt.ylabel("# of Pixels")
# features = []
#
# for (chan, color) in zip(chans, colors):
#     hist = cv2.calcHist([chan], [0], None, [16], [0, 256])
#     features.extend(hist)
#
#     plt.plot(hist, color = color)
#     plt.xlim([0, 16])

# print(f"flattened feature vector size: {np.array(features).flatten().shape}")
# plt.show()
