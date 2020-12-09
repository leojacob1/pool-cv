import numpy as np
import cv2
from basic import getBounds

cv2.namedWindow('img', cv2.WINDOW_NORMAL)

cap = cv2.VideoCapture('pool.mov')

fgbg = cv2.createBackgroundSubtractorMOG2()

while(1):
    ret, frame = cap.read()
    if ret == True:
        boundedFrame = getBounds(frame)
        fgmask = fgbg.apply(boundedFrame)
        cv2.imshow('frame',fgmask)
        cv2.waitKey(0)
        break
        # if cv2.waitKey(25) & 0xFF == ord('q'):
        #     break
    else:
        print('bad frame')
        break

print('out of loop')
cap.release()
cv2.destroyAllWindows()
