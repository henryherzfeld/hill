import cv2
import time
import os
import numpy as np

testing_limit = 200
threshold = .99


raw_frames_path = 'F:/hill/frames'
output_path = 'F:/hill/labeled'

_, _, filenames = next(os.walk(raw_frames_path))

strike_pattern_frame_path = os.path.join(raw_frames_path, filenames[144])
strike_pattern_frame = cv2.imread(strike_pattern_frame_path)
strike_pattern = strike_pattern_frame[545:603, 1020:1080]
cv2.imshow('image', strike_pattern)
cv2.waitKey()

for i, file in enumerate(filenames):
    if 100 < i < testing_limit:
        frame_path = os.path.join(raw_frames_path, file)

        print(frame_path)
        img = cv2.imread(frame_path)

        res = cv2.matchTemplate(img, strike_pattern, cv2.TM_CCOEFF)

        w, h = strike_pattern.shape[0:2]

        loc = np.where(res >= threshold)
        for pt in zip(*loc[::-1]):
            cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

        cv2.imshow('image', img)

        cv2.waitKey()

