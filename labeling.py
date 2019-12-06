import cv2
import time
import os
import numpy as np

draw = True
threshold = .99
strike_pattern_frame_label = 144
ball_pattern_frame_label = 724
box_pattern_frame_label = 144
r2_pattern_frame_label = 9

raw_frames_path = 'F:/hill/frames'
output_path = 'F:/hill/labeled'

_, _, filenames = next(os.walk(raw_frames_path))

box_pattern_frame_path = os.path.join(raw_frames_path, filenames[box_pattern_frame_label])
box_pattern_frame = cv2.imread(box_pattern_frame_path)
box_pattern = box_pattern_frame[332:338, 770:1145]

strike_pattern_frame_path = os.path.join(raw_frames_path, filenames[strike_pattern_frame_label])
strike_pattern_frame = cv2.imread(strike_pattern_frame_path)
strike_pattern = strike_pattern_frame[553:592, 1030:1073]

ball_pattern_frame_path = os.path.join(raw_frames_path, filenames[ball_pattern_frame_label])
ball_pattern_frame = cv2.imread(ball_pattern_frame_path)
ball_pattern = ball_pattern_frame[450:495, 1220:1262]

r2_pattern_frane_path = os.path.join(raw_frames_path, filenames[r2_pattern_frame_label])
r2_pattern_frame = cv2.imread(r2_pattern_frane_path)
r2_pattern = r2_pattern_frame[610:630, 1220:1245]

# viewing the patterns
cv2.imshow('box,', box_pattern)
cv2.imshow('strike', strike_pattern)
cv2.imshow('ball', ball_pattern)
cv2.imshow('r2', r2_pattern)
cv2.waitKey()

for i, file in enumerate(filenames):

    frame_path = os.path.join(raw_frames_path, file)

    print(frame_path)
    img = cv2.imread(frame_path)

    w, h = img.shape[0:2]

    cropx = 500
    cropy = 100
    img = img[50+cropy:w-50, 0+cropx:h-cropx]

    box_res = cv2.matchTemplate(img, box_pattern, cv2.TM_CCOEFF_NORMED)
    box_loc = np.where(box_res >= threshold)

    if box_loc:
        box_h, box_w = box_pattern.shape[0:2]

        if draw:
            for pt in zip(*box_loc[::-1]):
                cv2.rectangle(img, pt, (pt[0] +box_w, pt[1] + box_h), (0, 0, 255), 2)

        r2_res = cv2.matchTemplate(img, r2_pattern, cv2.TM_CCOEFF_NORMED)
        r2_loc = np.where(r2_res >= threshold)

        if not r2_loc:

            strike_res = cv2.matchTemplate(img, strike_pattern, cv2.TM_CCOEFF_NORMED)
            ball_res = cv2.matchTemplate(img, ball_pattern, cv2.TM_CCOEFF_NORMED)

            w, h = strike_pattern.shape[0:2]

            strike_loc = np.where(strike_res >= threshold)
            ball_loc = np.where(ball_res >= threshold)

            if strike_loc:
                name = "strike"
            elif ball_loc:
                name = "ball"

            if draw:
                for pt in zip(*strike_loc[::-1]):
                    print("strike")
                    cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

                for pt in zip(*ball_loc[::-1]):
                    print("ball")
                    cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

        else:
            if draw:
                for pt in zip(*box_loc[::-1]):
                    print("r2")
                    w, h = r2_pattern.shape[0:2]
                    cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

        cv2.imshow('image', img)

        cv2.waitKey()

