import cv2
import time
import os
import numpy as np
import shutil

raw_frames_path = 'F:/hill/frames'
output_folder = 'labeled'

_, _, filenames = next(os.walk(raw_frames_path))

n_pitch_frames = 15
post_save_buffer = 100
save_buffer_count = post_save_buffer
saved = False
label = True
draw = False
view = False

if label:
    output_path = os.path.join(os.getcwd(), output_folder)

    if not os.path.isdir(output_path):
        os.mkdir(output_folder)


def save_pitch_frames(label, seq_end_idx):

    frames = filenames[seq_end_idx - n_pitch_frames:seq_end_idx]

    save_path = os.path.join(output_path, label)

    os.makedirs(save_path)

    for frame in frames:
        source_path = os.path.join(raw_frames_path, frame)
        destination_path = os.path.join(save_path, frame)

        dest = shutil.copy2(source_path, save_path)

        print("saved to ", dest)

    return True


uncertain_frames = 0
uncertain_frame_threshold = 25

threshold = .90
strike_pattern_frame_label = 144
ball_pattern_frame_label = 724
box_pattern_frame_label = 144
r2_pattern_frame_label = 9
cropx = 500
cropy = 100

ball_count = 1
strike_count = 1

# reading in and cropping templates from frames
box_pattern_frame = cv2.imread(os.path.join(raw_frames_path, filenames[box_pattern_frame_label]))
box_pattern = box_pattern_frame[332:338, 770:1145]
box_h, box_w = box_pattern.shape[0:2]

strike_pattern_frame = cv2.imread(os.path.join(raw_frames_path, filenames[strike_pattern_frame_label]))
strike_pattern = strike_pattern_frame[553:565, 1030:1073]
strike_h, strike_w = strike_pattern.shape[0:2]

ball_pattern_frame = cv2.imread(os.path.join(raw_frames_path, filenames[ball_pattern_frame_label]))
ball_pattern = ball_pattern_frame[450:465, 1222:1257]
ball_h, ball_w = ball_pattern.shape[0:2]

r2_pattern_frame = cv2.imread(os.path.join(raw_frames_path, filenames[r2_pattern_frame_label]))
r2_pattern = r2_pattern_frame[610:630, 1280:1340]
r2_h, r2_w = r2_pattern.shape[0:2]

# xx = 900
# 
# filenames = filenames[xx:]

# viewing the templates
if view:
    cv2.imshow('box,', box_pattern)
    cv2.imshow('strike', strike_pattern)
    cv2.imshow('ball', ball_pattern)
    cv2.imshow('r2', r2_pattern)
    cv2.waitKey()

for i, file in enumerate(filenames):

    if not save_buffer_count:
        save_buffer_count = post_save_buffer
        saved = False

    if saved:
        save_buffer_count -= 1

    else:
        frame_path = os.path.join(raw_frames_path, file)
        img = cv2.imread(frame_path)

        # cropping image
        w, h = img.shape[0:2]
        img = img[50+cropy:w-50, 0+cropx:h-cropx]

        print(frame_path)

        # template matching for box pattern
        box_res = cv2.matchTemplate(img, box_pattern, cv2.TM_CCOEFF_NORMED)
        box_loc = np.where(box_res >= threshold)

        if np.any(box_loc):
            print("box")
            if draw:
                for pt in zip(*box_loc[::-1]):
                    cv2.rectangle(img, pt, (pt[0] + box_w, pt[1] + box_h), (0, 0, 255), 2)

            # template matching for r2 pattern
            r2_res = cv2.matchTemplate(img, r2_pattern, cv2.TM_CCOEFF_NORMED)
            r2_loc = np.where(r2_res >= threshold)

            if not np.any(r2_loc):
                # template matching for strike pattern
                strike_res = cv2.matchTemplate(img, strike_pattern, cv2.TM_CCOEFF_NORMED)
                strike_loc = np.where(strike_res >= threshold)

                if not np.any(strike_loc):
                    print("no strike pattern")
                    # template matching for ball pattern
                    ball_res = cv2.matchTemplate(img, ball_pattern, cv2.TM_CCOEFF_NORMED)
                    ball_loc = np.where(ball_res >= threshold)

                    if not np.any(ball_loc):
                        uncertain_frames += 1

                        if uncertain_frames >= uncertain_frame_threshold:
                            saved = save_pitch_frames("ball" + str(ball_count), i - uncertain_frames - 1)
                            ball_count += 1
                            uncertain_frames = 0

                    else:
                        saved = save_pitch_frames("ball" + str(ball_count), i-1)
                        ball_count += 1
                        uncertain_frames = 0

                else:
                    saved = save_pitch_frames("strike" + str(strike_count), i-1)
                    strike_count += 1
                    uncertain_frames = 0

                if draw:
                    for pt in zip(*strike_loc[::-1]):
                        print("strike")
                        cv2.rectangle(img, pt, (pt[0] + strike_w, pt[1] + strike_h), (0, 0, 255), 2)

                    for pt in zip(*ball_loc[::-1]):
                        print("ball")
                        cv2.rectangle(img, pt, (pt[0] + ball_w, pt[1] + ball_h), (0, 0, 255), 2)

            else:
                if draw:
                    for pt in zip(*r2_loc[::-1]):
                        print("r2")
                        cv2.rectangle(img, pt, (pt[0] + r2_w, pt[1] + r2_h), (0, 0, 255), 2)

            if view:
                cv2.imshow('image', img)
                cv2.waitKey()
