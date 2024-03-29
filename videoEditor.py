"""Skeleton code for python script to process a video using OpenCV package
:copyright: (c) 2021, Joeri Nicolaes
:license: BSD license
"""

"""
Script written by Karel Everaert
Shows basic video processing techniques
"""
import cv2
import numpy as np
from random import randint

# helper function to change what you do based on video seconds
def between(cap, lower: int, upper: int) -> bool:
    frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    framespersecond = int(cap.get(cv2.CAP_PROP_FPS))
    return lower <= int(2*frame/framespersecond) < upper

def main():
    input_video_file = '/home/karel/Documents/computervision/video_cv_small.mp4'
    cap = cv2.VideoCapture(input_video_file)

    # Find width, height and Frames-per-second of the video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = int(cap.get(cv2.CAP_PROP_FPS))

    # Create a Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('check.mp4', fourcc, FPS, (width, height))


    font = cv2.FONT_HERSHEY_SIMPLEX

    # while loop where the real work happens
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if cv2.waitKey(28) & 0xFF == ord('q'):
                break
            elif between(cap,0,3) or between(cap,6,9):
                # changing between grayscale and colors
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                cv2.putText(frame, 'gray', (50, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)
            elif between(cap, 3, 6):
                cv2.putText(frame, 'color', (50, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)
            elif between(cap,9,11):
                # performing a gaussian blur with different sizes for the filter
                frame = cv2.GaussianBlur(frame,(5,5),cv2.BORDER_DEFAULT)
                cv2.putText(frame, 'GaussianBlur with kernel (5,5)', (50, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)
            elif between(cap,11,15):
                frame = cv2.GaussianBlur(frame,(11,11),cv2.BORDER_DEFAULT)
                cv2.putText(frame, 'GaussianBlur with kernel (11,11)', (50, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)
                cv2.putText(frame, 'Bigger kernel = more smoothing', (50, 500), font, 1, (0, 255, 255), 2, cv2.LINE_4)
            elif between(cap,15,19):
                # using a bilateral filter
                frame = cv2.bilateralFilter(frame,15,80,80)
                cv2.putText(frame, 'bi-lateral', (20, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)
                cv2.putText(frame, 'No averaging across edges', (50, 300), font, 1, (0, 255, 255), 2, cv2.LINE_4)
                cv2.putText(frame, 'd = 15', (50, 500), font, 1, (0, 255, 255), 2, cv2.LINE_4)
            elif between(cap,19,23):
                frame = cv2.bilateralFilter(frame,20,80,80)
                cv2.putText(frame, 'bi-lateral', (20, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)
                cv2.putText(frame, ' d = 20 => more smoothing', (20, 500), font, 1, (0, 255, 255), 2, cv2.LINE_4)
            # can't start sooner
            elif between(cap,33,35): # 40
                # grabbing the object in HSV space
                GREEN_MIN = np.array([50, 100, 100], np.uint8)
                GREEN_MAX = np.array([70, 255, 255], np.uint8)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                frame = cv2.inRange(frame, GREEN_MIN, GREEN_MAX)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                cv2.putText(frame, 'Capturing the object in HSV space', (50, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)
            elif between(cap,35,40): # start 35
                # improving the grabbing
                # dilation
                # thresholding
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                GREEN_MIN = np.array([50, 100, 100], np.uint8)
                GREEN_MAX = np.array([70, 255, 255], np.uint8)
                mask = cv2.inRange(hsv, GREEN_MIN, GREEN_MAX)
                frame = cv2.bitwise_and(frame, frame, mask=mask)

                kernel = np.ones((5, 5), np.uint8)
                frame = cv2.dilate(frame, kernel, iterations=1)
                # morphological operations (Opening)
                kernel = np.ones((5, 5), np.uint8)
                frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
                GREEN_MIN = np.array([50, 100, 100], np.uint8)
                GREEN_MAX = np.array([70, 255, 255], np.uint8)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                frame = cv2.inRange(frame, GREEN_MIN, GREEN_MAX)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                cv2.putText(frame, 'Morphological operations', (50, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)
            elif between(cap,40,46):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Show binary frames with the foreground object in white and background in black.
                cv2.putText(frame, 'Method 2', (20, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)
                ret2,frame = cv2.threshold(frame, 100, 255, cv2.THRESH_BINARY_INV) # cv.THRESH_TRUNC can make stuff disapear
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB) # might have to be removed
            elif between(cap, 46, 50):
                # detect vertical edges
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
                frame = cv2.filter2D(frame, -1, sobel_x)
                frame = cv2.GaussianBlur(frame, (5, 5), cv2.BORDER_DEFAULT)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                # frame[np.where((frame==[0,0,0]).all(axis=2))] = [255,165,0]
                frame[np.where(frame[:, :, 0] > 10)] = [0, 255, 0]
                frame = cv2.GaussianBlur(frame, (5, 5), cv2.BORDER_DEFAULT)
                cv2.putText(frame, 'Detect Vertical edges', (20, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)
            elif between(cap, 50, 53):
                # detect horizontal edges
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
                frame = cv2.filter2D(frame, -1, sobel_y)
                frame = cv2.GaussianBlur(frame, (5, 5), cv2.BORDER_DEFAULT)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                # frame[np.where((frame==[0,0,0]).all(axis=2))] = [255,165,0]
                frame[np.where(frame[:, :, 0] > 10)] = [0, 255, 0]
                frame = cv2.GaussianBlur(frame, (5, 5), cv2.BORDER_DEFAULT)
                cv2.putText(frame, 'Detect Horizontal edges', (20, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)
            elif between(cap, 53 , 58):
                # box around object
                colorLow = np.array([20, 20, 20])
                colorHigh = np.array([120, 120, 120])
                mask = cv2.inRange(frame, colorLow, colorHigh)
                contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                contour_sizes = [(cv2.contourArea(contour), contour)for contour in contours]
                biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
                x, y, w, h = cv2.boundingRect(biggest_contour)
                color = (randint(100, 255), randint(100, 255), randint(100, 255))
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, 'Draw a box around the target', (50, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)
            elif between(cap,58,70):
                # showing the likelihood of where the object is in the picture
                frame_w = width
                frame_h = height
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                feature_img = cv2.imread('/home/karel/Documents/computervision/template.png')
                feature_img = cv2.cvtColor(feature_img, cv2.COLOR_BGR2GRAY)
                feature_img = cv2.resize(feature_img, (1536, 2048))
                likelihood_map = cv2.matchTemplate(frame_gray, feature_img,cv2.TM_SQDIFF_NORMED)
                likelihood_map = cv2.cvtColor(
                    cv2.normalize(likelihood_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U),
                    cv2.COLOR_GRAY2BGR)
                likelihood_map = cv2.resize(likelihood_map, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)
                frame = likelihood_map
            elif between(cap, 76, 86):
                # Drawing circles
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_gray = cv2.blur(frame_gray, (3, 3))
                detected_circles = cv2.HoughCircles(frame_gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=1,maxRadius=40)
                if detected_circles is not None:
                    detected_circles = np.uint16(np.around(detected_circles))
                    for pt in detected_circles[0, :]:
                        a, b, r = pt[0], pt[1], pt[2]
                        color = (randint(100, 255), randint(100, 255), randint(100, 255))
                        cv2.circle(frame, (a, b), r, color, 2)
                cv2.putText(frame, 'Draw flashy circles', (50, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)
            elif between(cap,97,100):
                # changing the colors of the white ball
                cv2.putText(frame, 'Ball', (20, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ret, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
                frame[thresh == 255] = 0
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                erosion = cv2.erode(frame, kernel, iterations=1)
                cv2.putText(frame, 'Changing white pixels to black', (50, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)
            elif between(cap,101,1025): # 97
                cv2.putText(frame, 'Ball', (20, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ret, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
                frame[thresh == 255] = 100
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                erosion = cv2.erode(frame, kernel, iterations=1)

            # write frame that you processed to output
            out.write(frame)

            # (optional) display the resulting frame
            cv2.imshow('Frame', frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release the video capture and writing object
    cap.release()
    out.release()
    # Closes all the frames
    cv2.destroyAllWindows()


main()
