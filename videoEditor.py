"""Skeleton code for python script to process a video using OpenCV package
:copyright: (c) 2021, Joeri Nicolaes
:license: BSD license
"""
import argparse
import cv2
import sys
import numpy as np


# helper function to change what you do based on video seconds
def between(cap, lower: int, upper: int) -> bool:
    frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    framespersecond = int(cap.get(cv2.CAP_PROP_FPS))
    return lower <= int(2*frame/framespersecond) < upper


def main():
    input_video_file = '/home/karel/Documents/computervision/video_cv_small.mp4'
    output_video_file = '/home/karel/Documents/computervision/AwesomeVideo.mp4'
    # OpenCV video objects to work with
    cap = cv2.VideoCapture(input_video_file)
    fps = int(round(cap.get(5)))
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    # saving output video as .mp4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_file, fourcc, fps,
                          (frame_width, frame_height))

    font = cv2.FONT_HERSHEY_SIMPLEX

    # while loop where the real work happens
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if cv2.waitKey(28) & 0xFF == ord('q'):
                break
            """
            elif between(cap,0,2) or between(cap,4,5):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cv2.putText(frame, 'gray', (50, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)
            elif between(cap, 2, 4):
                cv2.putText(frame, 'color', (50, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)

            elif between(cap,5,7):
                frame = cv2.GaussianBlur(frame,(5,5),cv2.BORDER_DEFAULT)
                cv2.putText(frame, 'GaussianBlur with kernel (5,5)', (50, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)
            elif between(cap,7,12):
                frame = cv2.GaussianBlur(frame,(11,11),cv2.BORDER_DEFAULT)
                cv2.putText(frame, 'GaussianBlur with kernel (11,11)', (50, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)
                cv2.putText(frame, 'Bigger kernel = more smoothing', (50, 500), font, 1, (0, 255, 255), 2, cv2.LINE_4)
            elif between(cap,10,13):
                frame = cv2.bilateralFilter(frame,15,80,80)
                cv2.putText(frame, 'bi-lateral', (20, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)
                cv2.putText(frame, 'No averaging across edges', (50, 300), font, 1, (0, 255, 255), 2, cv2.LINE_4)
                cv2.putText(frame, 'd = 15', (50, 500), font, 1, (0, 255, 255), 2, cv2.LINE_4)
            elif between(cap,22,25):
                frame = cv2.bilateralFilter(frame,20,80,80)
                cv2.putText(frame, 'bi-lateral', (20, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)
                cv2.putText(frame, ' d = 20 => more smoothing', (20, 500), font, 1, (0, 255, 255), 2, cv2.LINE_4)
            if between(cap,31,40):
                # grabbing the object in HSV space
                GREEN_MIN = np.array([50, 100, 100], np.uint8)
                GREEN_MAX = np.array([70, 255, 255], np.uint8)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                frame = cv2.inRange(frame, GREEN_MIN, GREEN_MAX)
            if between(cap,40,43):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Show binary frames with the foreground object in white and background in black.
                cv2.putText(frame, 'Method 2', (20, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)
                ret2,frame = cv2.threshold(frame, 100, 255, cv2.THRESH_BINARY_INV) # cv.THRESH_TRUNC can make stuff disapear
            """


            """
            if between(cap, 20, 40):
                # detect vertical edges
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
                sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
                frame = cv2.filter2D(frame, -1, sobel_x)
                frame = cv2.GaussianBlur(frame, (5, 5), cv2.BORDER_DEFAULT)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                # frame[np.where((frame==[0,0,0]).all(axis=2))] = [255,165,0]
                frame[np.where(frame[:, :, 0] > 10)] = [0, 255, 0]
                frame = cv2.GaussianBlur(frame, (5, 5), cv2.BORDER_DEFAULT)

            if between(cap, 40, 60):
                # detects cicles
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                circles = cv2.HoughCircles(
                    frame, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    # draw the outer circle
                    cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)

            if between(cap, 20, 40):
                # box around object
                colorLow = np.array([20, 20, 20])
                colorHigh = np.array([120, 120, 120])
                mask = cv2.inRange(frame, colorLow, colorHigh)
                contours, hierarchy = cv2.findContours(
                    mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                contour_sizes = [(cv2.contourArea(contour), contour)
                                 for contour in contours]
                biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
                x, y, w, h = cv2.boundingRect(biggest_contour)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            if between(cap, 0, 20):
                # use feature extraction to find object in video
                feature_img = cv2.imread('/home/karel/Downloads/green.jpg') 
                feature_img_bw = cv2.cvtColor(feature_img,cv2.COLOR_BGR2GRAY) 

                frame_img_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

                orb = cv2.ORB_create() 

                featureKeypoints, featureDescriptors = orb.detectAndCompute(feature_img_bw,None) 
                frameKeypoints, frameDescriptors = orb.detectAndCompute(frame_img_bw,None) 

                matcher = cv2.BFMatcher() 
                matches = matcher.match(featureDescriptors,frameDescriptors) 

                # TO DO: Find a way to use the feature matches to find the object and draw a box around it

                # frame = cv2.drawMatches(feature_img, featureKeypoints, train_img, trainKeypoints, matches[:20],None)
                frame = cv2.drawKeypoints(feature_img, featureKeypoints, outImage = None, color=(0, 255, 0), flags=0)  
            """

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
