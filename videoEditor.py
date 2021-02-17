"""Skeleton code for python script to process a video using OpenCV package
:copyright: (c) 2021, Joeri Nicolaes
:license: BSD license
"""
import argparse
import cv2
import sys


# helper function to change what you do based on video seconds
def between(cap, lower: int, upper: int) -> bool:
    frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    framespersecond= int(cap.get(cv2.CAP_PROP_FPS))
    return lower <= int(2*frame/framespersecond) < upper


def main():
    input_video_file = '/home/karel/Downloads/video-1613578983.mp4'
    output_video_file = '/home/karel/Downloads/video-1613578983_bewerkt.MOV'
    # OpenCV video objects to work with
    cap = cv2.VideoCapture(input_video_file)
    fps = int(round(cap.get(5)))
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')        # saving output video as .mp4
    out = cv2.VideoWriter(output_video_file, fourcc, fps, (frame_width, frame_height))

    font = cv2.FONT_HERSHEY_SIMPLEX 

    # while loop where the real work happens
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if cv2.waitKey(28) & 0xFF == ord('q'):
                break

            if between(cap,0,10):
                # Grab your object in RGB and HSV color space.  
                # Show binary frames with the fore-ground object in white and background in black.  
                # If you carefully choose your object (i.e.   with  a  distinct  color  compared  to  the  rest  of  the  scene), 
                # this  can  be  a  simple thresholding operation.  
                # Choose a color space and try to improve your grabbing (e.g.fill  holes,  undetected  edges)  
                # by  using  binary  morphological  operations.   
                # Put  the  improvements in a different color
                pass

            #if between(cap,0,2) or between(cap,4,5):
                #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
            #elif between(cap,5,7):
                #frame = cv2.GaussianBlur(frame,(5,5),cv2.BORDER_DEFAULT)
                #cv2.putText(frame, 'GaussianBlur with kernel (5,5)', (50, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4) 
            #elif between(cap,8,10):
                #frame = cv2.GaussianBlur(frame,(11,11),cv2.BORDER_DEFAULT)
                #cv2.putText(frame, 'GaussianBlur with kernel (11,11)', (50, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4) 
            #elif between(cap,11,13):
                #frame = cv2.bilateralFilter(frame,15,80,80)
                #cv2.putText(frame, 'bi-lateral', (50, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4) 

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