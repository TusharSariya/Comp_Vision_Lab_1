# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 01:32:57 2020

@author: ebrahim

Modified by Tushar Sariya and Richard Walmsley
"""

import cv2
import numpy as np
import argparse

#class for a line with end points defined
class line:
    def __init__(self, x0, y0, x1, y1):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1

#calculates intersetion using the begaining and end coordinates of each line
#parameters: two lines
#return: point of intersection
def line_intersection(line1, line2):
    def det(a, b, c, d):
        return a*d - b*c

    bottom = (line1.x0-line1.x1)*(line2.y0-line2.y1) - (line1.y0-line1.y1)*(line2.x0 - line2.x1)

    if bottom == 0:
        return ( -50,-50)

    a = det(line1.x0, line1.y0, line1.x1, line1.y1)
    b = det(line1.x0, 1, line1.x1, 1)
    c = det(line2.x0, line2.y0, line2.x1, line2.y1)
    d = det(line2.x0, 1, line2.x1, 1)
    top = det(a,b,c,d)
    Px = abs((int)(top/bottom))

    a = det(line1.x0, line1.y0, line1.x1, line1.y1)
    b = det(line1.y0, 1, line1.y1, 1)
    c = det(line2.x0, line2.y0, line2.x1, line2.y1)
    d = det(line2.y0, 1, line2.y1, 1)
    top = det(a,b,c,d)
    Py = abs((int)(top/bottom))
    return (Px,Py)

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required = True, help = "Path to the image")
args = vars(ap.parse_args())
cap = cv2.VideoCapture(args["input"]) #get the frame or image

cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
frames = -1
# iterate through all frames
while(cap.isOpened()):
    frames += 1
    hasframe, img = cap.read() #capture frame
    # see if frame exists, if it doesnt we are out of frames or there is another error
    try:
        size = img.shape
    except:
        print("out of frames")
        break
    # show every 30th frame, break if no images left
    if (frames % 30.0==0):
        # convert to grey scale and create Hough Lines
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        median = cv2.medianBlur(gray, 5)
        gaussian = cv2.GaussianBlur(median, (5,5), cv2.BORDER_DEFAULT)
        edges = cv2.Canny(gaussian,150,200)# The parameters are the thresholds for Canny
        lines = cv2.HoughLines(edges,0.5,0.01,230) # The parameters are accuracies and threshold

        # if there are no hough lines then skip the rest of the operations
        try:
            num = len(lines)
        except:
            print("No Hough Lines")
            continue

        # go from rho and theta coordinates to x and y coordinates of the begaining and end of the line and draw
        # rho,theta --> (x,y)start & (x,y)end
        line_points = []
        for n in range(num):
            rho, theta = lines[n][0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + size[1]*(-b))
            y1 = int(y0 + size[0]*(a))
            x2 = int(x0 - size[1]*(-b))
            y2 = int(y0 - size[0]*(a))

            cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

            line_points.append(line(x0,y0,x1,y1))

        cv2.imwrite('houghlines1.jpg',img)

        # calculate intersections for all lines and draw intersection
        # O(n^2)
        for n in range(num-1):
            for j in range(n+1,num):
                intersection = line_intersection(line_points[n],line_points[j])
                if(intersection[0]<3000 and intersection[1]<3000):
                    cv2.circle(img, intersection, 25, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cv2.imshow('Display', img)
        cv2.waitKey(0)
        cv2.imwrite('intersection.jpg',img)
    print(frames)
cv2.waitKey(0)
cv2.destroyAllWindows()
