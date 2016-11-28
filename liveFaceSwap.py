# Command to use for docker. Change <path_to_this_file> with actual path
# docker run -v <path_to_this_file>:/mnt/host -p 9000:9000 -p 8000:8000 -t -i bamos/openface /bin/bash

import os
import sys
import time
import cv2
import numpy as np
import openface
import dlib
from twisted.internet import task
from twisted.internet import reactor

# Initialize paths and directories
filepath = os.path.dirname(os.path.abspath(__file__))
if (len(sys.argv) >= 2):
    modelDir = os.path.join(sys.argv[2])
else:
    modelDir = os.path.join(os.path.expanduser('~'),'Downloads','openface','models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

# Initialize models and neural network
dlibModel = os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat")
networkModel = os.path.join(openfaceModelDir, "nn4.small2.v1.t7")
##net = openface.TorchNeuralNet(networkModel, imgDim=imageDimension, cuda=useCuda)

if (os.path.isfile(dlibModel) == False):
    print("dlib model not found at " + dlibModel)
    sys.exit()

align = openface.AlignDlib(dlibModel)

# Global variables
imageDimension = 100
verbose = True
FPS = 0.1
timeout = 0.001
# Set up webcam
video_capture = cv2.VideoCapture(0)

def isBlack(p):
    return p[0] <= 0 and p[1] <= 0 and p[2] <= 10

def runCode():
    # Set up: load the image and convert to RGB format for processing
    ret, frame = video_capture.read()
    if (frame != None):
        rgbImg = frame

        origRGB = np.copy(frame) #Save the original
        # Get all the faces in the image
        # Provides an array of bounding boxes
        # Each bounding box contains coordinates for top left and bottom right of the face
        cv2.imshow('Before', frame)
        cv2.waitKey(1)
        bbArr = align.getAllFaceBoundingBoxes(rgbImg)
        print("Found " + str(len(bbArr)) + " faces")
        # For each face in the array
        index = 0
        for bb in bbArr:
            # Get the landmarks of the face.
            # Returns an array of coordinates.
            # Each coordinate represents a certain point on the face.
            # To get a list of points, see:
            # http://openface-api.readthedocs.io/en/latest/openface.html#openface-aligndlib-class
            print("    Processing Face: " + str(index+1))
            landmarks = align.findLandmarks(rgbImg, bb)
            outerface = landmarks[0:16] + landmarks[26:22:-1] + landmarks[21:17:-1]
            pts = np.asarray(outerface)
            mask = np.zeros((rgbImg.shape[0], rgbImg.shape[1]), dtype=np.uint8)
            cv2.fillConvexPoly(mask, pts, 255)

            # Align the face and get make an image
            face = cv2.bitwise_and(rgbImg, rgbImg, mask=mask)

            face = align.align(imageDimension, face, bb, skipMulti=False, landmarkIndices=[0,16,8])
         
            if (index+1 < len(bbArr)):
                nextIndex = index+1
            else:
                nextIndex = 0

            otherImgLandmarks = align.findLandmarks(rgbImg, bbArr[nextIndex])
            center = otherImgLandmarks[33]
            right = abs(otherImgLandmarks[16][0] - center[0])
            left = abs(otherImgLandmarks[0][0] - center[0])
            top = abs(otherImgLandmarks[24][1] - center[1])
            bottom = abs(center[1] - otherImgLandmarks[8][1])
            (x, y) = center

            resizedFace = cv2.resize(face, (left+right, top+bottom))

            for i in range(-top, bottom):
                for j in range(-left, right):
                    if (np.all(resizedFace[i+top,j+left] >= 8)):
                        origRGB[y+i,x+j] = resizedFace[i+top,j+left]

            #origRGB[y-top:y+bottom,x-left:x+right] = cv2.inpaint(origRGB[y-top:y+bottom, x-left: x+right], inpaintMask, 2, cv2.INPAINT_NS)
            print("    Processed Face: " + str(index+1))
            index+=1
        cv2.imshow('After', origRGB)
        cv2.waitKey(1)
# Main function
if __name__ == '__main__':

    time.sleep(2)
    l = task.LoopingCall(runCode)
    l.start(timeout)

    reactor.run()


    video_capture.release()
    cv2.destroyAllWindows()
