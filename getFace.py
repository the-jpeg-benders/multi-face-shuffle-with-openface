# Command to use for docker. Change <path_to_this_file> with actual path
# docker run -v <path_to_this_file>:/mnt/host -p 9000:9000 -p 8000:8000 -t -i bamos/openface /bin/bash

import os
import time
import cv2
import numpy as np
import openface

# Initialize paths and directories
filepath = os.path.dirname(os.path.abspath(__file__))
imgPath = os.path.join(filepath,'mytestgroup.jpg')
modelDir = os.path.join('/root','openface','models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

# Initialize models and neural network
dlibModel = os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat")
networkModel = os.path.join(openfaceModelDir, "nn4.small2.v1.t7")
align = openface.AlignDlib(dlibModel)
##net = openface.TorchNeuralNet(networkModel, imgDim=imageDimension, cuda=useCuda)

# Global variables
imageDimension = 100
verbose = True


# Main function
if __name__ == '__main__':

    # Set up: load the image and convert to RGB format for processing
    bgrImg = cv2.imread(imgPath)
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    # Get all the faces in the image
    # Provides an array of bounding boxes
    # Each bounding box contains coordinates for top left and bottom right of the face
    bbArr = align.getAllFaceBoundingBoxes(rgbImg)

    # For each face in the array
    i = 1
    for bb in bbArr:

        # Get the landmarks of the face.
        # Returns an array of coordinates.
        # Each coordinate represents a certain point on the face.
        # To get a list of points, see:
        # http://openface-api.readthedocs.io/en/latest/openface.html#openface-aligndlib-class
        landmarks = align.findLandmarks(rgbImg, bb)

        # Align the face and get make an image
        alignedImg = align.align(imageDimension, rgbImg, bb, skipMulti=False)

        print(bb)
        print("\n\n")

        # Convert aligned image to BGR format for save
        alignedImgBGR = cv2.cvtColor(alignedImg, cv2.COLOR_RGB2BGR)

        # Save just the face on to the disk
        cv2.imwrite('output' + str(i) + '.jpg', alignedImgBGR)
    

