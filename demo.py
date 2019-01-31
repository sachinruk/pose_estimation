import cv2
import time
import numpy as np
from random import randint

from keyPoints import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--input", help="Name of input video", type=str)
parser.add_argument("--t", help="Time of video to extract", type=int, default=30)
parser.add_argument("--output", help="Name of output video", type=str, default='output.avi')
args = parser.parse_args()

input_source = args.input
dest_vid = args.output
t_len = args.t

cap = cv2.VideoCapture(input_source)
hasFrame, frame = cap.read()
frameRate = int(cap.get(cv2.CAP_PROP_FPS))

vid_writer = cv2.VideoWriter(dest_vid,
                             cv2.VideoWriter_fourcc('M','J','P','G'), 
                             frameRate, 
                             (frame.shape[1],frame.shape[0]))

protoFile = "pose/coco/pose_deploy_linevec.prototxt"
weightsFile = "pose/coco/pose_iter_440000.caffemodel"
nPoints = 18
# COCO Output Format
keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 
                    'L-Sho', 'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 
                    'R-Ank', 'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye', 
                    'L-Eye', 'R-Ear', 'L-Ear']





colors = [ [0,100,255], [0,100,255], [0,255,255], [0,100,255], [0,255,255], [0,100,255],
         [0,255,0], [255,200,100], [255,0,255], [0,255,0], [255,200,100], [255,0,255],
         [0,0,255], [255,0,0], [200,200,0], [255,0,0], [200,200,0], [0,0,0]]


frames = frameRate * t_len
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
frameWidth = frame.shape[1]
frameHeight = frame.shape[0]

# Fix the input Height and get the width according to the Aspect Ratio
inHeight = 368
inWidth = int((inHeight/frameHeight)*frameWidth)
for i in range(frames):
    if not hasFrame:
        break
    t = time.time()
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                            (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inpBlob)
    output = net.forward()
    
    detected_keypoints = []
    keypoints_list = np.zeros((0,3))
    keypoint_id = 0
    threshold = 0.1

    for part in range(nPoints):
        probMap = output[0,part,:,:]
        probMap = cv2.resize(probMap, (frame.shape[1], frame.shape[0]))
        keypoints = getKeypoints(probMap, threshold)
        keypoints_with_id = []
        # for i in range(len(keypoints)):
        #     keypoints_with_id.append(keypoints[i] + (keypoint_id,))
        #     keypoints_list = np.vstack([keypoints_list, keypoints[i]])
        #     keypoint_id += 1
        for keypoint in keypoints:
            keypoints_with_id.append(keypoint + (keypoint_id,))
            keypoints_list = np.vstack([keypoints_list, keypoint])
            keypoint_id += 1

        detected_keypoints.append(keypoints_with_id)

    frameClone = frame.copy()

    valid_pairs, invalid_pairs = getValidPairs(output, detected_keypoints, frameWidth, frameHeight)
    personwiseKeypoints = getPersonwiseKeypoints(valid_pairs, invalid_pairs, keypoints_list)

    for idx in range(17):
        for n in range(len(personwiseKeypoints)):
            index = personwiseKeypoints[n][np.array(POSE_PAIRS[idx])]
            if -1 in index:
                continue
            B = np.int32(keypoints_list[index.astype(int), 0])
            A = np.int32(keypoints_list[index.astype(int), 1])
            cv2.line(frameClone, (B[0], A[0]), (B[1], A[1]), colors[idx], 3, cv2.LINE_AA)
    
    hasFrame, frame = cap.read()
    vid_writer.write(frameClone)
    print("\rProcessed frame in {:.4f}s for frame {}".format(time.time() - t, i+1), end="")

vid_writer.release()