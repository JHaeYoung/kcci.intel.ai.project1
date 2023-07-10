#%%
import collections
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from IPython import display
from numpy.lib.stride_tricks import as_strided
from openvino.runtime import Core

from decoder import OpenPoseDecoder

# sys.path.append("../utils")
# import notebook_utils as utils

# The name of the model from Open Model Zoo.
model_name = "human-pose-estimation-0001.xml"

# Initialize OpenVINO Runtime
ie = Core()
# Read the network from a file.
model = ie.read_model(model=model_name)
# Let the AUTO device decide where to load the model (you can use CPU, GPU or MYRIAD as well).
compiled_model = ie.compile_model(model=model, device_name="AUTO", config={"PERFORMANCE_HINT": "LATENCY"})

# Get the input and output names of nodes.
input_layer = compiled_model.input(0)
output_layers = compiled_model.outputs

# Get the input size.
height, width = list(input_layer.shape)[2:]

input_layer.any_name, [o.any_name for o in output_layers]

decoder = OpenPoseDecoder()

# 2D pooling in numpy (from: https://stackoverflow.com/a/54966908/1624463)
def pool2d(A, kernel_size, stride, padding, pool_mode="max"):
    """
    2D Pooling

    Parameters:
        A: input 2D array
        kernel_size: int, the size of the window
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max' or 'avg'
    """
    # Padding
    A = np.pad(A, padding, mode="constant")

    # Window view of A
    output_shape = (
        (A.shape[0] - kernel_size) // stride + 1,
        (A.shape[1] - kernel_size) // stride + 1,
    )
    kernel_size = (kernel_size, kernel_size)
    A_w = as_strided(
        A,
        shape=output_shape + kernel_size,
        strides=(stride * A.strides[0], stride * A.strides[1]) + A.strides
    )
    A_w = A_w.reshape(-1, *kernel_size)

    # Return the result of pooling.
    if pool_mode == "max":
        return A_w.max(axis=(1, 2)).reshape(output_shape)
    elif pool_mode == "avg":
        return A_w.mean(axis=(1, 2)).reshape(output_shape)


# non maximum suppression
def heatmap_nms(heatmaps, pooled_heatmaps):
    return heatmaps * (heatmaps == pooled_heatmaps)


# Get poses from results.
def process_results(img, pafs, heatmaps):
    # This processing comes from
    # https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/common/python/models/open_pose.py
    pooled_heatmaps = np.array(
        [[pool2d(h, kernel_size=3, stride=1, padding=1, pool_mode="max") for h in heatmaps[0]]]
    )
    nms_heatmaps = heatmap_nms(heatmaps, pooled_heatmaps)

    # Decode poses.
    poses, scores = decoder(heatmaps, nms_heatmaps, pafs)
    output_shape = list(compiled_model.output(index=0).partial_shape)
    output_scale = img.shape[1] / output_shape[3].get_length(), img.shape[0] / output_shape[2].get_length()
    # Multiply coordinates by a scaling factor.
    poses[:, :, :2] *= output_scale
    return poses, scores


colors = ((255, 0, 0), (255, 0, 255), (170, 0, 255), (255, 0, 85), (255, 0, 170), (85, 255, 0),
          (255, 170, 0), (0, 255, 0), (255, 255, 0), (0, 255, 85), (170, 255, 0), (0, 85, 255),
          (0, 255, 170), (0, 0, 255), (0, 255, 255), (85, 0, 255), (0, 170, 255))
# colors = ( (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0),
#           (0, 0, 0), (0, 0, 0),(255, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0),
#           (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0))

default_skeleton = ((15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7),
                    (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6))
# 0 = 코, 1= 오른쪽 눈 , 2=왼쪽 눈 , 3=오른쪽 귀, 4.왼쪽 귀, 5.오른쪽어깨, 6.왼쪽어깨,
# 7.오른쪽팔꿈치,8왼쪽팔꿈치,9.오른쪽손목,10.왼쪽손목,11.오른쪽허리, 
# 12.왼쪽허리 13.오른쪽 무릎, 14.왼쪽무릎 15.오른쪽발목 16.왼쪽발목


cnt = 0
def draw_poses(img, poses, point_score_threshold, skeleton=default_skeleton):
    if poses.size == 0:
        return img
    global cnt
    img_limbs = np.copy(img)
    for pose in poses:
        points = pose[:, :2].astype(np.int32)
        points_scores = pose[:, 2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (0, 0, 255)  # BGR color tuple
        thickness = 2
        # Draw joints.
        for i, (p, v) in enumerate(zip(points, points_scores)):
            if v > point_score_threshold:
                cv2.circle(img, tuple(p), 1, colors[i], 2)
                #cv2.putText(img,str(tuple(p)),tuple(p),font, font_scale, color, thickness)
                # if cnt == 30 : 
                #     print(tuple(p))
                #     cnt=0
                # else:
                #     cnt=cnt+1       
        if points[9,1] < points[0,1] and points[10,1]>points[6,1]:
            cv2.putText(img,str(points[9,:]),points[9,:],font, font_scale, color, thickness)
            cv2.putText(img,str(points[0,:]),points[0,:],font, font_scale, color, thickness)
            cv2.imwrite("freedom.jpg",img)
        if points[9,1] < points[0,1] and  points[10,1] < points[0,1] and points[9,0] > points[7,0] and points[10,0] < points[8,0]:
            cv2.putText(img,str(points[9,:]),points[9,:],font, font_scale, (255, 0, 0), thickness)
            cv2.putText(img,str(points[0,:]),points[0,:],font, font_scale, (255, 0, 0), thickness)
            cv2.putText(img,str(points[10,:]),points[10,:],font, font_scale, (255, 0, 0), thickness)
            cv2.putText(img,str(points[7,:]),points[7,:],font, font_scale, (255, 0, 0), thickness)
            cv2.imwrite("japan.jpg",img)
        if points[9,1] < points[0,1] and  points[10,1] < points[0,1] and points[9,0] < points[7,0] and points[10,0] > points[8,0]:
            cv2.putText(img,str(points[9,:]),points[9,:],font, font_scale, (0, 255, 0), thickness)
            cv2.putText(img,str(points[0,:]),points[0,:],font, font_scale, (0, 255, 0), thickness)
            cv2.putText(img,str(points[10,:]),points[10,:],font, font_scale, (0, 255, 0), thickness)
            cv2.putText(img,str(points[7,:]),points[7,:],font, font_scale, (0, 255, 0), thickness)
            cv2.imwrite("egypt.jpg",img)
            

        # Draw limbs.
    #     for i, j in skeleton:
    #         if points_scores[i] > point_score_threshold and points_scores[j] > point_score_threshold:
    #             cv2.line(img_limbs, tuple(points[i]), tuple(points[j]), color=colors[j], thickness=4)
    # cv2.addWeighted(img, 0.4, img_limbs, 0.6, 0, dst=img)
    # cv2.imwrite("points_line.jpg",img)
    return img

# Main processing function to run pose estimation.
def run_pose_estimation(flip=False, use_popup=False, skip_first_frames=0):
    pafs_output_key = compiled_model.output("Mconv7_stage2_L1")
    heatmaps_output_key = compiled_model.output("Mconv7_stage2_L2")
    player = None
    
    # Open the webcam
    cap = cv2.VideoCapture(0)

    if use_popup:
        title = "Press ESC to Exit"
        cv2.namedWindow(title, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)

    processing_times = collections.deque()

    while True:
        # Read the frame from the webcam
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to read the frame")
            break

        # If the frame is larger than full HD, reduce size to improve the performance.
        scale = 1280 / max(frame.shape)
        if scale < 1:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        # Resize the image and change dims to fit neural network input.
        # (see https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/human-pose-estimation-0001)
        input_img = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        # Create a batch of images (size = 1).
        input_img = input_img.transpose((2,0,1))[np.newaxis, ...]

        # Measure processing time.
        start_time = time.time()
        # Get results.
        results = compiled_model([input_img])
        stop_time = time.time()

        pafs = results[pafs_output_key]
        heatmaps = results[heatmaps_output_key]
        # Get poses from network results.
        poses, scores = process_results(frame, pafs, heatmaps)

        # Draw poses on a frame.
        frame = draw_poses(frame, poses, 0.1)
        #cv2.imwrite("picture.jpg",frame)

        processing_times.append(stop_time - start_time)
        # Use processing times from last 200 frames.
        if len(processing_times) > 200:
            processing_times.popleft()

        _, f_width = frame.shape[:2]
        # mean processing time [ms]
        processing_time = np.mean(processing_times) * 1000
        fps = 1000 / processing_time
        cv2.putText(frame, f"Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)", (20, 40),
                    cv2.FONT_HERSHEY_COMPLEX, f_width / 1000, (0, 0, 255), 1, cv2.LINE_AA)

        # Use this workaround if there is flickering.
        if use_popup:
            cv2.imshow(title, frame)
            key = cv2.waitKey(1)
            # escape = 27
            if key == 27:
                break
        else:
            # Display the frame
            cv2.imshow("Webcam", frame)

            # Check for the 'q' key to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
     # Release the webcam and close any open windows
    cap.release()
    cv2.destroyAllWindows()

run_pose_estimation(flip=True, use_popup=False)

# %%
