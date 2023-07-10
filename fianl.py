#%%
from threading import Thread
import os
from collections import namedtuple
import collections
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import vstack as vs

from IPython.display import HTML, FileLink, display
from IPython import display
from numpy.lib.stride_tricks import as_strided
from openvino.runtime import Core
from openvino.tools import mo


import torch
import matplotlib.pyplot as plt

from decoder import OpenPoseDecoder

# Import local modules
sys.path.append("../utils")

from notebook_utils import load_image
from model.u2net import U2NET, U2NETP

cnt =0

#------여기까지 seg 모델

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
        # Draw joints.
        for i, (p, v) in enumerate(zip(points, points_scores)):
            if v > point_score_threshold:
                cv2.circle(img, tuple(p), 1, colors[i], 2)
        #Draw limbs.
        for i, j in skeleton:
            if points_scores[i] > point_score_threshold and points_scores[j] > point_score_threshold:
                cv2.line(img_limbs, tuple(points[i]), tuple(points[j]), color=colors[j], thickness=4)
    cv2.addWeighted(img, 0.4, img_limbs, 0.6, 0, dst=img)
    return img
MAXCNT = 60
def detect_pose(img, poses)->int:
    if poses.size == 0:
        return 0
    global cnt
    msg = "take a picture~"
    for pose in poses:
        points = pose[:, :2].astype(np.int32)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (0, 0, 255)  # BGR color tuple
        thickness = 2
        # Draw joints.
        if points[9,0] - points[10,0] < 50 and points[9,1] - points[10,1] < 50 and points[7,0] - points[8,0] < 50 and points[7,1] - points[8,1] < 50 :
            if cnt == MAXCNT : 
                cv2.imwrite("pisa.jpg",img)
                cv2.putText(img,msg,(100,100),font, font_scale, (0, 0, 0), thickness)
                cnt=0
                return 4 #피사의 사탑(이탈리아) 4
            else:
                cnt=cnt+1
        if points[9,1] < points[0,1] and points[10,1]>points[6,1]:
            if cnt == MAXCNT : 
                cv2.imwrite("freedom.jpg",img)
                cv2.putText(img,msg,(100,100),font, font_scale, (0, 0, 0), thickness)
                cnt=0
                return 3 #자유의 여신상(미국) 3
            else:
                cnt=cnt+1

        if points[9,1] < points[0,1] and  points[10,1] < points[0,1] and points[9,0] > points[7,0] and points[10,0] < points[8,0]:
            if cnt == MAXCNT : 
                cv2.imwrite("glico.jpg",img)
                cv2.putText(img,msg,(100,100),font, font_scale, (0, 0, 0), thickness)
                cnt=0
                return 2 #글리코상(일본) 2
            else:
                cnt=cnt+1 
        if points[9,1] < points[0,1] and  points[10,1] < points[0,1] and points[9,0] < points[7,0] and points[10,0] > points[8,0]:
            
            if cnt == MAXCNT : 
                cv2.imwrite("pyramid.jpg",img)
                cv2.putText(img,msg,(100,100),font, font_scale, (0, 0, 0), thickness)
                cnt=0
                return 1 #피라미드(이집트) 1
            else:
                cnt=cnt+1 
        
    return 0

def do_composition(img, bgimg): #segmentation
    # Background removal
    # compostion
    # return
    model_config = namedtuple("ModelConfig", ["name",  "model", "model_args"]) #"url",
    # 다양한 모델 중에서 가볍고 경량화된 모델을 선택하여 실행 시간과 메모리 사용량을 줄이기 위함
  
    u2net_human_seg = model_config(
        name="u2net_human_seg",
        # url="https://drive.google.com/uc?id=1-Yg0cxgrNhHP-016FPdp902BR-kSsA4P",
        model=U2NET,
        model_args=(3, 1),
    )

    # Set u2net_model to one of the three configurations listed above.
    u2net_model = u2net_human_seg

    MODEL_DIR = "model"
    model_path = Path(MODEL_DIR) / u2net_model.name / Path(u2net_model.name).with_suffix(".pth")
    #               model       /   u2net_lite      /     u2net_lite.pth  

    # Load the model.
    # 모델을 로드하고 다운로드한 가중치를 모델에 설정하는 부분
    net = u2net_model.model(*u2net_model.model_args)
    net.eval()

    # Load the weights.
    print(f"Loading model weights from: '{model_path}'")
    net.load_state_dict(state_dict=torch.load(model_path, map_location="cpu"))

    # 모델을 ONNX 형식으로 변환
    torch.onnx.export(net, torch.zeros((1,3,512,512)), "u2net.onnx")

    # ONNX 모델을 OpenVINO IR 모델로 변환하고 컴파일
    model_ir = mo.convert_model(
        "u2net.onnx",
        mean_values=[123.675, 116.28 , 103.53],
        scale_values=[58.395, 57.12 , 57.375],
        compress_to_fp16=True
    )


    IMAGE_URI = img
    image = cv2.cvtColor(
        src=load_image(IMAGE_URI),
        code=cv2.COLOR_BGR2RGB,
    )

    resized_image = cv2.resize(src=image, dsize=(512, 512))
    # Convert the image shape to a shape and a data type expected by the network
    # for OpenVINO IR model: (1, 3, 512, 512).
    input_image = np.expand_dims(np.transpose(resized_image, (2, 0, 1)), 0)


    # Load the network to OpenVINO Runtime.
    ie = Core()
    compiled_model_ir = ie.compile_model(model=model_ir, device_name="CPU")
    # Get the names of input and output layers.
    input_layer_ir = compiled_model_ir.input(0)
    output_layer_ir = compiled_model_ir.output(0)

    # Do inference on the input image.
    start_time = time.perf_counter()
    result = compiled_model_ir([input_image])[output_layer_ir]
    end_time = time.perf_counter()
    print(
        f"Inference finished. Inference time: {end_time-start_time:.3f} seconds, "
        f"FPS: {1/(end_time-start_time):.2f}."
    )

    # Resize the network result to the image shape and round the values
    # to 0 (background) and 1 (foreground).
    # The network result has (1,1,512,512) shape. The `np.squeeze` function converts this to (512, 512).
    resized_result = np.rint(
        cv2.resize(src=np.squeeze(result), dsize=(image.shape[1], image.shape[0]))
    ).astype(np.uint8)

    # Create a copy of the image and set all background values to 255 (white).
    bg_removed_result = image.copy()
    bg_removed_result[resized_result == 0] = 255
    # fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 7))
    # ax[0].imshow(image)
    # ax[1].imshow(resized_result, cmap="gray")
    # ax[2].imshow(bg_removed_result)
    # for a in ax:
    #     a.axis("off")

    BACKGROUND_FILE = bgimg
    OUTPUT_DIR = "output"

    os.makedirs(name=OUTPUT_DIR, exist_ok=True)

    background_image = cv2.cvtColor(src=load_image(BACKGROUND_FILE), code=cv2.COLOR_BGR2RGB)
    background_image = cv2.resize(src=background_image, dsize=(image.shape[1], image.shape[0]))

    # Set all the foreground pixels from the result to 0
    # in the background image and add the image with the background removed.
    background_image[resized_result == 1] = 0
    new_image = background_image + bg_removed_result

    # Save the generated image.
    new_image_path = Path(f"./{OUTPUT_DIR}/{Path(IMAGE_URI).stem}-{Path(BACKGROUND_FILE).stem}.jpg")
    cv2.imwrite(filename=str(new_image_path), img=cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR))

    # Display the original image and the image with the new background side by side
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 7))
    ax[0].imshow(image)
    ax[1].imshow(new_image)
    for a in ax:
        a.axis("off")
    plt.show()


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
    detect_cnt =0
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
        detected_pose = detect_pose(frame, poses)
        frame = draw_poses(frame, poses,0.1)
        # Draw poses on a frame.
       
        # if type(detected_pose) == Array:
        # import pdb;
        # pdb.set_trace()
        global cnt 
        
        if  (detected_pose == 1):
            img = "pyramid.jpg"
            bg = "egypt.jpg"
            composition_thread = Thread(target=do_composition, args=(img, bg))
            composition_thread.start()
            print(f"detected_pose :{detected_pose} ")
            detect_cnt +=1

        elif (detected_pose == 2) :
            img = "glico.jpg"
            bg = "japan.jpg"
            composition_thread_1 = Thread(target=do_composition, args=(img, bg))        
            composition_thread_1.start()
            print(f"detected_pose :{detected_pose} ")
            detect_cnt +=1

        elif (detected_pose == 3):
            img = "freedom.jpg"
            bg = "USA.jpg"
            composition_thread_2 = Thread(target=do_composition, args=(img, bg))
            composition_thread_2.start()
            print(f"detected_pose :{detected_pose} ")
            detect_cnt +=1

        elif (detected_pose == 4):
            img = "pisa.jpg"
            bg = "italy.jpg"
            composition_thread_2 = Thread(target=do_composition, args=(img, bg))
            composition_thread_2.start()
            print(f"detected_pose :{detected_pose} ")
            detect_cnt +=1

        elif detected_pose == 0:
            if cnt == MAXCNT:
                print("없는 사진입니다.")
                print("detect_cnt : ", detect_cnt)
                cnt =0
            elif cnt !=MAXCNT:
                cnt+=1    
        
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
            # escape = 27q
            if key == 27:                
                break
        else:
            # Display the frame
            cv2.imshow("Webcam", frame)

            # Check for the 'q' key to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        if detect_cnt ==4 :            
            break                     
     # Release the webcam and close any open windows    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":    
    result_cnt =0    
    torch.cuda.empty_cache()
    run_pose_estimation(flip=True, use_popup=False)
    time.sleep(10)
    result = vs.photo_stack()
    cv2.imwrite(f"result{result_cnt}.jpg",result)
    img = cv2.imread(f"result{result_cnt}.jpg")
    resized = cv2.resize(img, (500,600))
    cv2.imshow(f"result{result_cnt}.jpg",resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    result_cnt +=1
        
# %%
