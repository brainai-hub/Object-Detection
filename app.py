import streamlit as st
import PIL
import cv2
import numpy as np
import io
import time
from pathlib import Path
import collections
import openvino as ov

# Function to process object detection results
def process_results(frame, results, thresh=0.6):
    h, w = frame.shape[:2]
    results = results.squeeze()
    boxes, labels, scores = [], [], []
    for _, label, score, xmin, ymin, xmax, ymax in results:
        boxes.append(tuple(map(int, (xmin * w, ymin * h, (xmax - xmin) * w, (ymax - ymin) * h))))
        labels.append(int(label))
        scores.append(float(score))
    
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=thresh, nms_threshold=0.6)
    if len(indices) == 0:
        return []
    return [(labels[idx], scores[idx], boxes[idx]) for idx in indices.flatten()]

# Drawing bounding boxes on detected objects
def draw_boxes(frame, boxes):
    colors = cv2.applyColorMap(src=np.arange(0, 255, 255 / 80, dtype=np.float32).astype(np.uint8), colormap=cv2.COLORMAP_RAINBOW).squeeze()
    for label, score, box in boxes:
        color = tuple(map(int, colors[label]))
        x2 = box[0] + box[2]
        y2 = box[1] + box[3]
        cv2.rectangle(img=frame, pt1=box[:2], pt2=(x2, y2), color=color, thickness=3)
        cv2.putText(img=frame, text=f"{label} {score:.2f}", org=(box[0] + 10, box[1] + 30), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=frame.shape[1] / 1000, color=color, thickness=1, lineType=cv2.LINE_AA)
    return frame

# Run object detection on video or webcam input
def run_object_detection(video_source, conf_threshold):
    core = ov.Core()
    device = "CPU"
    model_path = download_and_convert_model()
    model = core.read_model(model=model_path)
    compiled_model = core.compile_model(model=model, device_name=device)
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)
    height, width = list(input_layer.shape)[1:3]
    
    camera = cv2.VideoCapture(video_source)
    st_frame = st.empty()
    
    while camera.isOpened():
        ret, frame = camera.read()
        if not ret:
            break

        input_img = cv2.resize(frame, (width, height))
        input_img = input_img[np.newaxis, ...]
        results = compiled_model([input_img])[output_layer]
        boxes = process_results(frame, results, conf_threshold)
        frame = draw_boxes(frame, boxes)
        st_frame.image(frame, channels="BGR")

    camera.release()

# Streamlit Interface
st.set_page_config(page_title="Facial & Object Detection", page_icon=":sun_with_face:", layout="centered", initial_sidebar_state="expanded")

st.title("Facial & Object Detection :sun_with_face:")
st.sidebar.header("Type")
source_radio = st.sidebar.radio("Select Source", ["WEBCAM"])

st.sidebar.header("Confidence")
conf_threshold = float(st.sidebar.slider("Select the Confidence Threshold", 10, 100, 20)) / 100

input_file = None
temporary_location = None

# Video or Webcam Processing Section
if source_radio in ["WEBCAM"]:
    if source_radio == "WEBCAM":
        run_object_detection(0, conf_threshold)
