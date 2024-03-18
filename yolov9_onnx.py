# -*- coding: utf-8 -*-

import math
import time
import cv2
import numpy as np
import onnxruntime
import argparse
import os
from util import xywh2xyxy, nms, draw_detections, sigmoid, imread_from_url


class YOLOSeg:

    def __init__(self, path, conf_thres=0.7, iou_thres=0.5, num_masks=32):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.num_masks = num_masks

        # Initialize model
        self.initialize_model(path)

    def __call__(self, image):
        return self.segment_objects(image)

    def initialize_model(self, path):
        self.session = onnxruntime.InferenceSession(path,
                                                    providers=['CUDAExecutionProvider',
                                                               'CPUExecutionProvider'])
        # Get model info
        self.get_input_details()
        self.get_output_details()

    def segment_objects(self, image):
        input_tensor = self.prepare_input(image)

        # Perform inference on the image
        outputs = self.inference(input_tensor)

        self.boxes, self.scores, self.class_ids, mask_pred = self.process_box_output(outputs[0])
        self.mask_maps = self.process_mask_output(mask_pred, outputs[1])

        return self.boxes, self.scores, self.class_ids, self.mask_maps

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor

    def inference(self, input_tensor):
        start = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})

        # print(f"Inference time: {(time.perf_counter() - start)*1000:.2f} ms")
        return outputs

    def process_box_output(self, box_output):

        predictions = np.squeeze(box_output).T
        num_classes = box_output.shape[1] - self.num_masks - 4

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:4+num_classes], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], [], np.array([])

        box_predictions = predictions[..., :num_classes+4]
        mask_predictions = predictions[..., num_classes+4:]

        # Get the class with the highest confidence
        class_ids = np.argmax(box_predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(box_predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = nms(boxes, scores, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices], mask_predictions[indices]

    def process_mask_output(self, mask_predictions, mask_output):

        if mask_predictions.shape[0] == 0:
            return []

        mask_output = np.squeeze(mask_output)

        # Calculate the mask maps for each box
        num_mask, mask_height, mask_width = mask_output.shape  # CHW
        masks = sigmoid(mask_predictions @ mask_output.reshape((num_mask, -1)))
        masks = masks.reshape((-1, mask_height, mask_width))

        # Downscale the boxes to match the mask size
        scale_boxes = self.rescale_boxes(self.boxes,
                                   (self.img_height, self.img_width),
                                   (mask_height, mask_width))

        # For every box/mask pair, get the mask map
        mask_maps = np.zeros((len(scale_boxes), self.img_height, self.img_width))
        blur_size = (int(self.img_width / mask_width), int(self.img_height / mask_height))
        for i in range(len(scale_boxes)):

            scale_x1 = int(math.floor(scale_boxes[i][0]))
            scale_y1 = int(math.floor(scale_boxes[i][1]))
            scale_x2 = int(math.ceil(scale_boxes[i][2]))
            scale_y2 = int(math.ceil(scale_boxes[i][3]))

            x1 = int(math.floor(self.boxes[i][0]))
            y1 = int(math.floor(self.boxes[i][1]))
            x2 = int(math.ceil(self.boxes[i][2]))
            y2 = int(math.ceil(self.boxes[i][3]))

            scale_crop_mask = masks[i][scale_y1:scale_y2, scale_x1:scale_x2]
            crop_mask = cv2.resize(scale_crop_mask,
                              (x2 - x1, y2 - y1),
                              interpolation=cv2.INTER_CUBIC)

            crop_mask = cv2.blur(crop_mask, blur_size)

            crop_mask = (crop_mask > 0.5).astype(np.uint8)
            mask_maps[i, y1:y2, x1:x2] = crop_mask

        return mask_maps

    def extract_boxes(self, box_predictions):
        # Extract boxes from predictions
        boxes = box_predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes,
                                   (self.input_height, self.input_width),
                                   (self.img_height, self.img_width))

        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)

        # Check the boxes are within the image
        boxes[:, 0] = np.clip(boxes[:, 0], 0, self.img_width)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, self.img_height)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, self.img_width)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, self.img_height)

        return boxes

    def draw_detections(self, image, draw_scores=True, mask_alpha=0.4):
        return draw_detections(image, self.boxes, self.scores,
                               self.class_ids, mask_alpha)

    def draw_masks(self, image, draw_scores=True, mask_alpha=0.5):
        return draw_detections(image, self.boxes, self.scores,
                               self.class_ids, mask_alpha, mask_maps=self.mask_maps)

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

    @staticmethod
    def rescale_boxes(boxes, input_shape, image_shape):
        # Rescale boxes to original image dimensions
        input_shape = np.array([input_shape[1], input_shape[0], input_shape[1], input_shape[0]])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([image_shape[1], image_shape[0], image_shape[1], image_shape[0]])

        return boxes



def is_image(file_path):
    """Check if the given path points to an image."""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
    return any(file_path.lower().endswith(ext) for ext in image_extensions)

def is_video(file_path):
    """Check if the given path points to a video."""
    video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv']
    return any(file_path.lower().endswith(ext) for ext in video_extensions)

def detect_input_type(input_path):
    """Detect the type of input based on the provided path."""
    if os.path.isfile(input_path):
        if is_image(input_path):
            return 'image'
        elif is_video(input_path):
            return 'video'
        else:
            return None  # Not an image or video
    elif os.path.isdir(input_path):
        return 'folder'
    else:
        return None  # Not a valid file or directory

if __name__ == '__main__':
    # Initialize argument parser
    parser = argparse.ArgumentParser(description='Object detection using YOLOv8')
    parser.add_argument('--model', type=str, default="gelan-c-pan.onnx", help='Path to the ONNX model')
    parser.add_argument('--input', type=str, default="video", help='Input type: "image", "folder", or "video"')
    args = parser.parse_args()

    # Initialize YOLOv8 Instance Segmentator
    yoloseg = YOLOSeg(args.model, conf_thres=0.3, iou_thres=0.5)

    # Create an output folder
    output_folder = "results"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    input_type = detect_input_type(args.input)
    if input_type == 'image':
        img = cv2.imread(args.input)
        print("image: ", args.input)
        # Detect objects in the image
        yoloseg(img)
        # Draw detections
        combined_img = yoloseg.draw_masks(img)
        output_path = os.path.join(output_folder, "result.jpg")
        cv2.imwrite(output_path, combined_img)
        cv2.imshow("Output", combined_img)
        cv2.waitKey(0)
    elif input_type == 'folder':
        # Loop through image files in the given folder
        for filename in os.listdir(args.input):
            if filename.endswith(".jpg") or filename.endswith(".png"):  # Assuming images are jpg or png format
                img_path = os.path.join(args.input, filename)
                print("folder:", img_path)
                img = cv2.imread(img_path)
                # Detect objects in the image
                yoloseg(img)
                # Draw detections
                combined_img = yoloseg.draw_masks(img)
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, combined_img)
                cv2.imshow("Output", combined_img)
                cv2.waitKey(0)
    elif input_type == 'video':
        cap = cv2.VideoCapture(args.input)  # Replace with the actual video path
        frame_width = int(cap.get(3))  # Frame width
        frame_height = int(cap.get(4))  # Frame height
        print(frame_width, frame_height)
        fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
        out = cv2.VideoWriter(os.path.join(output_folder, "result.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
        i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Detect objects in the frame
            yoloseg(frame)
            # Draw detections
            combined_frame = yoloseg.draw_masks(frame)
            print(combined_frame.shape)
            cv2.imshow("Output", combined_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if i > 100:
                break
            i += 1
        cap.release()
        cv2.destroyAllWindows()
