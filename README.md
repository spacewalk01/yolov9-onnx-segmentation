<div align="center">

YOLOv9 ONNX Segmentation
===========================

[![python](https://img.shields.io/badge/python-3.10.12-green)](https://www.python.org/downloads/release/python-31012/)
[![cuda](https://img.shields.io/badge/cuda-11.6-green)](https://developer.nvidia.com/cuda-downloads)
[![mit](https://img.shields.io/badge/license-MIT-blue)](https://github.com/spacewalk01/depth-anything-tensorrt/blob/main/LICENSE)

</div>

Instance and panoptic segmentation using yolov9 in onnxruntime.

<p align="center">
  <img src="onnxruntime/result.jpg" width="720px" />
</p>


## 🚀 Quick Start

```
# infer 
python main.py --model <onnx model> --input <image or folder or video> 
```

## 👏 Acknowledgement

This project is based on the following projects:
- [YOLOv9](https://github.com/WongKinYiu/yolov9) - YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information.
