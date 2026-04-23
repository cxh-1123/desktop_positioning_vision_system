Desktop Positioning Vision System - cup detection prep
Project root: D:\Cxh\desktop_positioning_vision_system
Current image directory: D:\Cxh\desktop_positioning_vision_system\raw_images\cup
Current image count: 23
Current detected YOLO label count: 0
Current missing label count: 23

Current label discovery status:
- No usable YOLO labels found yet.

Next step: annotate images in YOLO format
1) Use an annotation tool (LabelImg/Label Studio/Roboflow/etc.).
2) Define single class id 0 for cup.
3) For each image, create a same-name .txt file in the same folder as image now, or later move them for training split.

YOLO label format (one object per line):
class_id x_center y_center width height
- All coordinates are normalized to [0,1] relative to image size.
- For this single-class task, class_id must be 0.
Example:
0 0.512 0.476 0.221 0.338

Where to place labels after annotation:
- Recommended raw co-location: D:\Cxh\desktop_positioning_vision_system\raw_images\cup (same basename as image).

How to continue training after labels are complete:
- Re-run data prep/split to refresh dataset/cup/images and dataset/cup/labels.
- Then run:
  yolo detect train data=D:/Cxh/desktop_positioning_vision_system/dataset/cup/cup.yaml model=yolov8n.pt epochs=50 imgsz=640 batch=auto project=D:/Cxh/desktop_positioning_vision_system/outputs/train_runs name=cup_exp
