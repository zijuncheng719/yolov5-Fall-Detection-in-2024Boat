"""Run inference with a YOLOv5 model on images, videos, directories, streams

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import cv2
import argparse
import numpy as np
from pybaseutils import file_utils, image_utils
from pybaseutils.maker import maker_labelme
from demo import Yolov5Detector
from typing import List


class MakerLabelme(Yolov5Detector):
    def __init__(self,
                 weights='yolov5s.pt',  # model.pt path(s)
                 imgsz=640,  # inference size (pixels)
                 conf_thres=0.5,  # confidence threshold
                 iou_thres=0.5,  # NMS IOU threshold
                 max_det=1000,  # maximum detections per image
                 class_name=None,  # filter by class: --class 0, or --class 0 2 3
                 classes=None,  # filter by class: --class 0, or --class 0 2 3
                 agnostic_nms=False,  # class-agnostic NMS
                 augment=False,  # augmented inference
                 half=False,  # use FP16 half-precision inference
                 visualize=False,  # visualize features
                 batch_size=4,
                 device='cuda:0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                 fix_inputs=False,
                 ):
        super(MakerLabelme, self).__init__(weights=weights,  # model.pt path(s)
                                           imgsz=imgsz,  # inference size (pixels)
                                           conf_thres=conf_thres,  # confidence threshold
                                           iou_thres=iou_thres,  # NMS IOU threshold
                                           max_det=max_det,  # maximum detections per image
                                           class_name=class_name,  # filter by class: --class 0, or --class 0 2 3
                                           classes=classes,  # filter by class: --class 0, or --class 0 2 3
                                           agnostic_nms=agnostic_nms,  # class-agnostic NMS
                                           augment=augment,  # augmented inference
                                           half=half,  # use FP16 half-precision inference
                                           visualize=visualize,  # visualize features
                                           batch_size=batch_size,
                                           device=device,  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                                           fix_inputs=fix_inputs, )

    def detect(self, image: List[np.ndarray] or np.ndarray, vis: bool = False) -> List[List]:
        """
        :param image: 图像或者图像列表,BGR格式
        :param vis: 是否可视化显示检测结果
        :return: 返回检测结果[[List]], each bounding box is in [x1,y1,x2,y2,conf,cls] format.
        """
        if isinstance(image, np.ndarray): image = [image]
        dets = super().inference(image)
        if vis:
            self.draw_result(image, dets, delay=10)
        return dets

    def start_capture(self, video_file, out_dir, flag="", detect_freq=1, vis=True):
        """
        start capture video
        :param video_file: *.avi,*.mp4,...
        :param save_video: *.avi
        :param detect_freq:
        :return:
        """
        name = os.path.basename(video_file).split(".")[0]
        video_cap = image_utils.get_video_capture(video_file)
        width, height, numFrames, fps = image_utils.get_video_info(video_cap)
        detect_freq = fps // 2
        count = 0
        while True:
            if count % detect_freq == 0 and count > 0:
                # 设置抽帧的位置
                if isinstance(video_file, str): video_cap.set(cv2.CAP_PROP_POS_FRAMES, count)
                isSuccess, frame = video_cap.read()
                if not isSuccess:
                    break
                dets = self.detect(frame.copy(), vis=vis)
                image_name = "{}_{}_{:0=3d}.jpg".format(flag, name, count)
                path = file_utils.create_dir(out_dir, None, image_name)
                # self.save_lableme(frame, dets[0], path, out_dir=out_dir)
                cv2.imwrite(path, frame)
            count += 1
        video_cap.release()

    def detect_video_dir(self, video_dir, out_dir=None, vis=True):
        # Dataloader
        dataset = file_utils.get_files_lists(video_dir, postfix=['*.mp4', '*.avi'])
        # Run inference
        for video_file in dataset:
            sub = os.path.basename(os.path.dirname(video_file))
            self.start_capture(video_file, out_dir, flag=sub, detect_freq=1, vis=True)

    def detect_image_dir(self, image_dir, out_dir=None, vis=True):
        # Dataloader
        dataset = file_utils.get_files_lists(image_dir)
        # Run inference
        for path in dataset:
            image = cv2.imread(path)  # BGR
            # rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            dets = self.detect(image, vis=False)
            # self.save_crops(image, dets[0], path, out_dir=out_dir, vis=vis)
            self.save_lableme(image, dets[0], path, out_dir=out_dir)
            # if vis:image = self.draw_result([image], dets)[0]

    def save_lableme(self, image, dets, image_file, out_dir=None):
        h, w = image.shape[:2]
        image_name = os.path.basename(image_file)
        image_id = image_name.split(".")[0]
        dirname = out_dir if out_dir else os.path.dirname(image_file)
        json_file = os.path.join(dirname, image_id + ".json")
        boxes = dets[:, 0:4]
        conf = dets[:, 4:5]
        labels = dets[:, 5]
        points = image_utils.boxes2polygons(boxes)
        labels = [self.names[int(label)] for label in labels]
        maker_labelme.maker_labelme(json_file, points, labels, image_name, image_size=[h, w], image_bs64=None)

    def save_crops(self, image, dets, image_file, scale=[], square=False, padding=False, out_dir=None, vis=False):
        image_name = os.path.basename(image_file)
        image_id = image_name.split(".")[0]
        boxes = dets[:, 0:4]
        conf = dets[:, 4:5]
        labels = dets[:, 5]
        if square:
            boxes = image_utils.get_square_bboxes(boxes, use_max=True, baseline=-1)
        if scale:
            boxes = image_utils.extend_xyxy(boxes, scale=scale)
        if padding:
            crops = image_utils.get_bboxes_crop_padding(image, boxes)
        else:
            crops = image_utils.get_bboxes_crop(image, boxes)
        if vis:
            m = image_utils.draw_image_detection_bboxes(image.copy(), boxes, conf, labels, class_name=self.names)
            image_utils.cv_show_image("image", m, use_rgb=False, delay=0)
        for i, img in enumerate(crops):
            name = self.names[int(labels[i])] if self.names else labels[i]
            if out_dir:
                img_file = file_utils.create_dir(out_dir, name, "{}_{:0=3d}.jpg".format(image_id, i))
                cv2.imwrite(img_file, img)
            if vis: image_utils.cv_show_image("crop", img, use_rgb=False, delay=0)


def parse_opt():
    image_dir = '/home/dm/nasdata/Detector/YOLO/yolov5/data/test_image'  # 测试图片的目录
    image_dir = '/home/dm/nasdata/dataset/tmp/fall/videos/dataset2'  # 测试图片的目录
    out_dir = "runs/test-result"  # 保存检测结果
    weights = "/home/dm/nasdata/Detector/YOLO/yolov5/data/model/yolov5s_640/weights/best.pt"  # 模型文件
    # weights = "data/model/yolov5s05_320/weights/best.pt"  # 模型文件
    out_dir = os.path.join(os.path.dirname(image_dir), "crop")
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=weights, help='model.pt')
    parser.add_argument('--image_dir', type=str, default=image_dir, help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--out_dir', type=str, default=out_dir, help='save det result image')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='cuda:0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--class_name', nargs='+', type=list, default=None)
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    d = MakerLabelme(weights=opt.weights,  # model.pt path(s)
                     imgsz=opt.imgsz,  # inference size (pixels)
                     conf_thres=opt.conf_thres,  # confidence threshold
                     iou_thres=opt.iou_thres,  # NMS IOU threshold
                     max_det=opt.max_det,  # maximum detections per image
                     class_name=opt.class_name,  # filter by class: --class 0, or --class 0 2 3
                     classes=opt.classes,  # filter by class: --class 0, or --class 0 2 3
                     agnostic_nms=opt.agnostic_nms,  # class-agnostic NMS
                     augment=opt.augment,  # augmented inference
                     device=opt.device,  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                     )
    # d.detect_image_loader(opt.image_dir)
    # d.detect_image_dir(opt.image_dir, opt.out_dir, vis=True)
    d.detect_video_dir(opt.image_dir, opt.out_dir, vis=True)
