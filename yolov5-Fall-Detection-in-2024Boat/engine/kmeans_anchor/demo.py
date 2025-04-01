# -*-coding: utf-8 -*-
"""
    @Author : Pan
    @E-mail : 390737991@qq.com
    @Date   : 2022-06-09 10:31:41
    @Brief  :
"""
import numpy as np
import utils.autoanchor as autoAC

coco_anchor = [[10, 13, 16, 30, 33, 23],
               [30, 61, 62, 45, 59, 119],
               [116, 90, 156, 198, 373, 326]]


def get_size_anchor(anchor=coco_anchor, img_size=416):
    """"""
    size640 = 640
    # anchor = [[63, 26, 101, 37, 122, 43],
    #           [95, 66, 137, 53, 174, 58],
    #           [136, 80, 149, 120, 215, 88]]
    scale = img_size / size640
    anchor = np.asarray(anchor) * scale
    return anchor, img_size


def print_format_anchors(anchors, img_size=640):
    """格式化输出anchors"""
    print(anchors.tolist())
    anchors = np.asarray(np.round(anchors), dtype=np.int32)
    print("------------------python config {}------------------".format(str(img_size)))
    anchors = anchors.reshape(3, -1)
    for anchor in anchors:
        print(anchor.tolist())
    print("------------------C++    config {}------------------".format(str(img_size)))
    anchors = anchors.reshape(3, -1, 2).tolist()
    for i in range(len(anchors)):
        result = "{}".format(anchors[2 - i]).replace("[", "{").replace("]", "}")
        print(result)
    print("-------------------------------------------------")


if __name__ == "__main__":
    """
    YOLOv5s-640原始Anchor(通用):
      - [10,13, 16,30, 33,23]  # P3/8
      - [30,61, 62,45, 59,119]  # P4/16
      - [116,90, 156,198, 373,326]  # P5/32
    """
    # (1)重新计算Anchor
    data = '/home/dm/nasdata/Detector/YOLO/yolov5/engine/configs/voc_local.yaml'
    anchors = autoAC.kmean_anchors(data, n=9, img_size=640, thr=4.0, gen=1000, verbose=False)
    print_format_anchors(*get_size_anchor(anchors, img_size=640))
    print_format_anchors(*get_size_anchor(anchors, img_size=416))
    print_format_anchors(*get_size_anchor(anchors, img_size=320))

    # (2)使用原始的Anchor
    print_format_anchors(*get_size_anchor(img_size=640))
    print_format_anchors(*get_size_anchor(img_size=416))
    print_format_anchors(*get_size_anchor(img_size=320))
