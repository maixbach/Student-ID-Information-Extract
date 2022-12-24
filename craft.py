# pip install yolov5==6.2.3
import yolov5
import cv2
import numpy as np
from PIL import Image as im
import config as cf


def craft(img):
    """Craft: Image of Student Card into the boxes of information needed to extract
    Input: Image of Student card (.jpg, .png, ...)
    Output: The boxes of information (numpy array) """

    # load custom model
    model = yolov5.load('./weights/best.pt')

    # set model parameters
    model.conf = cf.conf  # NMS confidence threshold
    model.iou = cf.iou  # NMS IoU threshold
    model.agnostic = False  # NMS class-agnostic
    model.multi_label = False  # NMS multiple labels per box
    model.max_det = cf.max_detection  # maximum number of detections per image

    # inference with test time augmentation
    results = model(img, augment=True)

    # parse results
    predictions = results.pred[0]
    boxes = predictions[:, :4]  # x1, y1, x2, y2
    scores = predictions[:, 4]
    categories = predictions[:, 5]

    # crop each box
    crops = results.crop(save=False)

    # create a dictionary to save all the box and its label
    crops_dict = {}
    for i in range(len(crops)):
        label_and_weight = crops[i]["label"].split()
        label, weight = label_and_weight[0], label_and_weight[1]
        crops_dict.update({label: [weight, crops[i]["im"]]})

    return crops_dict