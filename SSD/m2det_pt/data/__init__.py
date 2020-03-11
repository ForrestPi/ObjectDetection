# from .voc import VOCDetection, AnnotationTransform, detection_collate, VOC_CLASSES
from .voc0712 import VOCDetection, AnnotationTransform, detection_collate, VOC_CLASSES
from .coco import COCODetection
from .custom_voc_like_helmet import CustomHelmetDataset
from .data_augment import *
from .anchors import *
