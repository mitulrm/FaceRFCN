from KerasRFCN.Model.Model import RFCN_Model
from KerasRFCN.Config import Config
from KerasRFCN.Utils import Dataset, generate_pyramid_anchors
import os
import numpy as np
#from PIL import Image
import re
import cv2
from KerasRFCN.Data_generator import data_generator, load_image_gt, build_rpn_targets
from IPython import embed
############################################################
#  Config
############################################################

class RFCNNConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "WiderFace"

    # Backbone model
    # choose one from ['resnet50', 'resnet101', 'resnet50_dilated', 'resnet101_dilated']
    BACKBONE = "resnet101"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    C = 1 + 1  # background + 2 tags
    NUM_CLASSES = C
    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    
    RPN_ANCHOR_RATIOS = [0.5, 1, 1.5]
    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)  # anchor side in pixels
    # Use same strides on stage 4-6 if use dilated resnet of DetNet
    # Like BACKBONE_STRIDES = [4, 8, 16, 16, 16]
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]
    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 200

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 1000

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 200

    RPN_NMS_THRESHOLD = 0.6
    POOL_SIZE = 7
    MAX_GT_INSTANCES = 1000
    RPN_TRAIN_ANCHORS_PER_IMAGE = 65412
    
############################################################
#  Dataset
############################################################

class WiderFaceDataset(Dataset):
    def initDB(self, img_dir, annotation_file):
        self.classes = {1:"face"}
        self.add_class("Widerface",1,"Face")
        newImage = False
        regex = re.compile(r".*\.jpg", re.I)
        img = None
        countImg = -1
        imagePath = None
        
        with open(annotation_file, "r") as fp:
            for line in fp:        
                #if countImg == 200:
                #    break
                line = line.strip()
                if regex.search(line):
                    if countImg !=-1:
                    #if self.image_info[-1]['id'] != countImg:
                        img = cv2.imread(imagePath)
                        self.add_image(source="Widerface", image_id = countImg, path=imagePath, width=img.shape[1], height=img.shape[0], bboxes = imageBoxes)
                    newImage = True
                    imagePath = os.path.join(img_dir, line)
                    imageBoxes = []
                    if not os.path.isfile(imagePath):
                        print("{} does not exist".format(line))
                        continue                        
                elif newImage:
                    newImage = False
                    countImg += 1
                else:
                    info = list(map(lambda x: int(x), line.split(" ")))
                    x, y, width, height = info[:4]
                    imageBoxes.append({'x1' : x, 'y1' : y, 'x2' : x + width, 'y2' : y + height, 'class' : 'face'})
                    
    # read image from file and get the 
    def load_image(self, image_id):
        info = self.image_info[image_id]
        img = cv2.imread(info['path'])
        return img

    def get_keys(self, d, value):
        return [k for k,v in d.items() if v == value]

    def load_bbox(self, image_id):
        info = self.image_info[image_id]
        bboxes = []
        class_ids = []
        for item in info['bboxes']:
            bboxes.append((item['y1'], item['x1'], item['y2'], item['x2']))
            class_id = self.get_keys(self.classes, item['class'])
            class_ids.extend(class_id)
        return np.array(bboxes), np.asarray(class_ids)

def show_original_image(dataset, image_id = 0):
    img = dataset.load_image(image_id)      
    bboxes, _ = dataset.load_bbox(image_id)
    print("Image Width : {0}\tImage Hight : {1}".format(img.shape[1], img.shape[0]))
    print("Total Faces : {}".format(len(bboxes)))
    print("Bounding Boxes: {0}".format(bboxes))     
    for bbox in bboxes:
        img = cv2.rectangle(img, (int(bbox[1]), int(bbox[0])), (int(bbox[3]), int(bbox[2])), (255, 0, 0), 1)
    cv2.imshow("", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img

def show_image(dataset, config, image_id = 0):
    img, image_meta, class_ids, bboxes = load_image_gt(dataset, config, image_id)
    print("Image Id: {}".format(image_meta[0]))
    print("Original Image Dimensions: ({0}, {1})".format(image_meta[1], image_meta[2]))
    print("Resized Image Window: ({0},{1}), (({2},{3}))".format(image_meta[3], image_meta[4], image_meta[5], image_meta[6]))
    print("Total Faces : {}".format(len(bboxes)))       
    print("Bounding Boxes: {0}".format(bboxes))
    for bbox in bboxes:
        img = cv2.rectangle(img, (int(bbox[1]), int(bbox[0])), (int(bbox[3]), int(bbox[2])), (255, 0, 0), 1)
    cv2.imshow("", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img
        
def show_anchors(dataset, config, image_id = 0):
    anchors = generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                             config.RPN_ANCHOR_RATIOS,
                                             config.BACKBONE_SHAPES,
                                             config.BACKBONE_STRIDES,
                                             config.RPN_ANCHOR_STRIDE)  
    img, image_meta, gt_class_ids, gt_boxes = load_image_gt(dataset_train, config, image_id, augment=False) 
    rpn_match, rpn_bbox = build_rpn_targets(img.shape, anchors, gt_class_ids, gt_boxes, config)
    positive_anchors = anchors[rpn_match==1]
    negative_anchors = anchors[rpn_match==-1]
    negative_anchors_sample_ids = np.random.choice(negative_anchors.shape[0], size = positive_anchors.shape[0], replace=False)
    negative_anchors = negative_anchors[negative_anchors_sample_ids]
    img = show_image(dataset, config, image_id)

    print("Total Positive Anchors: {}".format(positive_anchors.shape[0]))   
    print("Positive Anchors: {}".format(positive_anchors))
    print("Anchor Deltas: {}".format(rpn_bbox[rpn_match==1]))
    for anchor in positive_anchors:
        img = cv2.rectangle(img, (int(anchor[1]), int(anchor[0])), (int(anchor[3]), int(anchor[2])), (0, 255, 0), 1)
    for anchor in negative_anchors:
        img = cv2.rectangle(img, (int(anchor[1]), int(anchor[0])), (int(anchor[3]), int(anchor[2])), (0, 0, 255), 1)
    cv2.imshow("", img)
    cv2.imshow("", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img

'''
def data_generator(dataset, config, shuffle=True, augment=True, random_rois=0,
                   batch_size=1, detection_targets=False):
    """A generator that returns images and corresponding target class ids,
    bounding box deltas.
    dataset: The Dataset object to pick data from
    config: The model config object
    shuffle: If True, shuffles the samples before every epoch
    augment: If True, applies image augmentation to images (currently only
             horizontal flips are supported)
    random_rois: If > 0 then generate proposals to be used to train the
                 network classifier. Useful if training
                 the Mask RCNN part without the RPN.
    batch_size: How many images to return in each call
    detection_targets: If True, generate detection targets (class IDs, bbox
        deltas). Typically for debugging or visualizations because
        in trainig detection targets are generated by DetectionTargetLayer.
    Returns a Python generator. Upon calling next() on it, the
    generator returns two lists, inputs and outputs. The containtes
    of the lists differs depending on the received arguments:
    inputs list:
    - images: [batch, H, W, C]
    - image_meta: [batch, size of image meta]
    - rpn_match: [batch, N] Integer (1=positive anchor, -1=negative, 0=neutral)
    - rpn_bbox: [batch, N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
    - gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs
    - gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
    outputs list: Usually empty in regular training. But if detection_targets
        is True then the outputs list contains target class_ids, bbox deltas.
    """
    b = 0  # batch item index
    image_index = -1
    image_ids = np.copy(dataset.image_ids)
    error_count = 0

    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    anchors = KerasRFCN.Utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                             config.RPN_ANCHOR_RATIOS,
                                             config.BACKBONE_SHAPES,
                                             config.BACKBONE_STRIDES,
                                             config.RPN_ANCHOR_STRIDE)

    # Keras requires a generator to run indefinately.
    while True:
        try:
            # Increment index to pick next image. Shuffle if at the start of an epoch.
            image_index = (image_index + 1) % len(image_ids)
            if shuffle and image_index == 0:
                np.random.shuffle(image_ids)

            # Get GT bounding boxes for image.
            image_id = image_ids[image_index]
            image, image_meta, gt_class_ids, gt_boxes = load_image_gt(dataset, config, image_id, augment=augment)

            # Skip images that have no instances. This can happen in cases
            # where we train on a subset of classes and the image doesn't
            # have any of the classes we care about.
            if not np.any(gt_class_ids > 0):
                continue

            # RPN Targets
            rpn_match, rpn_bbox = build_rpn_targets(image.shape, anchors,
                                                    gt_class_ids, gt_boxes, config)
'''

if __name__ == '__main__':
    config = RFCNNConfig()
    dataset_train = WiderFaceDataset()
    dataset_train.initDB(img_dir = 'C:/Data/Widerface_Dataset/train/images', annotation_file = 'C:/Data/Widerface_Dataset/wider_face_split/wider_face_train_bbx_gt.txt')
    dataset_train.prepare()
    anchors = generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                             config.RPN_ANCHOR_RATIOS,
                                             config.BACKBONE_SHAPES,
                                             config.BACKBONE_STRIDES,
                                             config.RPN_ANCHOR_STRIDE)  
    image, image_meta, gt_class_ids, gt_boxes = load_image_gt(dataset_train, config, 10, augment=False)
    rpn_match, rpn_bbox = build_rpn_targets(image.shape, anchors, gt_class_ids, gt_boxes, config)
    #generator_train = data_generator(dataset_train, config, shuffle = False, batch_size = config.BATCH_SIZE, detection_targets = True)
    #Validation dataset
    #dataset_val = FashionDataset()
    #dataset_val.initDB(img_dir = 'C:\Data\Widerface_Dataset\train\images')
    #dataset_val.prepare()
    embed()