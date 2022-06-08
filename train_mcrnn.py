import tensorflow as tf
import tensorflow_addons as tfa
import os
import re
import pandas as pd
import os
import cv2

from PIL import Image
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib import patches
from tensorflow.python.ops.numpy_ops import np_config
#from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from scipy.ndimage.measurements import label
from keras_unet_collection import models
from itertools import chain



from src.data_prepare import *
from src.utils import *
from src.modeling.models import *
from src.modeling.model_utils import *
from src.modeling.iou_loss import *

#K.set_image_dim_ordering('th')
np_config.enable_numpy_behavior()
from src.mrcnn.config import Config
from src.mrcnn import model as modellib, utils
from src.mrcnn import visualize
from src.mrcnn.model import log


MODEL_DIR = "logs"


class GerConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "ger"
 
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
 
    # Number of classes (including background)
    NUM_CLASSES = 2  # background + 3 shapes
 
    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024

    DETECTION_MIN_CONFIDENCE = 0.95
    DETECTION_NMS_THRESHOLD = 0.0
 
    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)  # anchor side in pixels
 
    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    DETECTION_MAX_INSTANCES = 20
 
    # Use a small epoch since the data is simple

    BATCH_SIZE = 8
    STEPS_PER_EPOCH = 6000 // BATCH_SIZE
 
    # use small validation steps since the epoch is small
    VALIDATION_STEPS = len(os.listdir('kaggle/val/images/')) // BATCH_SIZE
 
 
config = GerConfig()
config.display()

# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)




class GerDataset(utils.Dataset):

    def load_dataset(self, dataset_dir, is_train=True):
        # Adds information (image ID, image path, and annotation file path) about each image in a dictionary.
        self.add_class("dataset", 1, "ger")

        images_dir = dataset_dir + '/images/'
        annotations_dir = dataset_dir + '/labels/'

        for filename in os.listdir(images_dir):
            
            if len(re.findall('sca',filename)) == 1 or len(re.findall('tra',filename)) == 1 or  len(re.findall('she',filename)) == 1:
                
                continue
            else:
            
                image_id = filename[:-4]
                

                img_path = images_dir + filename
                ann_path = annotations_dir + image_id + '.txt'

            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

    # Loads the binary masks for an image.
    def load_mask(self, image_id):
        info = self.image_info[image_id]
        path = info['annotation']
        boxes, w, h = self.extract_boxes(path)
        masks = np.zeros([h, w, len(boxes)], dtype='uint8')

        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[0], box[2]
            col_s, col_e = box[1], box[3]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index('ger'))
        return masks, np.asarray(class_ids, dtype='int32')

    # A helper method to extract the bounding boxes from the annotation file
    def extract_boxes(self, filename):

        tmp_df = pd.read_csv(filename,header=None)
        boxes = list()
        for j in tmp_df[0].values:
            xmin = int(j.split()[0])
            ymin = int(j.split()[2])
            xmax = int(j.split()[1])
            ymax = int(j.split()[3])
            #(y1, x1, y2, x2)
            coors = [ymin, xmin, ymax, xmax]
            boxes.append(coors)

        width = 1024
        height = 1024
        return boxes, width, height


train_dataset = GerDataset()
train_dataset.load_dataset(dataset_dir='kaggle/train')
train_dataset.prepare()


val_dataset = GerDataset()
val_dataset.load_dataset(dataset_dir='kaggle/val')
val_dataset.prepare()

model.train(train_dataset=train_dataset, 
            val_dataset=val_dataset, 
            learning_rate=1e-4, 
            epochs=50, 
            layers='heads')