import tensorflow as tf
import pandas as pd
import os
import tensorflow_addons as tfa
import cv2
import glob
import torch
import tensorflow_addons as tfa

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


######## Data Error ########

annot_df = {'img_name' : [],'classes' : [],'x' : [],'y' : [],'width' : [],'height' : []}
sample = {'image' : [],'bbox' : [],'label' : []}
empty_list = []
for idx,i in tqdm(enumerate(os.listdir('kaggle/train/labels/'))):
    ## Empty Labels
    try:
        tmp_df = pd.read_csv(f'kaggle/train/labels/{i}',header=None)
        for j in tmp_df[0].values:
            annot_df['img_name'].append(f'kaggle/train/images/{i.split(".")[0]}.png')
            #sample['image'] = Image.open(f'kaggle/train/images/{i.split(".")[0]}.png').convert('RGB')
            annot_df['classes'].append(j.split()[0])
            annot_df['x'].append(j.split()[1])
            annot_df['y'].append(j.split()[2])
            annot_df['width'].append(j.split()[3])
            annot_df['height'].append(j.split()[4])
            ###
            sample['label'].append(j.split()[0])
            sample['bbox'].append(j.split()[1])
            sample['bbox'].append(j.split()[2])
            sample['bbox'].append(j.split()[3])
            sample['bbox'].append(j.split()[4])
    except pd.errors.EmptyDataError:
        empty_list.append(i)
        continue


df = pd.DataFrame.from_dict(annot_df)

### Take Train and Val Df

def convert_bbox(x):
    img = Image.open(x['img_name'])
    #sizes = img.size
    # Load xy, width and height
    x_mod = (float(x['x']) - float(x['width']) * 0.5) * img.size[0]
    y_mod = (float(x['y']) - float(x['height']) * 0.5) * img.size[1]
    width_mod = float(x['width']) * img.size[0]
    height_mod = float(x['height']) * img.size[1]
    return pd.Series([x_mod,y_mod,width_mod,height_mod],index=['x_mod','y_mod','width_mod','height_mod'])


df[['x_mod','y_mod','width_mod','height_mod']] = df.apply(lambda x: convert_bbox(x),axis=1)
df[['x_mod','y_mod','width_mod','height_mod']] = df[['x_mod','y_mod','width_mod','height_mod']].astype(int)


df['xmin'] = df['x_mod']
df['xmax'] = df['xmin'] + df['width_mod']
df['ymin'] = df['y_mod']
df['ymax'] = df['ymin'] + df['height_mod']


model_unet = UNet((1024,1024,3))
#model_effunet = UXception(None,(512,512,3))
#model_swinunet = models.swin_unet_2d((1024, 1024, 3), filter_num_begin=64, n_labels=1, depth=4, stack_num_down=2, stack_num_up=2, 
#                            patch_size=(2, 2), num_heads=[4, 8, 8, 8], window_size=[4, 2, 2, 2], num_mlp=512, 
#                            output_activation='Sigmoid', shift_window=True, name='swin_unet')

# model
#input_layer = Input((1024, 1024, 3))
#output_layer = build_model(input_layer, 16,0.5)

#model_ures = Model(input_layer, output_layer)

batch_size = 8

# Train test Split
val_images = np.random.choice(df['img_name'].unique(),size=600)
val_df = df[df['img_name'].isin(val_images)]
train_df = df[~df['img_name'].isin(val_images)]
# Resetting Indexes
val_df = val_df.reset_index(drop=True)
train_df = train_df.reset_index(drop=True)

### Generator 

train_samples = []

for img in train_df['img_name'].unique():
    bboxes = train_df.query(f"img_name == '{img}'")
    bboxes['classes'] = bboxes['classes'].astype(float)
    bboxes = bboxes[['xmin','ymin','xmax','ymax','classes']]
    train_samples.append([img,
                        bboxes.values
                         ]
                        )

gen_main = generator_main(train_samples,batch_size)
gen_hsv = generator_hsv(train_samples,batch_size)
#gen_flip = generator_flip(train_samples,batch_size)
#gen_scale = generator_scale(train_samples,batch_size)
gen_trans = generator_translate(train_samples,batch_size)
gen_shear = generator_shear(train_samples,batch_size)


train_generator = chain(gen_main,gen_hsv)
#train_generator1 = chain(gen_scale,gen_flip)
train_generator2 = chain(gen_trans,gen_shear)
main_generator = chain(train_generator,train_generator2)
#main_generator_last = chain(main_generator,train_generator2)



val_samples = []

for img in val_df['img_name'].unique():
    bboxes = val_df.query(f"img_name == '{img}'")
    bboxes['classes'] = bboxes['classes'].astype(float)
    bboxes = bboxes[['xmin','ymin','xmax','ymax','classes']]
    val_samples.append([img,
                        bboxes.values
                        ])

val_gen = generator_main(val_samples,batch_size)

smooth = 1.
model_unet.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), 
              loss=bce_logdice_loss, metrics=[my_iou_metric])


steps_per_epoch = int(4 *len(train_df) / batch_size)
val_per_epoch = int(len(val_df) / batch_size)
custom_early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=int(3),
    min_delta=0.01,
    mode='min',
    restore_best_weights=True
)

history = model_unet.fit(main_generator,
                        validation_data=val_gen,
                        validation_steps=val_per_epoch,
                            steps_per_epoch=steps_per_epoch,
                            epochs=30,
                            callbacks=[custom_early_stopping]
                            )


model_unet.save('object_unet_0606_aug.h5')

K.clear_session()

#model_unet = keras.models.load_model('object_unet_0606.h5',
#                                        compile=False
#                                        )
## remove layter activation layer and use losvasz loss
#input_x = model_unet.layers[0].input
#
#output_layer = model_unet.layers[-1].input
#model = Model(input_x, output_layer)
#c = tf.keras.optimizers.Adam(learning_rate = 0.01)
#
## lovasz_loss need input range (-∞，+∞), so cancel the last "sigmoid" activation  
## Then the default threshod for pixel prediction is 0 instead of 0.5, as in my_iou_metric_2.
#model.compile(loss=lovasz_loss, optimizer=c, metrics=[my_iou_metric_2])
#
#history2 = model.fit(train_generator,
#                    validation_data=val_gen,
#                    validation_steps=val_per_epoch,
#                    steps_per_epoch=steps_per_epoch,
#                    callbacks=[custom_early_stopping],
#                    epochs=30
#                    )
#
#model.save('unet_0606.h5')