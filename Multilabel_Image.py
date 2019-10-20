'''
# ## Readme
# ###How to Run:
# From Spyder
#### 1. Please open code/Alogithm/Comp_5329_Assig2_Program.py 
# #### 2. Please goto  Menu -> Run ->  or from keyboard please press (ctrl + F5)

# From Anaconda Prompt
###     1.Please change directory to Code/Alogorithm then
# #### 2. Please type python Comp_5329_Assig2_Program.py

Please keep train images under Input -> train2014 and Test Images(15516 here) under Input -> val2014 folder before run code
'''


from Custom_Functions import create_train_test_val_folders, write_label_h5, Get_Class_Number, get_label, Intialize_model, get_single_label_df, copy_train_test_val_images

from Custom_Functions import precision_recall, get_predicted_labels, preds_all_images, Predict_on_Validation, get_confusion_matrix, heatmap_confusion_matrix, f1_score                      

import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Flatten,GlobalAveragePooling2D,Dropout, BatchNormalization
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import array_to_img


from tensorflow.keras.optimizers import Adam, RMSprop, Nadam, SGD
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications.resnet50 import preprocess_input
#keras.applications.resnet50



import pandas as pd
from PIL import Image


## The commented code below will create 3 folders 'train' , 'test' and 'val' and in train and val it will create 20 sub-folders folders 
## and named like class_label_0,... and copy sigle labelled images from train2014 folder for train and val. use both single and
## multilabel images for predicting test and val accyracy.

## Start---Create folders. please comment after 1st Run

df=pd.read_csv('../Input/train.txt', header=None, sep='\t')
print('Total labels:',len(df))

create_train_test_val_folders()

copy_train_test_val_images(df,type='train')
copy_train_test_val_images(df,type='val')
copy_train_test_val_images(df,type='test')

## End---Create folders. please comment after 1st Run


MODEL_TYPE='load' # load or train

MODEL_RUN_ON='validation' # 'validation' (data taken from train data) or 'test' (No label provided)

epochs = 25
input_shape = (224, 224, 3)
batch_size = 64
seed = 42
n_label=20
#dataset_path = '/Data/Resized' # same train/test dataset to overfit it
#dataset_path = data_root

np.random.seed(seed)
tf.set_random_seed(seed)

#Inilizize Resnet_50 Model
model=Intialize_model()

###
train_gen = ImageDataGenerator( preprocessing_function=preprocess_input)



val_gen = ImageDataGenerator( preprocessing_function=preprocess_input) #rescale=1. / 255

test_gen = ImageDataGenerator( preprocessing_function=preprocess_input)

test_gen = image.ImageDataGenerator().flow_from_directory('../Input/test', target_size=input_shape[:2], 
                                                batch_size=1, class_mode='categorical', shuffle=False)

train_gen = image.ImageDataGenerator().flow_from_directory('../Input/train', target_size=input_shape[:2], 
                                                batch_size=batch_size, class_mode='categorical', shuffle=True, seed=seed)

val_gen = image.ImageDataGenerator().flow_from_directory('../Input/val', target_size=input_shape[:2], 
                                                         batch_size=batch_size, class_mode='categorical', shuffle=True)

train_steps = train_gen.samples//batch_size
val_steps = val_gen.samples//batch_size


reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, mode='auto')


#Load our pretrend model weights
if MODEL_TYPE=='load':
    model = load_model('../Input/model/Resnet_single_preprocess_v1.h5')
else:
    #Train last few layers of Resnet_50 Pretrend model with Imagnet weights
    model.fit_generator(train_gen, train_steps, epochs=epochs, validation_data=val_gen, validation_steps=val_steps)

    val_gen.reset()
    print('Before Save:', model.evaluate_generator(val_gen, val_steps)) 
    model.save('../Output/Resnet_model.h5')

#Summary of model after we changed last few layers
model.summary()

#Actual labels and indices
labels = (train_gen.class_indices)
labels = dict((v,k) for k,v in labels.items())


#model = load_model('../Input/model/Resnet_single_preprocess_v1.h5')

#Run on validation data, so here we can check how our model should perform in unknown data
if MODEL_RUN_ON=='validation':
    true_labels, pred_labels, c, acc=Predict_on_Validation(model,labels)
    print('Accuracy on Validation data:', acc)
    conf_matrix=get_confusion_matrix(pred_labels, true_labels)
    print( conf_matrix)
        
    heat_map=heatmap_confusion_matrix(conf_matrix)
    print(heat_map)
    
    presicion_recall=precision_recall(true_labels, pred_labels)
    print(presicion_recall)
    
    
    
else:
    #predict unlabelled test images provided
    preds_all=preds_all_images(model,labels, total_iamges=15516, batch_size=1000)

