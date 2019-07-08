import h5py  
import numpy as np
import pandas as pd

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

from sklearn.metrics import classification_report

#keras.applications.resnet50

#copy files
import shutil
import os, sys;
from PIL import Image

import seaborn as sns
import matplotlib.pyplot as plt



#write as h5File   
def write_label_h5(df):
    preds=df.iloc[:,1]
   
    file_name='data/Predicted_labels/Predicted_Label_all' + '.h5'
    hf = h5py.File(file_name, 'w')
    hf.create_dataset('label', data=preds)
    hf.close()
    
    
def Get_Class_Number(df):
    preds = np.zeros(len(df))
    i=0
    for row in df:
        
        preds[i]=int(row[12:])
    
        i +=1
                
    return preds

def get_label(df):
    labels=[]
    for i,j in df.iterrows():
   
        if ',' in j[1]:
            s= j[1].split(',')

            lbs=[int(x) for x in s]
            #labels.append(np.asarray(lbs))
            labels.append(j[1])

        else:

            labels.append(j[1])
            
    return np.asarray(labels)

def Intialize_model():
    
        epochs = 12
        input_shape = (224, 224, 3)
        batch_size = 64
        seed = 42
        #dataset_path = '/Data/Resized' # same train/test dataset to overfit it
        #dataset_path = data_root
        
        np.random.seed(seed)
        tf.set_random_seed(seed)
        
        
        #K.set_learning_phase(1)
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
        
        
        ###test to avoid overfitting
        
        for layer in base_model.layers:
            layer.trainable=False
        x = base_model.output
        
        x = GlobalAveragePooling2D()(x)
        
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        #x = BatchNormalization()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        #x = BatchNormalization()(x)
        predictions = Dense(20, activation="softmax")(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        
        
        #optimizer = Adam(lr=0.001)#'rmsprop'
        SGD_optimizer=SGD(lr=0.0125, momentum=0.9)
        #adam_optimizer = Adam(lr=0.0001)
        #RMSprop_optimizer = RMSprop(lr=0.0001)
        model.compile(optimizer=SGD_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
        return model
    
def get_single_label_df(df):
    #data=pd.DataFrame(columns=(0,1))
    lst=[]
    for i,j in df.iterrows():
        dct={}

        if ',' in j[1]:
            s= j[1].split(',')

            dct[0]=j[0]
            dct[1]=s[0]
            #lst.append(dct)

        else:

            dct[0]=j[0]
            dct[1]=j[1]
            lst.append(dct)

    
    df_label=pd.DataFrame(lst)
    
    return df_label

def copy_train_test_val_images(df_all,type='train', start=3100, end=20100):
    df_single= get_single_label_df(df_all)
    #df=df_single[start:end]#df.copy()

    it=0
    image_dir='../Input/train2014/'
    
    if type=='train':
        df=df_single[start:end]
        copy_dir='../Input/train/'
    elif type=='val':
        df=df_single[end:]
        copy_dir='../Input/val/'
    elif type=='test':
        df=df_all[:4000]
        copy_dir='../Input/test/test_images/'
    
    if type=='test':
        for i,j in df.iterrows():
            fr=image_dir + str(j[0])
            to=copy_dir + str(j[0])
            shutil.copy(fr, to)
        
    else:
        
        for i,j in df.iterrows():
             
            dct={}
    
            if ',' in j[1]:
                s= j[1].split(',')
    
                dct[0]=j[0]
                dct[1]=s[0]
                folder_name=s[0]
    
            else:
    
                dct[0]=j[0]
                dct[1]=j[1]
                folder_name=j[1]
            
            #class_label_
            fr=image_dir + str(j[0])
            
            it +=1
            #if it>=6000:
                      
            to=copy_dir + 'class_label_'  + folder_name + '/' + str(j[0])
            shutil.copy(fr, to)
        
        
def create_train_test_val_folders():
    os.mkdir('../Input/train')
    os.mkdir('../Input/val')
    os.mkdir('../Input/test')
    os.mkdir('../Input/test/test_images')
    
    str_cls='class_label_'
    for i in range(20):
        sub_folder=str_cls + str(i)
        os.mkdir('../Input/train/' + sub_folder)
        os.mkdir('../Input/val/' + sub_folder)
        
    
def write_labels_txt_file(df):
    
  
    #open file_chat in write mode and write in it using content of lst_chat
    with open('../Output/Predicted_labels.txt', 'w') as myfile:
        line=''
        for i, row in df.iterrows():
            line =str(int(row[0])) + '.jpg' + '\t' + str(int(row[1]))
            myfile.write(line + '\n')

        myfile.close()



def Predict_on_Validation(model,labels):
    
        df=pd.read_csv('../Input/train.txt', header=None, sep='\t')
        
        df2=df[:4000]
        
        Predicted_labels=preds_all_images(model, labels,total_iamges=4000, batch_size=1000, is_valid=True)
        target=get_label(df2)
        true_label=[]
        count=0
        for i in range(4000):
            lables=target[i].split(',')
            
            
            if(str(int(Predicted_labels[i])) in lables):
                true_label.append(int(Predicted_labels[i]))
                count +=1
            else:
                true_label.append(int(lables[0]))
                
        accuracy= count/len(Predicted_labels)
        print('Correct:', count, 'Accuracy:',count/len(Predicted_labels))
        
        return np.asarray(true_label), Predicted_labels, count, accuracy
    
def preds_all_images(model, labels,total_iamges=15516, batch_size=1000, is_valid=False):
    total_batches=total_iamges//1000
    odd=total_iamges % 1000
    all_preds=np.zeros(0)
    
    start_idx=0
    for i in range(total_batches):
        #batch=np.ones(1000)
        batch_label=get_predicted_labels(model, labels=labels, start_index= start_idx,batch_size=1000, is_valid=is_valid)
        
        if( len(all_preds)==0):
            all_preds=batch_label
        else:
            tmp=np.concatenate((all_preds,batch_label),axis=0)
            all_preds=tmp
            
        start_idx +=batch_size
    
    if odd>0:             
        batch_label=get_predicted_labels(model, labels=labels, start_index= start_idx,batch_size=odd, is_valid=is_valid)
    
        if( len(all_preds)>0):
            tmp=np.concatenate((all_preds,batch_label),axis=0)
            all_preds=tmp
        
    return all_preds

def get_predicted_labels(model, labels,start_index=0,batch_size=1000,is_valid=False):
    test_img_paths=[]
    if (is_valid== False):
        input__test_path='../Input/val2014/'
    else:
        #input__test_path='data/val2014/'
        input__test_path='../Input/test/test_images/'
        
    for i in range(batch_size):
        test_img_paths.append(str(i + start_index)+'.jpg')

    img_list_test = [Image.open(input__test_path + img_path) for img_path in test_img_paths]
    
    final_test_batch = np.stack([preprocess_input(np.array(img.resize((224,224))))
                             for img in img_list_test])
    pred_probs = model.predict(final_test_batch)
    
    Predicted_label_index=np.argmax(pred_probs, axis=1) #y_pred#
    predictions_class = [labels[k] for k in Predicted_label_index]
    Predicted_labels=Get_Class_Number(predictions_class)
    
    return Predicted_labels

#calculate and show confustion matrix
def get_confusion_matrix(prediction, actual):
    conf_matrix = pd.DataFrame(list(zip(prediction,actual)), 
                                columns=['predicted labels','actual labels'])
    conf_matrix['const'] = 1
    conf_matrix = pd.pivot_table(data=conf_matrix, 
                               index='actual labels', 
                               columns='predicted labels', 
                               values='const', 
                               aggfunc=sum)
    conf_matrix = conf_matrix.fillna(0)
    return conf_matrix

#Draw heat map from confusion matrix
def heatmap_confusion_matrix(conf_matrix):
    plt.figure(figsize=(8,6))
    g = sns.heatmap(conf_matrix, cbar_kws={'label':'Frequency'}, center=0, cmap=sns.diverging_palette(10, 5, as_cmap=True)).set_title('Confusion Matrix')
    return g
    

def f1_score(conf_matrix):
  
    lst_FP=[]
    lst_FN=[]
    lst_TP=[]
    lst_Pres=[]
    lst_Recall=[]
    lst_f1_score=[]
    for k in range(len(conf_matrix)):
      FP=sum([conf_matrix[k][i] for i in range(len(conf_matrix))]) - conf_matrix[k][k]
      lst_FP.append(FP)
      lst_TP.append(conf_matrix[k][k])
    
    for k in range(len(conf_matrix)):
      FN=sum([conf_matrix[i][k] for i in range(len(conf_matrix))]) - conf_matrix[k][k]
      lst_FN.append(FN)
      
    for i in range(len(conf_matrix)):
      pres=lst_TP[i]/(lst_TP[i]+lst_FP[i])
      recall=lst_TP[i]/(lst_TP[i]+lst_FN[i])
      
      lst_Pres.append(pres)
      lst_Recall.append(recall)
      
    for i in range(len(conf_matrix)):
      f1_score=((lst_Pres[i]*lst_Recall[i])/(lst_Pres[i] + lst_Recall[i]))*2
      lst_f1_score.append(f1_score)
      
    return lst_Pres, lst_Recall,lst_f1_score

def precision_recall(true_label, pred_label):
  
    y_true = true_label
    y_pred = pred_label
    #unique_relations=np.unique(tr)

    #target_names = unique_relations
    precision_recall=classification_report(y_true, y_pred)
    
    return precision_recall
    
    
    
    
    