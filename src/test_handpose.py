import numpy as np
import os
import pandas as pd
import csv
import cv2
import pickle
import cPickle
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, History
from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dropout
from tensorflow.keras.layers import Activation, BatchNormalization, Add, Reshape, DepthwiseConv2D
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm


DATA_PATH='/data/multimedia/Dataset/multiview_hand_pose_dataset_release/'
RESIZED_IMG_SIZE=128
ORG_TEST_IMG_DIR='/data/multimedia/yogesh/2020/hand_pose_estimation/org_test_img/'
OUT_TEST_IMG_DIR='/data/multimedia/yogesh/2020/hand_pose_estimation/out_test_img/'
err_count = 0
total_count = 0


def define_model4(img_size, num_labels, epochs):
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=3, strides=(2,2), activation='relu', input_shape=(RESIZED_IMG_SIZE, RESIZED_IMG_SIZE, 3)))
    model.add(Conv2D(filters=16, kernel_size=1, activation='relu'))
    model.add(Conv2D(filters=16, kernel_size=3, strides=(2,2), activation='relu'))
    model.add(Conv2D(filters=16, kernel_size=1, activation='relu'))
    
    model.add(Conv2D(filters=32, kernel_size=3, strides=(2,2), activation='relu'))
    #model.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
    #model.add(Conv2D(filters=32, kernel_size=3, strides=(2,2), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3,  strides=(2,2), activation='relu'))
    model.add(Conv2D(filters=128, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    #model.add(Dropout(0.2))

    model.add(Dense(42))
    model.summary()
    #model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001), metrics=['accuracy'])
    return model

def define_model3(img_size, num_labels, epochs):
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=3, strides=(2,2), activation='relu', input_shape=(RESIZED_IMG_SIZE, RESIZED_IMG_SIZE, 3)))
    
    model.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
    model.add(Conv2D(filters=32, kernel_size=3, strides=(2,2), activation='relu'))
    model.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
    model.add(Conv2D(filters=32, kernel_size=3, strides=(2,2), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3,  strides=(2,2), activation='relu'))
    model.add(Conv2D(filters=128, kernel_size=3, activation='relu'))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(42))
    model.summary()
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model

def define_model2(img_size, num_labels, epochs):
    model = Sequential()
    model.add(Conv2D(filters=512, kernel_size=3, activation='relu', input_shape=(128, 128, 3)))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters=256, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters=128, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters=128, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters=128, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(42))
    model.summary()
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model


def calc_test_error(pred, gt):
    predictions = pred.reshape(21, 2)
    ground_truth = gt.reshape(21, 2)
    threshold =  float(20.0)/128.0
    for (xp,yp), (xg, yg) in zip(predictions, ground_truth):
        global err_count
        global total_count
        diff = np.sqrt((xp-xg)**2 + (yp-yg)**2)
        print ('diff = '+str(diff))
        if diff > threshold:
            err_count += 1
        total_count += 1
    return


def plot_landmarks(img, points, file_name, color):
    for (x,y) in points:
         #print(x, y)
         cv2.circle(img, (x,y), 3, color, -1)
    cv2.imwrite(file_name, img)

    
def test_model(test_img, model, count, landmarks_org):
   
    #print(landmarks)
    org_img = test_img * 255
    landmarks = (landmarks_org * 129).astype(np.int)
    #print('###############')
    #print(landmarks)
    landmarks = landmarks.reshape(21, 2)
    
    points_test_pred = model.predict(test_img.reshape(1, RESIZED_IMG_SIZE, RESIZED_IMG_SIZE, 3))
    points_test = (points_test_pred * 128).astype(np.int)
    points_test = points_test.reshape(21,2)
    #print(points_test)
    plot_landmarks(org_img, landmarks, ORG_TEST_IMG_DIR+'org_test_img_'+str(count)+'.jpg', (255, 0, 0)) 

    plot_landmarks(org_img, points_test, OUT_TEST_IMG_DIR+'out_test_img'+str(count)+'.jpg', (0, 255, 0))

    calc_test_error(points_test_pred, landmarks_org)

def main():
    hand_key_points = 21
    num_labels = hand_key_points * 2
    epoch = 500

    #checkpoint = ModelCheckpoint('./trained_model/best_weights_model4.h5', verbose=1, save_best_only=True)
    #hist = History()
    model = define_model4(RESIZED_IMG_SIZE, num_labels, epoch)
    model.load_weights('./trained_model/best_weights_model4.h5')

    if os.path.exists("test_data.npz"):
        data = np.load("test_data.npz")
        img_test = data['img']
        out_test = data['landmarks']
      
        print(img_test.shape)
        print(out_test.shape)

    else:
        test_img, test_gt = load_dataset("/data/multimedia/Dataset/multiview_hand_pose_dataset_release/data3.csv")
        np.savez('test_data.npz', img=test_img, landmarks=test_gt)

    count = 0
    for im, gt in zip(img_test, out_test):
        #print(im)
        #print(gt)
        test_model(im, model, count, gt)

        count += 1

    print ('err_count = %', err_count) 
    print ('total_count = %', total_count) 
    print('percentage point error = ', err_count*100/total_count, '%')
        #jvlksv




if __name__ == "__main__":
    main ()
    #test()

