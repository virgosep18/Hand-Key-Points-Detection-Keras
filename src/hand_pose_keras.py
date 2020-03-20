import numpy as np
import os
import csv
import cv2
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
from tensorflow.keras.layers import Activation, BatchNormalization, Add, Reshape, DepthwiseConv2D
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm



DATA_PATH='/data/multimedia/Dataset/multiview_hand_pose_dataset_release/'
RESIZED_IMG_SIZE=128

def custom_loss(y_true, y_pred):
        return K.mean(y_true - y_pred)**2

def define_model4_org(img_size, num_labels, epochs):
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
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001), metrics=['mae'])
    return model


def define_model4(img_size, num_labels, epochs):
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=3, strides=(2,2), activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros', input_shape=(RESIZED_IMG_SIZE, RESIZED_IMG_SIZE, 3)))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=16, kernel_size=1, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros' ))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Conv2D(filters=16, kernel_size=3, strides=(2,2), kernel_initializer='random_uniform', bias_initializer='zeros', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=16, kernel_size=1, kernel_initializer='random_uniform', bias_initializer='zeros', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    
    model.add(Conv2D(filters=32, kernel_size=3, strides=(2,2), kernel_initializer='random_uniform', bias_initializer='zeros', activation='relu'))
    model.add(BatchNormalization())
    #model.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
    #model.add(Conv2D(filters=32, kernel_size=3, strides=(2,2), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3, kernel_initializer='random_uniform', bias_initializer='zeros', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Conv2D(filters=64, kernel_size=3,  strides=(2,2), kernel_initializer='random_uniform', bias_initializer='zeros', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Conv2D(filters=128, kernel_size=3, kernel_initializer='random_uniform', bias_initializer='zeros', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))

    model.add(Flatten())
    model.add(Dense(512, kernel_initializer='random_uniform', bias_initializer='zeros', activation='relu'))
    model.add(Dropout(0.1))

    model.add(Dense(42))
    model.summary()
    #model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001), metrics=['mae'])
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
    for (xp,yp), (xg, yg) in zip(predictions, ground_truth):
        print(xp, yp)
        print("#########")
        pritn(xg, yg)
    fjk
    return 

def plot_landmrks(img, points, file_name):
    for (x,y) in points:
         cv2.circle(img, (x,y), 3, (0, 255, 255), -1)
    cv2.imwrite(file_name, img)


def crop(image, xmin, ymin, xmax, ymax):
    return image[ymin:ymax , xmin:xmax, :]


# does data augmentation by flipping the image
def flip_img(img, points):
    rows, cols = img.shape[:2]
    new_img = np.copy(img)

    #flip the image
    for i in range(RESIZED_IMG_SIZE):
        for j in range(RESIZED_IMG_SIZE/2):
            temp = img[i][j]
            new_img[i][j] = img[i][cols-j-1]
            new_img[i][cols-j-1] = temp

    # flip the points
    new_points = np.copy(points)
    for i in range(0,42,2):
        new_points[i] = -points[i]

    return new_img, new_points

	

def augment_data(imgs_train, points_train):
    aug_imgs_train = []
    aug_points_train = []
    # apply flipping operation
    for i in tqdm(range(0,imgs_train.shape[0])):
	aug_img, aug_point = flip_img(imgs_train[i], points_train[i])
	# original data
	aug_imgs_train.append(imgs_train[i])
	aug_points_train.append(points_train[i]) 
   
	# augmented data
	aug_imgs_train.append(aug_img)
	aug_points_train.append(aug_point) 

    # convert to numpy
    aug_imgs_train = np.array(aug_imgs_train)
    aug_points_train = np.copy(aug_points_train)
	

    return aug_imgs_train, aug_points_train

def load_dataset(file_path):
    X_train = []
    Y_train = []
    skip_cnt = 0
    line_cnt = 0
    with open(file_path, 'rb') as f:
        for each in xrange(skip_cnt):
            next(f)
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            #if cnt < 40000:
            #    cnt += 1
            #    continue
            print (row[0])
            img = cv2.imread(DATA_PATH+row[0])
            img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_cropped = crop(img, int(row[2]), int(row[1]), int(row[4]), int(row[3]))

            landmarks = map(float, row[5:47])
            landmarks = map(int, landmarks)
            landmarks = np.asarray(landmarks)
            landmarks = landmarks.reshape(21,2)
            landmarks_cropped = landmarks - (int(row[2]), int(row[1]))
    
            img_resized = cv2.resize(img_cropped, (RESIZED_IMG_SIZE, RESIZED_IMG_SIZE))
            scale_x = float(img_resized.shape[1])/img_cropped.shape[1]
            scale_y = float(img_resized.shape[0])/img_cropped.shape[0]
            landmarks_resized = landmarks_cropped * (scale_x, scale_y)
            landmarks_resized = landmarks_resized + (0.5, 0.5)
            landmarks_resized = landmarks_resized.astype(int)
            #plot_landmrks(img_resized, landmarks_resized, '1.jpg')
            lm = landmarks_resized.reshape(42)
            X_train.append(img_resized)
            Y_train.append(lm)
            #print(np.shape(X_train))
            #print(np.shape(Y_train))
            line_cnt += 1
            if line_cnt== 82000:
                break
        print (line_cnt)
        X_train = np.array(X_train)/255.0
        Y_train = np.array(Y_train)/128.0
        #print(X_train[5])
        print('##################')
        #print(Y_train[5])
        return X_train, Y_train
    

def test_model(test_img, model):
    
    with open("dataset_img_size128.pkl", "r") as f:
        img_train, out_train = pickle.load(f)
    
    print(img_train[5])
    test_img =  img_train[5]*255.0
    cv2.imwrite('org_test_img.jpg', test_img)
    print(out_train[5]*128.0)
    points_test = model.predict(img_train[5].reshape(1, RESIZED_IMG_SIZE, RESIZED_IMG_SIZE, 3))
    print(points_test)
    #points_test = points_test * 128
    #points_test = points_test.reshape(21,2)

    #plot_landmrks(img1, points_test, 'test_img_out.jpg')

def test():
    model = define_model4(RESIZED_IMG_SIZE, 42, 500)
    model.load_weights('/data/multimedia/yogesh/2020/hand_pose_estimation/trained_model/model4_trainacc-69_trainloss-15_valacc-67_valloss-54/best_weights_model4.h5')
    
    a = [1, 2,3, 4, 5]
    test_model('./test_img.jpg', model)

	
from tensorflow.python.keras.utils import Sequence
import numpy as np   

class Mygenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        # read your data here using the batch lists, batch_x and batch_y
        x = [my_readfunction(filename) for filename in batch_x] 
        y = [my_readfunction(filename) for filename in batch_y]

        x = [self.x[index] for index in batch_x] 
        y = [self.y[index] for index in batch_y]
		
        return np.array(x), np.array(y)	
	
def data_generator(X_data, y_data, batch_size):

  samples_per_epoch = X_data.shape[0]
  number_of_batches = samples_per_epoch/batch_size
  counter=0

  while 1:

    X_batch = np.array(X_data[batch_size*counter:batch_size*(counter+1)]).astype('float32')
    y_batch = np.array(y_data[batch_size*counter:batch_size*(counter+1)]).astype('float32')
    counter += 1
    yield X_batch,y_batch

    #restart counter to yeild data in the next epoch as well
    if counter >= number_of_batches:
        counter = 0	
	
def main():
    hand_key_points = 21
    num_labels = hand_key_points * 2
    epoch = 500

    checkpoint = ModelCheckpoint('./trained_model/best_weights_model3.h5', verbose=1, save_best_only=True)
    hist = History()
    model = define_model3(RESIZED_IMG_SIZE, num_labels, epoch)
    
    if os.path.exists("data.npz"):
        data = np.load("data.npz")
        img_train = data['img']
        out_train = data['landmarks']
      
        print(img_train.shape)
        print(out_train.shape)

        print(img_train[5])
        print(out_train[5])
    else:
        img_train, out_train = load_dataset("/data/multimedia/Dataset/multiview_hand_pose_dataset_release/data3.csv")
        np.savez('data.npz', img=img_train, landmarks=out_train)
    
    #lrate = ReduceLROnPlateau(monitor='val_loss', factor=0.5,verbose=1,patience=20, min_lr=0.001)
	batch_size = 512
    #hist = model.fit(img_train.reshape(img_train.shape[0], RESIZED_IMG_SIZE, RESIZED_IMG_SIZE, 3), out_train, validation_split=0.2, batch_size=256, callbacks=[checkpoint, hist], shuffle=True, epochs=500, verbose=1)
	hist = hist = model.fit(data_generator(img_train, out_train, batch_size), epochs=500, steps_per_epoch=img_train.shape[0]/batch_size, validation_data=data_generator(img_train, out_train, batch_size*2), validation_steps=img_train.shape[0]/batch_size*2, verbose=1)
    #model.save_weights('trained_model/weights_model4_seva.h5')

    #model.save('trained_model/model_model4.h5')

    #test_model('./test_img.jpg', model)




if __name__ == "__main__":
    main ()
    #test()

