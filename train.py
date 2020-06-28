from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.utils import np_utils, generic_utils, to_categorical
import keras	
import sys
from keras import regularizers
from keras.regularizers import l2
import numpy as np
from keras import backend as K
K.set_image_dim_ordering('th')


batch_size = 128
nb_classes = 10
nb_epoch = 100

img_channels = 3
img_rows = 112
img_cols = 112

if len(sys.argv)>1:
   X_train = np.load(sys.argv[1])
   Y_train = np.load(sys.argv[2])
else:
   X_train = np.load('x_train.npy')
   Y_train = np.load('y_train.npy')


print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')


print('Y_train shape:', Y_train.shape)
# print('Y_test shape:', Y_test.shape)


Y_train = to_categorical(Y_train, nb_classes)
print('Y_train shape:', Y_train.shape)


#exit()

model = Sequential()


#Layer 1
model.add(Conv2D(filters=128, kernel_size=(3,3), padding='valid', activation="relu", input_shape=(3,112,112), kernel_regularizer=regularizers.l2(0.0000005)))#Convo$
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.50))

model.add(Conv2D(filters=128, kernel_size=(3,3), padding='valid', activation="relu", kernel_regularizer=regularizers.l2(0.0000005)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.50))

model.add(Conv2D(filters=128, kernel_size=(3,3), padding='valid', activation="relu", kernel_regularizer=regularizers.l2(0.0000005)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.50))

model.add(Conv2D(filters=256, kernel_size=(3,3), padding='valid', activation="relu", kernel_regularizer=regularizers.l2(0.0000005)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.50))

model.add(Conv2D(filters=256, kernel_size=(3,3), padding='valid', activation="relu", kernel_regularizer=regularizers.l2(0.0000005)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.50))


model.add(Flatten())# shape equals to [batch_size, 32] 32 is the number of filters
model.add(Dense(10))#Fully connected layer
model.add(Activation('softmax'))


keras.utils.multi_gpu_model(model, gpus=2, cpu_merge=False, cpu_relocation=False)
opt = keras.optimizers.Adamax(lr=0.001, beta_1=0.9, beta_2=0.99, decay=1e-6)# best one

model.compile(loss='categorical_crossentropy',
           optimizer=opt,
           metrics=['accuracy'])

def train():
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=nb_epoch,
              shuffle=True)
    if len(sys.argv)>3:
        model.save(sys.argv[3])
    else:
        model.save('model.h5')
#train()

