from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.utils import np_utils, generic_utils, to_categorical
import numpy as np
import keras
from keras.models import load_model
import sys

classes=10
X_test = np.load(sys.argv[1])
Y_test = np.load(sys.argv[2])
Y_test = to_categorical(Y_test, classes)
model_file = sys.argv[3]
model = load_model(model_file)

def score():
    predicted_classes = model.predict_classes(X_test)
    misclassifies = np.where(predicted_classes != Y_test)
#    print(len(predicted_classes))
#   print(len(misclassifies))
    score = model.evaluate(X_test,Y_test)
    print(model.metrics_names[1], score[1]*100)

score()

