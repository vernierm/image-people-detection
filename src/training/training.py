"""
Training and validation.
"""

from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator

"""
GLOBAL VARIABLES
All values are set here.
"""
imgWidth = 96
imgHeight = 160
inputShapeTF = (imgHeight, imgWidth, 3)

trainDataPath = "../../dataset/train"
trainDataNum = 10000 

validationDataPath = "../../dataset/validation"
validationDataNum = 1000

epochsNum = 20
batchSize = 32

modelSavePath = "../../models/proba.h5"

#SIMPLE MODEL
def buildModelSimple():			
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape = inputShapeTF))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return model

#COMPLEX ARCHITECTURE
def buildModelComplex():
    model = Sequential()

    #FIRST BLOCK
    model.add(ZeroPadding2D((1,1), input_shape=inputShapeTF))
    model.add(Conv2D(32, (3, 3)))
    model.add(LeakyReLU(alpha=0.1))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(32, (3, 3)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #SECOND BLOCK
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(32, (3, 3)))
    model.add(LeakyReLU(alpha=0.1))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(32, (3, 3)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #THIRD BLOCK
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3)))
    model.add(LeakyReLU(alpha=0.1))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #DATA DIMENSIONS 12 x 20 x 128
    
    model.add(Flatten())
    model.add(Dense(64))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return model

def trainValidFitModel(model):
    trainDatagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    
    validationDatagen = ImageDataGenerator(
            rescale = 1./255)

    trainGenerator = trainDatagen.flow_from_directory(
            trainDataPath,
            batch_size = batchSize,
            target_size = (imgHeight, imgWidth),
            class_mode = 'binary')

    validationGenerator = validationDatagen.flow_from_directory(
            validationDataPath,
            batch_size = batchSize,
            target_size = (imgHeight, imgWidth),
            class_mode = 'binary')

    model.fit_generator(
            trainGenerator,
            steps_per_epoch = trainDataNum // batchSize,
            epochs = epochsNum,
            validation_data = validationGenerator,
            validation_steps = validationDataNum // batchSize,
            verbose = 2)

if __name__ == "__main__":
    """
    There are two implementations of network architectures that can be used in this program:

    1) buildModelSimple() loads model of architecture I downloaded from:
        https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
    2) buildModelComplex() loads my custom network architecture
    
    """
    model = buildModelSimple()
    model.summary()
    trainValidFitModel(model)
    model.save(modelSavePath)
    