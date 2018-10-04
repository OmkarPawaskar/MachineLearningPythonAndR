#Convolutional Neural Network

# Installing Theano - open source numerical computations library- can run on CPU as well as GPU
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing TensorFlow - open source numerical computations library- can run on CPU as well as GPU
# Install TensorFlow from website : https://www.tensorflow.org/versions/r0.11/get_started/
#or go to anaconda prompt - conda create -n tensorflow python = 3.6.6 -> conda activate tensorflow ->pip install --ignore-installed --upgrade tensorflow

# Installing Keras - based on Theano and Tensorflow.
# pip install --upgrade keras

# Part 1 - Building CNN

# Importing the Keras libraries and packages
from keras.models import Sequential #to initialize neural network :2 types :as sequence of layers or as graph ..we choose sequence of layers
from keras.layers import Convolution2D #to add convolutional layers..since it is images we use Convolution2D
from keras.layers import MaxPooling2D #Pooling step -to reduce dimensions,to be able to recognize entity even if its turned and twisted,faster performance
from keras.layers import Flatten #to convert into large featue vector
from keras.layers import Dense #to add fully connected layer and neural network

# Initializing the CNN
classifier = Sequential()

#Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))#32-no. of feature detectors, 3,3-dimensions,input_shape(64x64 pixels,3d colored image)

#Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))


#Step 3 - Flattening
classifier.add(Flatten()) 

#Step 4 - Full Connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

#Compiling the CNN
classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy', metrics= ['accuracy'])

# Part 2 - Fitting the CNN to images
#Image Augmentation - it will create various copies of images and perform transformations on it to increase the size of dataset to help model
#check keras documentation to understand method - /Image preprocessing
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,#to get pixel values between 0 and 1
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 25,
                         validation_data = test_set,
                         nb_val_samples = 2000)


