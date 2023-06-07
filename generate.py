from keras2c import k2c
#from tensorflow import keras
import os
from huggingface_hub import from_pretrained_keras
import keras
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
# Import python libraries required in this example:
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from numpy.random import seed
from keras.models import load_model
from keras.initializers import RandomUniform
seed(1)

def random_tensor(shape):
  return np.random.rand(*shape)

def float_to_nbit_uint(x, n=8):
  # Convert the tensor to an n-bit unsigned int tensor
  x_int = [(int(i * (2 ** n - 1)) & (2 ** n - 1)) for i in x]
  return x_int
# Create a sequential model
model = Sequential()

# Add a 2D convolutional layer with 2 filters, a 3x3 kernel, and 'relu' activation
model.add(Conv2D(2, (3, 3), activation='relu', input_shape=(32, 32, 3)))

# Add a max pooling layer with 2x2 pool size
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the input to a 1D array
model.add(Flatten())

# Add a fully connected layer with 2 units and 'relu' activation
model.add(Dense(2, activation='relu'))

# Compile the model with mean squared error loss and stochastic gradient descent (SGD) optimizer
model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])

# Print the model summary
model.summary()

weights = model.layers[0].get_weights()[0]
biases = model.layers[0].get_weights()[1]

print("Weight matrix:")
print(weights)

print("\nBias vector:")
print(biases)
#model = from_pretrained_keras()

model = load_model('resnet9_cifar10.h5')

#function_name = 'resnet20'
function_name = 'simplenet'

try:
  os.remove(function_name+".c")
except:
  print(function_name+ ".c not found")
try:
  os.remove(function_name+".h")
except:
  print(function_name+".h not found")
try:
  os.remove(function_name+"_test_suite.c")
except:
  print(function_name+"_test_suite.c not found")


#model = keras.models.load_model("convnet.h5")
#model = ResNet50(weights='imagenet')
#model.summary()
#model = from_pretrained_keras("nateraw/keras-mnist-convnet-demo")

model = from_pretrained_keras("merve/mnist")

# allowed datatype for now are 'float ', 'int ', and 'int8_t ', 'bool ' and 'fixed '
k2c(model, function_name, malloc=False, num_tests=1, verbose=True, datatype='fixed ')
