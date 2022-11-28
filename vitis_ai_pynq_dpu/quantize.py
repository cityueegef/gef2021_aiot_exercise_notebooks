import os
 
# Silence TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import tensorflow.keras.models as models
from tensorflow_model_optimization.quantization.keras import vitis_quantize

MODEL_DIR = './' # Output directory
FLOAT_MODEL = 'float_model.h5'
QUANT_MODEL = 'quantized_model.h5'

# Loading the float model
print('Loading the float model...')
path = os.path.join(MODEL_DIR, FLOAT_MODEL)
try:
    float_model = models.load_model(path)
except:
    print('\nError:loading float model failed!')


# Write your code here
#================================================================================
print("\nLoading the Fashion MNIST dataset...")
# Loading the Fashion MNIST dataset

# Reshaping the data and feature scaling

# Defining the calibration dataset using about 1000 samples from the training set

print('\nRunning model quantization...')
# Running model quantization



# Saving the quantized model
path = os.path.join(MODEL_DIR, QUANT_MODEL)

print(f'\nQuantized model has been saved to "{path}"')
#================================================================================

