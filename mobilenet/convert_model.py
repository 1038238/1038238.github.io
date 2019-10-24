import os; 
# ensure Keras is using tensorflow regardless of ~/.keras/keras.json
os.environ['KERAS_BACKEND'] = 'tensorflow'

import argparse
import numpy as np
import tensorflowjs as tfjs
import tensorflow as tf
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
tf.compat.v1.disable_eager_execution()

import keras
# Ensure Keras is using a channels first image format for PyTorch regardless of ~/.keras/keras.json
keras.backend.set_image_data_format('channels_first')
from pytorch2keras import pytorch_to_keras

parser = argparse.ArgumentParser()
parser.add_argument('--fulln', type=str, required=True)
args = parser.parse_args()

# Load trained MobileNetV2 model
n_classes = 10
network = torchvision.models.MobileNetV2()
n_inputs = network.classifier[1].in_features
network.classifier[1] = nn.Linear(in_features=n_inputs, out_features=n_classes)       
network.load_state_dict(torch.load(args.fulln, map_location=torch.device('cpu')))

# Add an explicit AvgPool layer, since Onnx cannot interpret mean() (sees it as a Lambda layer)
print(network.classifier)
classifier = nn.Sequential(
    nn.AvgPool2d(7, stride=1),
    nn.Flatten(),
    nn.Dropout(0.2),
    nn.Linear(1280, 10),
)
def forward(self, x):
    x = self.features(x)
    #x = x.mean([2,3])  # cannot be exported to Onnx
    x = self.classifier(x)
    return x
# Monkey patch the model
network.__class__.forward = forward
classifier[2] = network.classifier[0]
classifier[3] = network.classifier[1]
network.classifier = classifier

# Convert to Keras format
IMAGE_SIZE = 224
input_np = np.random.uniform(0, 1, (1, 3, IMAGE_SIZE, IMAGE_SIZE))
network.eval();
with torch.no_grad():
    input_var = Variable(torch.FloatTensor(input_np))
    pytorch_output = network(input_var).data.numpy()
    k_model = pytorch_to_keras(network, input_var, [(3, IMAGE_SIZE, IMAGE_SIZE)], verbose=False, name_policy='renumerate')  
    keras_output = k_model.predict(input_np)
    # This should be less than 1e-6
    error = np.max(pytorch_output - keras_output)
    print('Conversion prediction error =',error,"(fine if around 1e-6)")
    #print(k_model.summary())

# Convert to Tensorflow.js
tfjs.converters.save_keras_model(k_model, 'mobilenetjs')
tfjs.converters.save_keras_model(k_model, 'mobilenetjs_quantised', quantization_dtype=np.uint8)

# Bug with reshape in tensorflowjs https://github.com/tensorflow/tfjs/issues/824
# Fix by editing expected shape in model.json:
def fix_model_json(location):
    import json
    with open(location, "r") as f:
        modeljson = json.load(f)
        #print(modeljson)
    # Find flatten layer ("LAYER_151")
    for layer in modeljson["modelTopology"]["model_config"]["config"]["layers"]:
        if(layer["name"] == "LAYER_151"):
            # change from [-1] to [-1, 1280] (size of subsequent linear layer)
            layer["config"]["target_shape"] = [-1, 1280] 
            break;
    # Overwrite original definition
    with open(location, "w") as w:
        json.dump(modeljson, w)
fix_model_json("mobilenetjs/model.json")
fix_model_json("mobilenetjs_quantised/model.json")

print("Finished conversion!")
