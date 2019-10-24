### Install dependencies

pip install git+https://github.com/nerox8664/pytorch2keras
pip install tensorflowjs

### Convert the saved MobileNetV2 model to Tensorflow.js format (PyTorch -> Onnx -> Keras -> Tensorflow.js)

python convert_model.py --fulln=cifar_MobileNetV2.pth

This converts the PyTorch model and produces two folders: `mobilenetjs` and `mobilenetjs_quantised`. The latter contains the 8-bit quantized version and is loaded in index.html.

Most Tensorflow warnings output from the script can be ignored.


