### Install dependencies

pip install git+https://github.com/nerox8664/pytorch2keras
pip install tensorflowjs

### Convert the saved MobileNetV2 model to Tensorflow.js format (PyTorch -> Onnx -> Keras -> Tensorflow.js)

python convert_model.py --fulln=cifar_MobileNetV2.pth

This converts the PyTorch model and produces two folders: `mobilenetjs` and `mobilenetjs_quantised`. The latter contains the 8-bit quantized version and is loaded in index.html.

Most Tensorflow warnings output from the script can be ignored.

### Start a web server and open a browser to localhost:8000

python -m http.server

### Image Credit

All images are free-to-use, sourced from unsplash.com with attributions to the photographers below:

cat.jpg: Caleb Woods (https://unsplash.com/photos/9KpQrPEy8P8)
dog.jpg: Angel Luciano (https://unsplash.com/photos/LATYeZyw88c)
bird.jpg: Boris Smokrovic (https://unsplash.com/photos/T_diWdiLbg0)
ship.jpg: Kinsey (https://unsplash.com/photos/cB8YiJt_0Y0)
truck.jpg: Rhys Moult (https://unsplash.com/photos/7eaFIKeo1MQ)
springbok.jpg: Cameron Oxley (https://unsplash.com/photos/DU4fuhD5CMI)
frog.jpg: Joel Henry (https://unsplash.com/photos/Rcvf6-n1gc8)
horse.jpg: Helena Lopes (https://unsplash.com/photos/lIeqGEdvex0)
airplane.jpg: Leio McLaren (https://unsplash.com/photos/FwdZYz0yc9g)
car.jpg: Sven D (https://unsplash.com/photos/a4S6KUuLeoM)