from PIL import Image
import numpy as np
CLASSES = ['Airplane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    img = img.resize((224, 224), Image.BILINEAR)
    data = np.asarray( img, dtype="int32" )
    return data
img = load_image('tabby.png')

import torch
network = torch.load("mobn.pth")
network.eval();

mean = np.array([0.485, 0.456, 0.406]);
std = np.array([0.229, 0.224, 0.225]);
#mean = np.array([0.5, 0.5, 0.5]);
#std = np.array([0.5, 0.5, 0.5]);
img = (img/255.0 - mean)/std;

#print(img)

i = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float();
out = network(i);
print(out.data)
print(out.data.numpy().argmax());
print(softmax(out.data.numpy()));
print(CLASSES[out.data.numpy().argmax()]);

"""
import torchvision
import torch
transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False)"""
