import torch
import numpy as np
from util import load_mnist_data
from model import MNIST_Encoder, MNIST_Decoder
import matplotlib.pyplot as plt
from PIL import Image


def construct_imgs(imgs):
    constructed = np.zeros((1, 16*28, 16*28))
    for i in range(imgs.shape[0]):
        y = int(i / 16)
        x = int(i - 16 * y)
        constructed[0, (28*x):(28*(x+1)), (28*y):(28*(y+1))] = (imgs[i].reshape(28, 28) * 255).astype(np.uint8)
    return constructed


imgs, labels = load_mnist_data('/home/chen/datasets/MNIST')
batch_size = 256
encoder = MNIST_Encoder().cuda()
decoder = MNIST_Decoder().cuda()
encoder.load_state_dict(torch.load('./model/encoder.para'))
decoder.load_state_dict(torch.load('./model/decoder.para'))

batch_index = np.random.choice(len(labels), batch_size)
batch_imgs = imgs[batch_index].astype(np.float) / 255
batch_imgs = torch.FloatTensor(batch_imgs).cuda()
hidden_feature = encoder(batch_imgs)
out = decoder(hidden_feature)

img = construct_imgs(out.cpu().detach().numpy()).reshape(448, 448)
plt.figure('test')
plt.imshow(img, cmap='Greys_r')
plt.show()
