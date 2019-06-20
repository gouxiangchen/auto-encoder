import torch
import numpy as np
from model import MNIST_Discriminator, MNIST_Generator
import matplotlib.pyplot as plt


def construct_imgs(imgs):
    constructed = np.zeros((1, 16*28, 16*28))
    for i in range(imgs.shape[0]):
        y = int(i / 16)
        x = int(i - 16 * y)
        constructed[0, (28*x):(28*(x+1)), (28*y):(28*(y+1))] = (imgs[i].reshape(28, 28) * 255).astype(np.uint8)
    return constructed


input_dim = 5

discriminator = MNIST_Discriminator().cuda()
generator = MNIST_Generator(input_dim=input_dim).cuda()

batch_size = 256

discriminator.load_state_dict(torch.load('./model/discriminator.para'))
generator.load_state_dict(torch.load('./model/generator.para'))

rand_input = torch.ones(batch_size, input_dim).cuda()

fake_img = generator(rand_input)
imgs = construct_imgs(fake_img.cpu().detach().numpy()).reshape(448, 448)
plt.figure('test')
plt.imshow(imgs, cmap='Greys_r')
plt.show()
