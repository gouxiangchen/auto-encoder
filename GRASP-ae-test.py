import torch
import torch.nn.functional as F
import numpy as np
from util import load_grasp_img
from model import GRASP_Encoder, GRASP_Decoder
from itertools import count
from tensorboardX import SummaryWriter
from itertools import count
import matplotlib.pyplot as plt


def construct_imgs(imgs):
    constructed = np.zeros((1, 8*224, 8*224))
    for i in range(imgs.shape[0]):
        y = int(i / 8)
        x = int(i - 8 * y)
        constructed[0, (224*x):(224*(x+1)), (224*y):(224*(y+1))] = (imgs[i].reshape(224, 224) * 255).astype(np.uint8)
    return constructed


imgs = np.load('img_data.npy')
data_size = imgs.shape[0]
batch_size = 64

encoder = GRASP_Encoder().cuda()
decoder = GRASP_Decoder().cuda()

encoder.load_state_dict(torch.load('./model/encoder_grasp.para'))
decoder.load_state_dict(torch.load('./model/decoder_grasp.para'))

batch_index = np.random.choice(data_size, batch_size)
batch_imgs = imgs[batch_index]

origin = construct_imgs(batch_imgs).reshape(224 * 8, 224 * 8)
plt.figure('origin', figsize=(18, 18))
plt.imshow(origin, cmap='Greys')
plt.show()

batch_imgs = batch_imgs.astype(np.float) / 255
batch_imgs = torch.FloatTensor(batch_imgs).unsqueeze(1).cuda()
hidden_feature = encoder(batch_imgs)
out = decoder(hidden_feature)

img = construct_imgs(out.cpu().detach().numpy()).reshape(224 * 8, 224 * 8)
plt.figure('test', figsize=(18, 18))
plt.imshow(img, cmap='Greys_r')
plt.show()

