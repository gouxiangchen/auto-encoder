import torch
import torch.nn.functional as F
import numpy as np
from util import load_mnist_data
from model import MNIST_Encoder, MNIST_Decoder
from itertools import count
from tensorboardX import SummaryWriter


def construct_imgs(imgs):
    constructed = np.zeros((1, 16*28, 16*28))
    for i in range(imgs.shape[0]):
        y = int(i / 16)
        x = int(i - 16 * y)
        constructed[0, (28*x):(28*(x+1)), (28*y):(28*(y+1))] = (imgs[i].reshape(28, 28) * 255).astype(np.uint8)
    return constructed


imgs, labels = load_mnist_data('/home/chen/datasets/MNIST')
batch_size = 256
writer = SummaryWriter('logs_re')
encoder = MNIST_Encoder().cuda()
decoder = MNIST_Decoder().cuda()
optimizer = torch.optim.Adam([{'params': encoder.parameters()}, {'params': decoder.parameters()}],
                             lr=1e-5)

k = 0
for t in count():
    batch_index = np.random.choice(len(labels), batch_size)
    batch_imgs = imgs[batch_index].astype(np.float) / 255
    batch_imgs = torch.FloatTensor(batch_imgs).cuda()

    hidden_feature = encoder(batch_imgs)
    out = decoder(hidden_feature)
    loss = F.mse_loss(out, batch_imgs)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    writer.add_scalar('loss', loss.item(), t)

    if t % 10000 == 9999:
        torch.save(encoder.state_dict(), './model/encoder_re.para')
        torch.save(decoder.state_dict(), './model/decoder_re.para')
    if t % 100 == 0:
        print('epoch', t, 'loss : ', loss.item())
        writer.add_image('reconstruct', construct_imgs(out.cpu().detach().numpy()), k)
        k += 1

