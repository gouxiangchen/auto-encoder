import torch
import torch.nn.functional as F
import numpy as np
from util import load_mnist_data
from model import MNIST_Discriminator, MNIST_Generator
from itertools import count
from tensorboardX import SummaryWriter


def construct_imgs(imgs):
    constructed = np.zeros((1, 16*28, 16*28))
    for i in range(imgs.shape[0]):
        y = int(i / 16)
        x = int(i - 16 * y)
        constructed[0, (28*x):(28*(x+1)), (28*y):(28*(y+1))] = (imgs[i].reshape(28, 28) * 255).astype(np.uint8)
    return constructed


d_steps = 10
g_steps = 10

input_dim = 5

discriminator = MNIST_Discriminator().cuda()
generator = MNIST_Generator(input_dim=input_dim).cuda()
imgs, labels = load_mnist_data('/home/chen/datasets/MNIST')
batch_size = 256
writer = SummaryWriter('logs_gan')

d_optim = torch.optim.Adam(discriminator.parameters(), lr=1e-5)
g_optim = torch.optim.Adam(generator.parameters(), lr=1e-5)

fake_label = torch.zeros(batch_size, 1).cuda()
true_label = torch.ones(batch_size, 1).cuda()

k = 0
p = 0
q = 0
loss_func = torch.nn.BCELoss()
for t in count():
    for i in range(d_steps):
        p += 1
        rand_input = torch.rand(batch_size, input_dim).cuda()
        fake_imgs = generator(rand_input)
        batch_index = np.random.choice(len(labels), batch_size)
        batch_imgs = imgs[batch_index].astype(np.float) / 255
        batch_imgs = torch.FloatTensor(batch_imgs).cuda()

        d = discriminator(fake_imgs)
        floss = loss_func(d, fake_label)

        d = discriminator(batch_imgs)
        tloss = loss_func(d, true_label)

        loss = floss + tloss
        d_optim.zero_grad()
        loss.backward()
        d_optim.step()

        writer.add_scalars('discriminator loss', {'fake loss': floss.item(), 'true loss': tloss.item()}, global_step=p)

    for i in range(g_steps):
        q += 1
        rand_input = torch.rand(batch_size, input_dim).cuda()
        fake_imgs = generator(rand_input)
        d = discriminator(fake_imgs)
        loss = loss_func(d, true_label)
        g_optim.zero_grad()
        loss.backward()
        g_optim.step()
        writer.add_scalar('generator loss', loss.item(), global_step=q)

    if t % 10000 == 9999:
        print('model saved')
        torch.save(discriminator.state_dict(), './model/discriminator.para')
        torch.save(generator.state_dict(), './model/generator.para')
    if t % 100 == 0:
        print('epoch : ', t, 'image generated')
        with torch.no_grad():
            rand_input = torch.rand(batch_size, input_dim).cuda()
            out = generator(rand_input)
        writer.add_image('reconstruct', construct_imgs(out.cpu().numpy()), k)
        k += 1



