import torch
import torch.nn.functional as F
import numpy as np
from model import GRASP_Encoder, GRASP_Decoder
from tensorboardX import SummaryWriter
from itertools import count


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
optimizer = torch.optim.Adam([{'params': encoder.parameters()}, {'params': decoder.parameters()}],
                             lr=1e-5)

k = 0
writer = SummaryWriter('./logs_grasp')
for t in count():
    batch_index = np.random.choice(data_size, batch_size)
    batch_img = imgs[batch_index].astype(np.float) / 255
    batch_img = torch.FloatTensor(batch_img).unsqueeze(1).cuda()

    hidden_feature = encoder(batch_img)
    out = decoder(hidden_feature)

    loss = F.mse_loss(out, batch_img.flatten(start_dim=1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    writer.add_scalar('loss', loss.item(), t)

    if t % 10000 == 9999:
        torch.save(encoder.state_dict(), './model/encoder_grasp.para')
        torch.save(decoder.state_dict(), './model/decoder_grasp.para')
        print('model saved')
    if t % 100 == 0:
        print('epoch', t, 'loss : ', loss.item())
        writer.add_image('reconstruct', construct_imgs(out.cpu().detach().numpy()), k)
        k += 1



