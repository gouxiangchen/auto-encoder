import torch.nn as nn


class MNIST_Encoder(nn.Module):
    def __init__(self, input_dim=784, output_dim=5):
        super(MNIST_Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, output_dim)
        self.relu3 = nn.ReLU()

        nn.init.xavier_normal_(self.fc1.weight.data)
        nn.init.xavier_normal_(self.fc2.weight.data)
        nn.init.xavier_normal_(self.fc3.weight.data)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x


class MNIST_Decoder(nn.Module):
    def __init__(self, input_dim=5, output_dim=784):
        super(MNIST_Decoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 1024)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(1024, output_dim)
        self.sigmoid = nn.Sigmoid()

        nn.init.xavier_normal_(self.fc1.weight.data)
        nn.init.xavier_normal_(self.fc2.weight.data)
        nn.init.xavier_normal_(self.fc3.weight.data)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


class MNIST_VAE(nn.Module):
    def __init__(self, input_dim=784, output_dim=5):
        super(MNIST_VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 256)
        self.relu2 = nn.ReLU()
        self.mean = nn.Linear(256, output_dim)
        self.std = nn.Linear(256, output_dim)
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

        nn.init.xavier_normal_(self.fc1.weight.data)
        nn.init.xavier_normal_(self.fc2.weight.data)
        nn.init.xavier_normal_(self.mean.weight.data)
        nn.init.xavier_normal_(self.std.weight.data)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        mean = self.mean(x)
        std = self.softplus(self.std(x))
        return mean, std


class MNIST_VAD(nn.Module):
    def __init__(self, input_dim=5, output_dim=784):
        super(MNIST_VAD, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 1024)
        self.relu2 = nn.ReLU()
        self.mean = nn.Linear(1024, output_dim)
        self.sigmoid = nn.Sigmoid()
        nn.init.xavier_normal_(self.fc1.weight.data)
        nn.init.xavier_normal_(self.fc2.weight.data)
        nn.init.xavier_normal_(self.mean.weight.data)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        mean = self.sigmoid(self.mean(x))
        return mean


class GRASP_Encoder(nn.Module):
    def __init__(self, output_dim=64):
        super(GRASP_Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=6, stride=2)  # 224 * 224 * 1 -> 110 * 110 * 32
        self.relu1 = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)  # 110 * 110 * 32 -> 55 * 55 * 32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # 55 * 55 * 32 -> 55 * 55 * 64
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 64, kernel_size=5, stride=2)  # 55 * 55 * 64 -> 26 * 26 * 64
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(26 * 26 * 64, 1024)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(1024, output_dim)

        nn.init.kaiming_normal_(self.conv1.weight.data)
        nn.init.kaiming_normal_(self.conv2.weight.data)
        nn.init.kaiming_normal_(self.conv3.weight.data)
        nn.init.kaiming_normal_(self.fc1.weight.data)
        nn.init.kaiming_normal_(self.fc2.weight.data)

        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)
        nn.init.zeros_(self.conv3.bias)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.fc1(x.view(x.size(0), -1))
        x = self.relu4(x)
        x = self.fc2(x)
        return x


class GRASP_Decoder(nn.Module):
    def __init__(self, input_dim=64):
        super(GRASP_Decoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 2048)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(2048, 224 * 224)
        self.sigmoid = nn.Sigmoid()

        nn.init.kaiming_normal_(self.fc1.weight.data)
        nn.init.kaiming_normal_(self.fc2.weight.data)
        nn.init.kaiming_normal_(self.fc3.weight.data)

        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


class MNIST_Discriminator(nn.Module):
    def __init__(self, input_dim=784):
        super(MNIST_Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

        nn.init.xavier_normal_(self.fc1.weight.data)
        nn.init.xavier_normal_(self.fc2.weight.data)
        nn.init.xavier_normal_(self.fc3.weight.data)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


class MNIST_Generator(nn.Module):
    def __init__(self, input_dim=100, output_dim=784):
        super(MNIST_Generator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 1024)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(1024, output_dim)
        self.sigmoid = nn.Sigmoid()

        nn.init.xavier_normal_(self.fc1.weight.data)
        nn.init.xavier_normal_(self.fc2.weight.data)
        nn.init.xavier_normal_(self.fc3.weight.data)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x
