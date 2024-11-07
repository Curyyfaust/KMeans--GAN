import torch
import torch.nn as nn
import torch.optim as optim


#定义生成器
# 生成器网络
class Generator(nn.Module):
    def __init__(self, noise_dim, condition_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim + condition_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, output_dim),
            nn.Tanh()  # 假设时间序列数据归一化到 -1 到 1
        )

    def forward(self, noise, condition):
        # 将噪声和条件输入连接
        input = torch.cat([noise, condition], dim=1)
        return self.model(input)

# 判别器网络
class Discriminator(nn.Module):
    def __init__(self, input_dim, condition_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim + condition_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, data, condition):
        # 将数据和条件输入连接
        input = torch.cat([data, condition], dim=1)
        return self.model(input)

