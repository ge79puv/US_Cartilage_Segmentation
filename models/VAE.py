import torch
import torch.nn as nn
from torch import Tensor
from functools import reduce
from operator import mul
from typing import Tuple





class VAE(nn.Module):
    def __init__(self, input_shape: Tuple[int, int, int], output_shape, init_channels=48, z_dim=5):    # z_dim !!!
        super(VAE, self).__init__()
        block_in_channels = input_shape[0]
        blocks = 3                              # 4

        ########### Encoder
        
        block_out_channels = init_channels * 2**0
        self.conv1 = nn.Sequential(
            nn.Conv2d(block_in_channels, block_out_channels, kernel_size=3, stride=2, padding=1, bias=True),         
            nn.BatchNorm2d(block_out_channels),
            nn.ELU(inplace=True),
            nn.Conv2d(block_out_channels, block_out_channels, kernel_size=3, stride=1, padding='same', bias=True),
            nn.BatchNorm2d(block_out_channels),
            nn.ELU(inplace=True),
        )
        
        block_out_channels = init_channels * 2**1
        self.conv2 = nn.Sequential(
            nn.Conv2d(init_channels * 2**0, block_out_channels, kernel_size=3, stride=2, padding=1, bias=True),         
            nn.BatchNorm2d(block_out_channels),
            nn.ELU(inplace=True),
            nn.Conv2d(block_out_channels, block_out_channels, kernel_size=3, stride=1, padding='same', bias=True),
            nn.BatchNorm2d(block_out_channels),
            nn.ELU(inplace=True),
        )
        
        block_out_channels = init_channels * 2**2
        self.conv3 = nn.Sequential(
            nn.Conv2d(init_channels * 2**1, block_out_channels, kernel_size=3, stride=2, padding=1, bias=True),         
            nn.BatchNorm2d(block_out_channels),
            nn.ELU(inplace=True),
            nn.Conv2d(block_out_channels, block_out_channels, kernel_size=3, stride=1, padding='same', bias=True),
            nn.BatchNorm2d(block_out_channels),
            nn.ELU(inplace=True),
        )
        '''
        block_out_channels = init_channels * 2**3
        self.conv4 = nn.Sequential(
            nn.Conv2d(init_channels * 2**2, block_out_channels, kernel_size=3, stride=2, padding=1, bias=True),         
            nn.BatchNorm2d(block_out_channels),
            nn.ELU(inplace=True),
            nn.Conv2d(block_out_channels, block_out_channels, kernel_size=3, stride=1, padding='same', bias=True),
            nn.BatchNorm2d(block_out_channels),
            nn.ELU(inplace=True),
        )
        '''
        self.bottleneck1 = nn.Sequential(
            nn.Conv2d(init_channels * 2**2, init_channels, kernel_size=3, stride=2, padding=1, bias=True),         
            nn.BatchNorm2d(init_channels),
            nn.ELU(inplace=True),
        )

        feature_shape = (init_channels, input_shape[1] // 2 ** (blocks + 1), input_shape[2] // 2 ** (blocks + 1))
        self.mu_head = nn.Linear(reduce(mul, feature_shape), z_dim)        # (48, 15, 20)
        self.logvar_head = nn.Linear(reduce(mul, feature_shape), z_dim)         


        ########### Decoder

        block_in_channels = init_channels

        # Projection from encoding to bottleneck
        self.feature_shape = (init_channels, output_shape[1] // 2 ** (blocks + 1), output_shape[2] // 2 ** (blocks + 1))
        # print(z_dim, self.feature_shape)     #(48, 8, 10)
        self.projection = nn.Linear(z_dim, reduce(mul, self.feature_shape))
        '''
        block_out_channels = init_channels * 2**3
        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(block_in_channels, block_out_channels, kernel_size=2, stride=2, padding=0, bias=True),         
            nn.BatchNorm2d(block_out_channels),
            nn.ELU(inplace=True),
            nn.Conv2d(block_out_channels, block_out_channels, kernel_size=3, stride=1, padding='same', bias=True),
            nn.BatchNorm2d(block_out_channels),
            nn.ELU(inplace=True),
        )
        '''
        block_out_channels = init_channels * 2**2
        self.conv6 = nn.Sequential(
            nn.ConvTranspose2d(block_in_channels, block_out_channels, kernel_size=2, stride=2, padding=0, bias=True),         
            nn.BatchNorm2d(block_out_channels),
            nn.ELU(inplace=True),
            nn.Conv2d(block_out_channels, block_out_channels, kernel_size=3, stride=1, padding='same', bias=True),
            nn.BatchNorm2d(block_out_channels),
            nn.ELU(inplace=True),
        )
        
        block_out_channels = init_channels * 2**1
        self.conv7 = nn.Sequential(
            nn.ConvTranspose2d(init_channels * 2**2, block_out_channels, kernel_size=2, stride=2, padding=0, bias=True),         
            nn.BatchNorm2d(block_out_channels),
            nn.ELU(inplace=True),
            nn.Conv2d(block_out_channels, block_out_channels, kernel_size=3, stride=1, padding='same', bias=True),
            nn.BatchNorm2d(block_out_channels),
            nn.ELU(inplace=True),
        )
        
        block_out_channels = init_channels * 2**0
        self.conv8 = nn.Sequential(
            nn.ConvTranspose2d(init_channels * 2**1, block_out_channels, kernel_size=2, stride=2, padding=0, bias=True),         
            nn.BatchNorm2d(block_out_channels),
            nn.ELU(inplace=True),
            nn.Conv2d(block_out_channels, block_out_channels, kernel_size=3, stride=1, padding='same', bias=True),
            nn.BatchNorm2d(block_out_channels),
            nn.ELU(inplace=True),
        )

        self.bottleneck2 = nn.Sequential(
            nn.ConvTranspose2d(block_out_channels, block_out_channels, kernel_size=2, stride=2, padding=0, bias=True),         
            nn.BatchNorm2d(block_out_channels),
            nn.ELU(inplace=True),
        )

        # Classifier: outputs the pixel-wise unnormalized score for each class in the input's reconstruction.
        self.classifier = nn.Conv2d(block_out_channels, output_shape[0], kernel_size=3, stride=1, padding='same')
        

    def encode(self, x):
        features = self.conv3(self.conv2(self.conv1(x)))      # self.conv4()
        features = self.bottleneck1(features)           # torch.Size([2, 48, 8, 10])

        features = torch.flatten(features, 1)           # 只保留第一个的batch维度，每一行代表一个图片的所有元素 2x3840
        mu = self.mu_head(features)                     # features: torch.Size([2, 3840])   self.mu_head: in_features=3840, out_features=32
        logvar = self.logvar_head(features)

        return mu, logvar                 
    

    def decode(self, z):
        
        features = self.projection(z)
        features = self.conv8(self.conv7(self.conv6(features.view((-1, *self.feature_shape)))))     # 恢复成batch的形式，4个dimension  self.conv5()
        features = self.bottleneck2(features)
        out = torch.sigmoid(self.classifier(features))         # F.softmax(self.classifier(features), dim=1)

        return out


    # Randomly sample latent vectors from a distribution in a way that allows backpropagation to flow through
    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """Samples item from a distribution in a way that allows backpropagation to flow through.

        Args:
            mu: (N, M), Mean of the distribution.
            logvar: (N, M), Log variance of the distribution.

        Returns:
            (N, M), Item sampled from the distribution.
        """
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)

        return mu + eps * std
    

    def forward(self, x):

        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        #x_reconst = torch.argmax(self.decode(z), axis=1).type(torch.float32)
        x_reconst = self.decode(z)

        return x_reconst, z, mu, log_var        




