import torch
import torch.nn as nn
import utils_network
import numpy as np


# -----------------------------------
# FFDNet Model
# -----------------------------------
class FFDNet(nn.Module):

    def __init__(self, is_gray):
        super(FFDNet, self).__init__()

        bias = True
        sf = 2      #sf: scaling factor

        if is_gray:
            self.nl= 15
            self.nc = 64
            self.in_nc = 1
            self.out_nc = 1
        else:
            self.nl = 12
            self.nc = 96
            self.in_nc = 3
            self.out_nc = 3
            
        self.kernel_size = 3
        self.padding = 1

        self.m_down = utils_network.Pixel_unshuffle(upscale_factor=sf)
        
        layers = []
        
        layers.append(nn.Conv2d(in_channels=self.in_nc*sf*sf+1, out_channels=self.nc, 
                                kernel_size=self.kernel_size, padding=self.padding, bias=True))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(self.nl-2):
            layers.append(nn.Conv2d(in_channels=self.nc, out_channels=self.nc,
                                    kernel_size=self.kernel_size, padding=self.padding, bias=True))
            layers.append(nn.BatchNorm2d(self.nc))
            layers.append(nn.ReLU(inplace=True))
        
        layers.append(nn.Conv2d(in_channels=self.nc, out_channels=self.out_nc*sf*sf,
                                kernel_size=self.kernel_size, padding=self.padding, bias=True))
        
        
        self.main = nn.Sequential(*layers)
        
        self.m_up = nn.PixelShuffle(upscale_factor=sf)
        

    def forward(self, x, sigma):

        h, w = x.shape[-2:]
        # Pixel need to be multiplication of 2
        paddingBottom = int(np.ceil(h/2)*2-h)
        paddingRight = int(np.ceil(w/2)*2-w)
        x = torch.nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)

        x = self.m_down(x)
        
        sigma = sigma.view(sigma.shape[0], 1, 1, 1)
        n = sigma.repeat(1, 1, x.shape[2], x.shape[3])
        x = torch.cat((x, n), 1)
        x = self.main(x)
        x = self.m_up(x)

        x = x[..., :h, :w]
        return x


if __name__ == '__main__':
    model = FFDNet(is_gray = False)
    
    
    #x = torch.randn([64,1,32,32])
    #n = torch.ones([64,1])
    
    #model(x,n)
    
    print(model)
    for name, parameter in model.named_parameters():
        print('name:', name, 'shape:', parameter.shape)
        # if name == 'main.3.weight':
        #     print(parameter.shape)



