import torch
import torch.nn as nn
class Conv(nn.Module):
    def __init__(self, in_c=1, out_c=1, padding=2):
        super(Conv, self).__init__()
        self.CHconv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=9, stride=1, padding=4, padding_mode="replicate"),
            nn.ReLU(),
            nn.Conv2d(in_c, out_c, kernel_size=7, stride=1, padding=3, padding_mode="replicate"),
            nn.ReLU()
        )

        self.PARconv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=9, stride=1, padding=4, padding_mode="replicate"),
            nn.ReLU(),
            nn.Conv2d(in_c, out_c, kernel_size=7, stride=1, padding=3, padding_mode="replicate"),
            nn.ReLU()
        )
        
        self.SSTconv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=9, stride=1, padding=4, padding_mode="replicate"),
            nn.ReLU(),
            nn.Conv2d(in_c, out_c, kernel_size=7, stride=1, padding=3, padding_mode="replicate"),
            nn.ReLU()
        )

    def forward(self, x):

        if x.shape[0]==1:
            '''prediction'''
            x_chlor = x[:, :, 0:2304].reshape(x.shape[1],1,48,48)
            x_par = x[:, :, 2304:4608].reshape(x.shape[1],1,48,48)
            x_sst = x[:, :, 4608:6912].reshape(x.shape[1],1,48,48)
        else:
            x_chlor = x[:, :, 0:2304].reshape(32*x.shape[1],1,48,48)
            x_par = x[:, :, 2304:4608].reshape(32*x.shape[1],1,48,48)
            x_sst = x[:, :, 4608:6912].reshape(32*x.shape[1],1,48,48)


        x_chlor = self.CHconv(x_chlor)
        x_par = self.PARconv(x_par)
        x_sst = self.SSTconv(x_sst)

        #return x_chlor.reshape(32,x.shape[1],2304) 
        if x.shape[0]==1:
            x[:, :, 0:2304] = x_chlor.reshape(1,x.shape[1],2304)
            x[:, :, 2304:4608] = x_par.reshape(1,x.shape[1],2304)
            x[:, :, 4608:6912] = x_sst.reshape(1,x.shape[1],2304)
        else:
            x[:, :, 0:2304] = x_chlor.reshape(32,x.shape[1],2304)
            x[:, :, 2304:4608] = x_par.reshape(32,x.shape[1],2304)
            x[:, :, 4608:6912] = x_sst.reshape(32,x.shape[1],2304)
        return x

class Conv2(nn.Module):
    def __init__(self, in_c=1, out_c=1, padding=2):
        super(Conv2, self).__init__()
        self.CHconv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=9, stride=1, padding=4, padding_mode="replicate"),
            nn.ReLU(),
            nn.Conv2d(in_c, out_c, kernel_size=7, stride=1, padding=3, padding_mode="replicate"),
            nn.ReLU(),
        )

    def forward(self, x):

        if x.shape[0]==1:
            '''prediction'''
            x_chlor = x[:, :, 0:2304].reshape(x.shape[1],1,48,48)
        else:
            x_chlor = x[:, :, 0:2304].reshape(32*x.shape[1],1,48,48)

        x_chlor = self.CHconv(x_chlor)

        if x.shape[0]==1:
            x[:, :, 0:2304] = x_chlor.reshape(1,x.shape[1],2304)
        else:
            x[:, :, 0:2304] = x_chlor.reshape(32,x.shape[1],2304)
        return x