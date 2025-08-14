import torch
import torch.nn as nn
import numpy as np
from torch.nn.modules import dropout
import scipy.io as sio
from torch.nn.modules.activation import LeakyReLU
import torchvision
import argparse
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch.nn.functional as F
import torchvision.transforms as transforms
# from torchstat import stat
import transformer

import time

from utils import LPA

start = time.time()

# EPOCH = 800

# alpha = 0.2
# beta = 0.01
# drop_out = 0.1
# learning_rate = 0.001

seed = 1
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# Load DATA
data = sio.loadmat("samson_dataset.mat")

abundance_GT = torch.from_numpy(data["A"])  # true abundance
original_HSI = torch.from_numpy(data["Y"])  # mixed abundance

# VCA_endmember and GT
VCA_endmember = data["M1"]
GT_endmember = data["M"]

endmember_init = torch.from_numpy(VCA_endmember).unsqueeze(2).unsqueeze(3).float()
GT_init = torch.from_numpy(GT_endmember).unsqueeze(2).unsqueeze(3).float()

band_Number = original_HSI.shape[0]
endmember_number, pixel_number = abundance_GT.shape

col = 95

original_HSI = torch.reshape(original_HSI, (band_Number, col, col))
abundance_GT = torch.reshape(abundance_GT, (endmember_number, col, col))

batch_size = 1
EPOCH = 800

alpha = 0.1
beta = 0.03
drop_out = 0.1
learning_rate = 0.0223





class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(4,1,1,1)
        if torch.cuda.is_available():
            self.kernel = self.kernel#.cuda()
        self.loss = CharbonnierLoss()
        self.downsampling22 = nn.AvgPool2d(2, 2,ceil_mode =True)

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)    # filter
        down        = filtered[:,:,::2,::2]               # downsample
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4                  # upsample
        filtered    = self.conv_gauss(new_filter) # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        x =self.downsampling22(x)
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss







# abundance normalization
def norm_abundance_GT(abundance_input, abundance_GT_input):
    abundance_input = abundance_input / (torch.sum(abundance_input, dim=1))
    abundance_input = torch.reshape(
        abundance_input.squeeze(0), (endmember_number, col, col)
    )
    abundance_input = abundance_input.cpu().detach().numpy()
    abundance_GT_input = abundance_GT_input / (torch.sum(abundance_GT_input, dim=0))
    abundance_GT_input = abundance_GT_input.cpu().detach().numpy()
    return abundance_input, abundance_GT_input


# endmember normalization
def norm_endmember(endmember_input, endmember_GT):
    for i in range(0, endmember_number):
        endmember_input[:, i] = endmember_input[:, i] / np.max(endmember_input[:, i])
        endmember_GT[:, i] = endmember_GT[:, i] / np.max(endmember_GT[:, i])
    return endmember_input, endmember_GT


# plot abundance
def plot_abundance(abundance_input, abundance_GT_input):
    for i in range(0, endmember_number):

        plt.subplot(2, endmember_number, i + 1)
        plt.imshow(abundance_input[i, :, :], cmap="jet")

        plt.subplot(2, endmember_number, endmember_number + i + 1)
        plt.imshow(abundance_GT_input[i, :, :], cmap="jet")
    plt.show()


# plot endmember
def plot_endmember(endmember_input, endmember_GT):
    plt.figure(figsize=(8, 8), dpi=150)
    for i in range(0, endmember_number):
        plt.subplot(2, endmember_number // 2 if endmember_number % 2 == 0 else endmember_number, i + 1)
        plt.plot(endmember_input[:, i], label="Extracted")
        plt.plot(endmember_GT[:, i], label="GT")
        plt.legend(loc="upper left")
    plt.tight_layout()

    plt.show()


# change the index of abundance and endmember
def arange_A_E(abundance_input, abundance_GT_input, endmember_input, endmember_GT):
    RMSE_matrix = np.zeros((endmember_number, endmember_number))
    SAD_matrix = np.zeros((endmember_number, endmember_number))
    RMSE_index = np.zeros(endmember_number).astype(int)
    SAD_index = np.zeros(endmember_number).astype(int)
    RMSE_abundance = np.zeros(endmember_number)
    SAD_endmember = np.zeros(endmember_number)

    for i in range(0, endmember_number):
        for j in range(0, endmember_number):
            RMSE_matrix[i, j] = AbundanceRmse(
                abundance_input[i, :, :], abundance_GT_input[j, :, :]
            )
            SAD_matrix[i, j] = SAD_distance(endmember_input[:, i], endmember_GT[:, j])

        RMSE_index[i] = np.argmin(RMSE_matrix[i, :])
        SAD_index[i] = np.argmin(SAD_matrix[i, :])
        RMSE_abundance[i] = np.min(RMSE_matrix[i, :])
        SAD_endmember[i] = np.min(SAD_matrix[i, :])

    abundance_input[np.arange(endmember_number), :, :] = abundance_input[
        RMSE_index, :, :
    ]
    endmember_input[:, np.arange(endmember_number)] = endmember_input[:, SAD_index]

    return abundance_input, endmember_input, RMSE_abundance, SAD_endmember


class load_data(torch.utils.data.Dataset):
    def __init__(self, img, gt, transform=None):
        self.img = img.float()
        self.gt = gt.float()
        self.transform = transform

    def __getitem__(self, idx):
        return self.img, self.gt

    def __len__(self):
        return 1


# calculate RMSE of abundance
def AbundanceRmse(inputsrc, inputref):
    rmse = np.sqrt(((inputsrc - inputref) ** 2).mean())
    return rmse


def conv33(inchannel, outchannel):
    return nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=1)


def conv11(inchannel, outchannel):
    return nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=1)


def transconv11(inchannel,outchannel):
    return nn.Conv2d(inchannel, outchannel, kernel_size=1,stride=1)


# calculate SAD of endmember
def SAD_distance(src, ref):
    cos_sim = np.dot(src, ref) / (np.linalg.norm(src) * np.linalg.norm(ref))
    SAD_sim = np.arccos(cos_sim)
    return SAD_sim


class dlk(nn.Module):
    def __init__(self, dim, drop_path, drop_path_rate=0.1):
        super(dlk, self).__init__()

        self.base_layers1 = nn.Sequential(
            DLKBlock(dim=96),
        )
    def forward(self, x):
        # x = x.unsqueeze(1).unsqueeze(2)

        # x = torch.reshape(x, (512,-1, 162))
        x = self.base_layers1(x)
        return x







class DynamicFusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, 6, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(6, 3, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, code1, code2):
        # trans_feat: (B,C,H,W) 来自Transformer
        # lstm_feat: (B,C,H,W) 来自LSTM
        combined = torch.cat([code1, code2], dim=1)
        # print('combined', combined.shape)
        gate_weights = self.gate(combined)  # (B,2,H,W)
        # print('gate_weights', gate_weights.shape)
        return gate_weights[:, 0:1] * code1 + gate_weights[:, 1:2] * code2



class Denoise(nn.Module):
    def __init__(self, P, L, size, patch, dim, drop_path_rate=0.1,channel=1, k_size=3):
        super(Denoise, self).__init__()
        self.P, self.L, self.col, self.size, self.patch, self.dim = 3, 156, 95, 95, 5, 200
        # self.channel1 = dlk(dim=dim,drop_path=0.2)
        # self.channel2 = eca()
        self.vtrans = transformer.ViT(image_size=size, patch_size=patch, dim=dim*P, depth=2,
                                      heads=8, mlp_dim=12, pool='cls')
        self.upscale = nn.Sequential(
            nn.Linear(dim, size ** 2),
        )

        self.smooth = nn.Sequential(
            nn.Conv2d(P, P, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Softmax(dim=1),
        )
        self.channel2 = LPA(in_channel=24, out_channel=3)
        self.fusion = DynamicFusion(6)  # 输入通道数为Trans+LSTM的总通道数
        # self.channel3 = LSTM(input_dim=L, hidden_dim=16, num_layers=2, output_dim=32, col=col)

    def forward(self, x):
        code1 = self.vtrans(x)#1,3200
        cls_emb = code1.view(1, self.P, -1)#1,4,800
        abu_est = self.upscale(cls_emb).view(1, self.P, self.size, self.size)
        code1 = self.smooth(abu_est)
        # print('code1', code1.shape)
        # code2 = self.channel2(x)
        code2 = self.channel2(x)

        fused_code = self.fusion(code1, code2)
        return fused_code





# my net
class multiStageUnmixing(nn.Module):
    def __init__(self):
        super(multiStageUnmixing, self).__init__()
        self.layer1 = nn.Sequential(
            conv33(band_Number,24),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(24),
            nn.Dropout(drop_out),
            Denoise(P=3, L=156, size=95,
                 patch=5, dim=200),
            conv33(3, 48),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(48),
            nn.Dropout(drop_out),
            conv33(48, endmember_number),

        )



        self.downsampling22 = nn.AvgPool2d(2, 2,ceil_mode =True)
        self.downsampling44 = nn.AvgPool2d(4, 4,ceil_mode =True)


        self.encodelayer = nn.Sequential(nn.Softmax())
        self.transconv = transconv11(endmember_number,endmember_number)
        self.transconv2 = transconv11(band_Number, band_Number)
        self.decoderlayer4 = nn.Sequential(
            nn.Conv2d(
                in_channels=endmember_number,
                out_channels=band_Number,
                kernel_size=(1, 1),
                bias=False,
            ),
        )


    def forward(self, x):

        convlayer1 = self.transconv2(x)
        # print(convlayer1.shape)
        layer1out = self.layer1(convlayer1)


        en_result1 = self.encodelayer(layer1out)



        de_result1 = self.decoderlayer4(en_result1)

        return en_result1, de_result1


# SAD loss of reconstruction
def reconstruction_SADloss(output, target):

    _, band, h, w = output.shape
    output = torch.reshape(output, (band, h * w))
    target = torch.reshape(target, (band, h * w))
    abundance_loss = torch.acos(torch.cosine_similarity(output, target, dim=0))
    abundance_loss = torch.mean(abundance_loss)

    return abundance_loss

MSE = torch.nn.MSELoss(size_average=True)

# weights_init
def weights_init(m):
    # nn.init.kaiming_normal_(m.weight.data)
    nn.init.kaiming_normal_(net.layer1[0].weight.data)
    nn.init.kaiming_normal_(net.layer1[5].weight.data)
    nn.init.kaiming_normal_(net.layer1[9].weight.data)



# load data
train_dataset = load_data(
    img=original_HSI, gt=abundance_GT, transform=transforms.ToTensor()
)
# Data loader
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=False
)

net = multiStageUnmixing()#.cuda()
# net = multiStageUnmixing()
edgeLoss = EdgeLoss()
# weight init
net.apply(weights_init)

# decoder weight init by VCA
model_dict = net.state_dict()
model_dict["decoderlayer4.0.weight"] = endmember_init


net.load_state_dict(model_dict)

# optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= EPOCH // 30 , gamma=0.8)

# train
for epoch in range(EPOCH):
    for i, (x, y) in enumerate(train_loader):
        scheduler.step()
        x = x#.cuda()
        net.train()#.cuda()

        en_abundance, reconstruction_result = net(x)

        abundanceLoss = reconstruction_SADloss(x, reconstruction_result)

        MSELoss = MSE(x, reconstruction_result)

        ALoss = abundanceLoss
        BLoss = MSELoss


        total_loss = ALoss + (alpha * BLoss)

        optimizer.zero_grad()

        total_loss.backward()
        optimizer.step()


        if epoch % 100 == 0:
            print(
                "Epoch:",
                epoch,
                "| loss: %.4f" % total_loss.cpu().data.numpy(),
            )



net.eval()


en_abundance, reconstruction_result = net(x)

decoder_para = net.state_dict()["decoderlayer4.0.weight"].cpu().numpy()
decoder_para = np.mean(np.mean(decoder_para, -1), -1)

en_abundance, abundance_GT = norm_abundance_GT(en_abundance, abundance_GT)
decoder_para, GT_endmember = norm_endmember(decoder_para, GT_endmember)

en_abundance, decoder_para, RMSE_abundance, SAD_endmember = arange_A_E(
    en_abundance, abundance_GT, decoder_para, GT_endmember
)
print("RMSE", RMSE_abundance)
print("mean_RMSE", RMSE_abundance.mean())
print("endmember_SAD", SAD_endmember)
print("mean_SAD", SAD_endmember.mean())
end = time.time()
print(end-start)

plot_abundance(en_abundance, abundance_GT)
plot_endmember(decoder_para, GT_endmember)

from fvcore.nn import FlopCountAnalysis

model = multiStageUnmixing()  # 或.cpu()

# 示例输入尺寸 (根据实际修改)
input_tensor = torch.randn(1, 156, 95, 95)  # [batch, bands, height, width]

# 计算FLOPs
flops = FlopCountAnalysis(model, input_tensor)
total_flops = flops.total()
print(f"Total FLOPs: {total_flops / 1e9:.2f}G")