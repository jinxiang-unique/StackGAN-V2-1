import torch
import torch.nn as nn
from torch.autograd import Variable
import config as cfg

def KLloss(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLDelement = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLDelement).mul_(-0.5)
    return KLD

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.orthogonal_(m.weight.data, 1.0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight.data, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

############################
# GENERATOR NEURAL NETWORK #
############################

class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode, size=None):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.size = size

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, size=self.size)
        return x

# custom activation function
class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, "channels do not divide 2!"
        nc = int(nc/2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])

# Upsale the spatial size by a factor of 2
def upBlock(inn, out):
    block = nn.Sequential(
        Interpolate(scale_factor=2, mode="nearest"),
        nn.Conv2d(inn, out * 2, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out * 2),
        GLU()
    )
    return block

# Keep the spatial size
def Block3x3_relu(inn, out):
    block = nn.Sequential(
        nn.Conv2d(inn, out * 2, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out * 2),
        GLU()
    )
    return block

class ResBlock(nn.Module):
    def __init__(self, channelNum):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channelNum, channelNum * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channelNum * 2),
            GLU(),
            nn.Conv2d(channelNum, channelNum, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channelNum)
        )

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return out

# condition aguement
class CAnet(nn.Module):
    # some code is modified from vae examples
    # (https://github.com/pytorch/examples/blob/master/vae/main.py)
    def __init__(self):
        super(CAnet, self).__init__()
        self.CD = cfg.embeddingsDim # 1024, 512 : 256, 512 / 4
        self.fc = nn.Linear(cfg.textDim, self.CD * 4, bias=True)
        self.relu = GLU()

    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :self.CD]
        logvar = x[:, self.CD:]
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar

class INIT_STAGE_G(nn.Module):
    def __init__(self, ngf):
        super(INIT_STAGE_G, self).__init__()
        self.gf_dim = ngf

        self.in_dim = cfg.zDim + cfg.embeddingsDim

        self.define_module()

    def define_module(self):
        in_dim = self.in_dim
        ngf = self.gf_dim
        self.fc = nn.Sequential(
            nn.Linear(in_dim, ngf * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4 * 2),
            GLU())

        self.upsample1 = upBlock(ngf, ngf // 2)
        self.upsample2 = upBlock(ngf // 2, ngf // 4)
        self.upsample3 = upBlock(ngf // 4, ngf // 8)
        self.upsample4 = upBlock(ngf // 8, ngf // 16)

    def forward(self, z_code, c_code):

        in_code = torch.cat((c_code, z_code), 1)

        # state size 16ngf x 4 x 4
        out_code = self.fc(in_code)
        out_code = out_code.view(-1, self.gf_dim, 4, 4)
        # state size 8ngf x 8 x 8
        out_code = self.upsample1(out_code)
        # state size 4ngf x 16 x 16
        out_code = self.upsample2(out_code)
        # state size 2ngf x 32 x 32
        out_code = self.upsample3(out_code)
        # state size ngf x 64 x 64
        out_code = self.upsample4(out_code)

        return out_code

class NEXT_STAGE_G(nn.Module):
    def __init__(self, ngf, num_residual = 2):
        super(NEXT_STAGE_G, self).__init__()
        self.gf_dim = ngf

        self.ef_dim = cfg.embeddingsDim
        self.num_residual = num_residual
        self.define_module()

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(self.num_residual):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def define_module(self):
        ngf = self.gf_dim
        efg = self.ef_dim

        self.jointConv = Block3x3_relu(ngf + efg, ngf)
        self.residual = self._make_layer(ResBlock, ngf)
        self.upsample = upBlock(ngf, ngf // 2)

    def forward(self, h_code, c_code):
        s_size = h_code.size(2)
        c_code = c_code.view(-1, self.ef_dim, 1, 1)
        c_code = c_code.repeat(1, 1, s_size, s_size)
        # state size (ngf+egf) x in_size x in_size
        h_c_code = torch.cat((c_code, h_code), 1)
        # state size ngf x in_size x in_size
        out_code = self.jointConv(h_c_code)
        out_code = self.residual(out_code)
        # state size ngf/2 x 2in_size x 2in_size
        out_code = self.upsample(out_code)

        return out_code

class GET_IMAGE_G(nn.Module):
    def __init__(self, ngf):
        super(GET_IMAGE_G, self).__init__()
        self.gf_dim = ngf
        self.img = nn.Sequential(
            nn.Conv2d(ngf, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh())

    def forward(self, h_code):
        out_img = self.img(h_code)
        return out_img

class G_NET(nn.Module):
    def __init__(self):
        super(G_NET, self).__init__()
        self.gf_dim = cfg.generatorDim
        self.define_module()

    def define_module(self):
        self.ca_net = CAnet()

        if cfg.StageNum > 0:
            self.h_net1 = INIT_STAGE_G(self.gf_dim * 16)
            self.img_net1 = GET_IMAGE_G(self.gf_dim)
        if cfg.StageNum > 1:
            self.h_net2 = NEXT_STAGE_G(self.gf_dim)
            self.img_net2 = GET_IMAGE_G(self.gf_dim // 2)
        if cfg.StageNum > 2:
            self.h_net3 = NEXT_STAGE_G(self.gf_dim // 2)
            self.img_net3 = GET_IMAGE_G(self.gf_dim // 4)
        if cfg.StageNum > 3:  # Recommended structure (mainly limited by GPU memory), and not test yet
            self.h_net4 = NEXT_STAGE_G(self.gf_dim // 4, num_residual=1)
            self.img_net4 = GET_IMAGE_G(self.gf_dim // 8)
        if cfg.StageNum > 4:
            self.h_net4 = NEXT_STAGE_G(self.gf_dim // 8, num_residual=1)
            self.img_net4 = GET_IMAGE_G(self.gf_dim // 16)

    def forward(self, z_code, text_embedding=None):

        c_code, mu, logvar = self.ca_net(text_embedding)

        fake_imgs = []
        if cfg.StageNum > 0:
            h_code1 = self.h_net1(z_code, c_code)
            fake_img1 = self.img_net1(h_code1)
            fake_imgs.append(fake_img1)
        if cfg.StageNum > 1:
            h_code2 = self.h_net2(h_code1, c_code)
            fake_img2 = self.img_net2(h_code2)
            fake_imgs.append(fake_img2)
        if cfg.StageNum > 2:
            h_code3 = self.h_net3(h_code2, c_code)
            fake_img3 = self.img_net3(h_code3)
            fake_imgs.append(fake_img3)
        if cfg.StageNum > 3:
            h_code4 = self.h_net4(h_code3, c_code)
            fake_img4 = self.img_net4(h_code4)
            fake_imgs.append(fake_img4)

        return fake_imgs, mu, logvar

################################
# DISCRIMINATOR NEURAL NETWORK #
################################

def Block3x3_leakRelu(inn, out):
    block = nn.Sequential(
        nn.Conv2d(inn, out, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block

# Downsale the spatial size by a factor of 2
def downBlock(inn, out):
    block = nn.Sequential(
        nn.Conv2d(inn, out, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block

# Downsale the spatial size by a factor of 16
def encode_image_by_16times(ndf = cfg.discriminatorDim):
    encode_img = nn.Sequential(
        # --> state size. ndf x in_size/2 x in_size/2
        nn.Conv2d(3, ndf, kernel_size=4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 2ndf x x in_size/4 x in_size/4
        nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 4ndf x in_size/8 x in_size/8
        nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 8ndf x in_size/16 x in_size/16
        nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(ndf * 8),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return encode_img

# For 64 x 64 images
class D_NET64(nn.Module):
    def __init__(self):
        super(D_NET64, self).__init__()
        self.df_dim = cfg.discriminatorDim
        self.ef_dim = cfg.embeddingsDim
        self.define_module()

    def define_module(self):
        ndf = self.df_dim
        efg = self.ef_dim
        self.img_code_s16 = encode_image_by_16times(ndf)

        self.logits = nn.Sequential(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4), nn.Sigmoid())

        self.jointConv = Block3x3_leakRelu(ndf * 8 + efg, ndf * 8)
        self.uncond_logits = nn.Sequential(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4), nn.Sigmoid())

    def forward(self, x_var, c_code=None):
        x_code = self.img_code_s16(x_var)

        c_code = c_code.view(-1, self.ef_dim, 1, 1)
        c_code = c_code.repeat(1, 1, 4, 4)
        # state size (ngf+egf) x 4 x 4
        h_c_code = torch.cat((c_code, x_code), 1)
        # state size ngf x in_size x in_size
        h_c_code = self.jointConv(h_c_code)

        output = self.logits(h_c_code)

        out_uncond = self.uncond_logits(x_code)
        return [output.view(-1), out_uncond.view(-1)]

# For 128 x 128 images
class D_NET128(nn.Module):
    def __init__(self):
        super(D_NET128, self).__init__()
        self.df_dim = cfg.discriminatorDim
        self.ef_dim = cfg.embeddingsDim
        self.define_module()

    def define_module(self):
        ndf = self.df_dim
        efg = self.ef_dim
        self.img_code_s16 = encode_image_by_16times(ndf)
        self.img_code_s32 = downBlock(ndf * 8, ndf * 16)
        self.img_code_s32_1 = Block3x3_leakRelu(ndf * 16, ndf * 8)

        self.logits = nn.Sequential(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4), nn.Sigmoid())

        self.jointConv = Block3x3_leakRelu(ndf * 8 + efg, ndf * 8)
        self.uncond_logits = nn.Sequential(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4), nn.Sigmoid())

    def forward(self, x_var, c_code=None):
        x_code = self.img_code_s16(x_var)
        x_code = self.img_code_s32(x_code)
        x_code = self.img_code_s32_1(x_code)

        c_code = c_code.view(-1, self.ef_dim, 1, 1)
        c_code = c_code.repeat(1, 1, 4, 4)
        # state size (ngf+egf) x 4 x 4
        h_c_code = torch.cat((c_code, x_code), 1)
        # state size ngf x in_size x in_size
        h_c_code = self.jointConv(h_c_code)

        output = self.logits(h_c_code)

        out_uncond = self.uncond_logits(x_code)
        return [output.view(-1), out_uncond.view(-1)]

# For 256 x 256 images
class D_NET256(nn.Module):
    def __init__(self):
        super(D_NET256, self).__init__()
        self.df_dim = cfg.discriminatorDim
        self.ef_dim = cfg.embeddingsDim
        self.define_module()

    def define_module(self):
        ndf = self.df_dim
        efg = self.ef_dim
        self.img_code_s16 = encode_image_by_16times(ndf)
        self.img_code_s32 = downBlock(ndf * 8, ndf * 16)
        self.img_code_s64 = downBlock(ndf * 16, ndf * 32)
        self.img_code_s64_1 = Block3x3_leakRelu(ndf * 32, ndf * 16)
        self.img_code_s64_2 = Block3x3_leakRelu(ndf * 16, ndf * 8)

        self.logits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid())

        self.jointConv = Block3x3_leakRelu(ndf * 8 + efg, ndf * 8)
        self.uncond_logits = nn.Sequential(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4), nn.Sigmoid())

    def forward(self, x_var, c_code=None):
        x_code = self.img_code_s16(x_var)
        x_code = self.img_code_s32(x_code)
        x_code = self.img_code_s64(x_code)
        x_code = self.img_code_s64_1(x_code)
        x_code = self.img_code_s64_2(x_code)

        c_code = c_code.view(-1, self.ef_dim, 1, 1)
        c_code = c_code.repeat(1, 1, 4, 4)
        # state size (ngf+egf) x 4 x 4
        h_c_code = torch.cat((c_code, x_code), 1)
        # state size ngf x in_size x in_size
        h_c_code = self.jointConv(h_c_code)

        output = self.logits(h_c_code)

        out_uncond = self.uncond_logits(x_code)
        return [output.view(-1), out_uncond.view(-1)]


# For 512 x 512 images: Recommended structure, not test yet
class D_NET512(nn.Module):
    def __init__(self):
        super(D_NET512, self).__init__()
        self.df_dim = cfg.discriminatorDim
        self.ef_dim = cfg.embeddingsDim
        self.define_module()

    def define_module(self):
        ndf = self.df_dim
        efg = self.ef_dim
        self.img_code_s16 = encode_image_by_16times(ndf)
        self.img_code_s32 = downBlock(ndf * 8, ndf * 16)
        self.img_code_s64 = downBlock(ndf * 16, ndf * 32)
        self.img_code_s128 = downBlock(ndf * 32, ndf * 64)
        self.img_code_s128_1 = Block3x3_leakRelu(ndf * 64, ndf * 32)
        self.img_code_s128_2 = Block3x3_leakRelu(ndf * 32, ndf * 16)
        self.img_code_s128_3 = Block3x3_leakRelu(ndf * 16, ndf * 8)

        self.logits = nn.Sequential(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4), nn.Sigmoid())

        self.jointConv = Block3x3_leakRelu(ndf * 8 + efg, ndf * 8)
        self.uncond_logits = nn.Sequential(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4), nn.Sigmoid())

    def forward(self, x_var, c_code=None):
        x_code = self.img_code_s16(x_var)
        x_code = self.img_code_s32(x_code)
        x_code = self.img_code_s64(x_code)
        x_code = self.img_code_s128(x_code)
        x_code = self.img_code_s128_1(x_code)
        x_code = self.img_code_s128_2(x_code)
        x_code = self.img_code_s128_3(x_code)

        c_code = c_code.view(-1, self.ef_dim, 1, 1)
        c_code = c_code.repeat(1, 1, 4, 4)
        # state size (ngf+egf) x 4 x 4
        h_c_code = torch.cat((c_code, x_code), 1)
        # state size ngf x in_size x in_size
        h_c_code = self.jointConv(h_c_code)

        output = self.logits(h_c_code)

        out_uncond = self.uncond_logits(x_code)
        return [output.view(-1), out_uncond.view(-1)]

# For 1024 x 1024 images: Recommended structure, not test yet
class D_NET1024(nn.Module):
    def __init__(self):
        super(D_NET1024, self).__init__()
        self.df_dim = cfg.discriminatorDim
        self.ef_dim = cfg.embeddingsDim
        self.define_module()

    def define_module(self):
        ndf = self.df_dim
        efg = self.ef_dim
        self.img_code_s16 = encode_image_by_16times(ndf)
        self.img_code_s32 = downBlock(ndf * 8, ndf * 16)
        self.img_code_s64 = downBlock(ndf * 16, ndf * 32)
        self.img_code_s128 = downBlock(ndf * 32, ndf * 64)
        self.img_code_s256 = downBlock(ndf * 64, ndf * 128)
        self.img_code_s256_1 = Block3x3_leakRelu(ndf * 128, ndf * 64)
        self.img_code_s256_2 = Block3x3_leakRelu(ndf * 64, ndf * 32)
        self.img_code_s256_3 = Block3x3_leakRelu(ndf * 32, ndf * 16)
        self.img_code_s256_4 = Block3x3_leakRelu(ndf * 16, ndf * 8)

        self.logits = nn.Sequential(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4), nn.Sigmoid())

        self.jointConv = Block3x3_leakRelu(ndf * 8 + efg, ndf * 8)
        self.uncond_logits = nn.Sequential(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4), nn.Sigmoid())

    def forward(self, x_var, c_code=None):
        x_code = self.img_code_s16(x_var)
        x_code = self.img_code_s32(x_code)
        x_code = self.img_code_s64(x_code)
        x_code = self.img_code_s128(x_code)
        x_code = self.img_code_s256(x_code)
        x_code = self.img_code_s256_1(x_code)
        x_code = self.img_code_s256_2(x_code)
        x_code = self.img_code_s256_3(x_code)
        x_code = self.img_code_s256_4(x_code)

        c_code = c_code.view(-1, self.ef_dim, 1, 1)
        c_code = c_code.repeat(1, 1, 4, 4)
        # state size (ngf+egf) x 4 x 4
        h_c_code = torch.cat((c_code, x_code), 1)
        # state size ngf x in_size x in_size
        h_c_code = self.jointConv(h_c_code)

        output = self.logits(h_c_code)

        out_uncond = self.uncond_logits(x_code)
        return [output.view(-1), out_uncond.view(-1)]

# evaluator Neurel network

# eval 256
class eval256(nn.Module):
    def __init__(self):
        super(eval256, self).__init__()
        self.df_dim = cfg.discriminatorDim
        self.ef_dim = cfg.embeddingsDim
        self.define_module()

    def define_module(self):
        ndf = self.df_dim
        efg = self.ef_dim
        self.img_code_s16 = encode_image_by_16times(ndf)
        self.img_code_s32 = downBlock(ndf * 8, ndf * 16)
        self.img_code_s64 = downBlock(ndf * 16, ndf * 32)
        self.img_code_s64_1 = Block3x3_leakRelu(ndf * 32, ndf * 16)
        self.img_code_s64_2 = Block3x3_leakRelu(ndf * 16, ndf * 8)

        self.logits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid())

    def forward(self, x_var, c_code=None):
        x_code = self.img_code_s16(x_var)
        x_code = self.img_code_s32(x_code)
        x_code = self.img_code_s64(x_code)
        x_code = self.img_code_s64_1(x_code)
        x_code = self.img_code_s64_2(x_code)

        h_c_code = x_code

        output = self.logits(h_c_code)

        return output.view(-1)