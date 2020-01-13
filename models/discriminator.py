import torch
from torch import nn
import torch.nn.functional as F
from models.networks import Conv2dBlock


class Discriminator(nn.Module):
    def __init__(self, input_dim, params):
        super(Discriminator, self).__init__()
        self.n_layer = params['n_layer']  # 4
        self.gan_type = params['gan_type']
        self.dim = params['dim']  # 64
        self.norm = params['norm']  # sn
        self.activ = params['activ']
        self.num_scales = params['num_scales']  # 3
        self.pad_type = params['pad_type']
        self.input_dim = input_dim
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.cnns = nn.ModuleList()
        for _ in range(self.num_scales):
            self.cnns.append(self._make_net())

        # --- for cam
        self.gap_fc = nn.utils.spectral_norm(nn.Linear(self.output_dim, 1, bias=False))
        self.gmp_fc = nn.utils.spectral_norm(nn.Linear(self.output_dim, 1, bias=False))
        self.conv1x1 = nn.Conv2d(self.output_dim * 2, self.output_dim, kernel_size=1, stride=1, bias=True)
        self.leaky_relu = nn.LeakyReLU(0.2, True)

        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.utils.spectral_norm(
            nn.Conv2d(self.output_dim, 1, kernel_size=4, stride=1, padding=0, bias=False))

    def _make_net(self):
        dim = self.dim
        cnn_x = []
        cnn_x += [Conv2dBlock(self.input_dim, dim, 4, 2, 1, norm='none', activation=self.activ, pad_type=self.pad_type)]

        for i in range(self.n_layer - 1):
            cnn_x += [Conv2dBlock(dim, dim * 2, 4, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
            dim *= 2
        # cnn_x += [nn.Conv2d(dim, 1, 1, 1, 0)]  # 512 -> 1

        cnn_x = nn.Sequential(*cnn_x)
        self.output_dim = dim  # 512
        return cnn_x

    def forward(self, x):
        outputs = []
        cam_logits = []
        heatmaps = []
        for model in self.cnns:
            out = model(x)  # [1, 512, 16, 16]

            # auxiliary classifier (same)
            gap = F.adaptive_avg_pool2d(out, 1)
            gmp = F.adaptive_max_pool2d(out, 1)
            gap_logit = self.gap_fc(gap.view(out.shape[0], -1))
            gmp_logit = self.gmp_fc(gmp.view(out.shape[0], -1))
            cam_logit = torch.cat([gap_logit, gmp_logit], 1)

            gap_weight = list(self.gap_fc.parameters())[0]
            gmp_weight = list(self.gmp_fc.parameters())[0]
            gap = out * gap_weight.unsqueeze(2).unsqueeze(3)
            gmp = out * gmp_weight.unsqueeze(2).unsqueeze(3)
            out = torch.cat([gap, gmp], 1)
            out = self.leaky_relu(self.conv1x1(out))

            heatmap = torch.sum(out, dim=1, keepdim=True)  # _
            heatmaps.append(heatmap)

            # classifier: 1 conv + 1 sn
            out = self.pad(out)
            out = self.conv(out)

            outputs.append(out)
            cam_logits.append(cam_logit)
            x = self.downsample(x)
        return outputs, cam_logits

    def calc_dis_loss(self, input_fake, input_real):
        # calculate the loss to train D
        fake_outs, fake_cam_logits = self.forward(input_fake)
        real_outs, real_cam_logits = self.forward(input_real)
        loss = 0

        for it, (fake_out, fake_cam_logit,  real_out, real_cam_logit) in enumerate(
                zip(fake_outs, fake_cam_logits, real_outs, real_cam_logits)):
            if self.gan_type == 'lsgan':
                loss += torch.mean((fake_out - 0)**2) + torch.mean((real_out - 1)**2)
                # cam
                loss += torch.mean((fake_cam_logit - 0)**2) + torch.mean((real_cam_logit - 1)**2)
            elif self.gan_type == 'nsgan':
                all0 = torch.zeros_like(fake_out.data, requires_grad=False).cuda()
                all1 = torch.ones_like(real_out.data, requires_grad=False).cuda()
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(fake_out), all0) +
                                   F.binary_cross_entropy(F.sigmoid(real_out), all1))
                # cam
                loss += torch.mean((fake_cam_logit - 0) ** 2) + torch.mean((real_cam_logit - 1) ** 2)
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

    def calc_gen_loss(self, input_fake):
        # calculate the loss to train G
        fake_outs, fake_cam_logits = self.forward(input_fake)
        loss = 0
        for it, (fake_out, fake_cam_logit) in enumerate(zip(fake_outs, fake_cam_logits)):
            if self.gan_type == 'lsgan':
                loss += torch.mean((fake_out - 1)**2) + torch.mean((fake_cam_logit - 1)**2)
            elif self.gan_type == 'nsgan':
                all1 = torch.ones_like(fake_out.data, requires_grad=False).cuda()
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(fake_out), all1))
                loss += torch.mean((fake_cam_logit - 1)**2)
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

