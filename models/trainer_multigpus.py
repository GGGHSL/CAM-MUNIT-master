from models.generator import Generator
from models.discriminator import Discriminator
from utils import weights_init, get_model_list, get_scheduler, get_transform, get_mean_image
import torch
import torch.nn as nn


class Trainer(nn.Module):
    def __init__(self, hyperparameters, main_device=None):
        super(Trainer, self).__init__()
        self.hyperparameters = hyperparameters
        self.main_device = main_device

        # Initiate the networks
        self.gen_a = Generator(hyperparameters['input_dim_a'], hyperparameters['gen'])  # auto-encoder for domain a
        self.gen_b = Generator(hyperparameters['input_dim_b'], hyperparameters['gen'])  # auto-encoder for domain b
        self.dis_a = Discriminator(hyperparameters['input_dim_a'], hyperparameters['dis'])  # discriminator for domain a
        self.dis_b = Discriminator(hyperparameters['input_dim_b'], hyperparameters['dis'])  # discriminator for domain b
        self.style_dim = hyperparameters['gen']['style_dim']

        # fix the noise used in sampling
        display_size = int(hyperparameters['display_size'])
        self.s_a = torch.randn(display_size, self.style_dim, 1, 1).cuda()
        self.s_b = torch.randn(display_size, self.style_dim, 1, 1).cuda()

        # Network weight initialization
        self.apply(weights_init(hyperparameters['init']))
        self.dis_a.apply(weights_init('gaussian'))
        self.dis_b.apply(weights_init('gaussian'))

    def forward(self, x_a, x_b):
        self.eval()

        s_a = torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda()
        s_b = torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda()

        # encode
        (c_a, c_a_logit), s_a_prime = self.gen_a.encode(x_a)
        (c_b, c_b_logit), s_b_prime = self.gen_b.encode(x_b)

        # decode (within domain)
        x_a_recon = self.gen_a.decode(c_a, s_a_prime)  #
        x_b_recon = self.gen_b.decode(c_b, s_b_prime)  #

        # decode (cross domain)
        x_ba = self.gen_a.decode(c_b, s_a)
        x_ab = self.gen_b.decode(c_a, s_b)

        (c_ba, _), s_ba = self.gen_a.encode(x_ba)
        (c_ab, _), s_ab = self.gen_b.encode(x_ab)

        x_aba = self.gen_a.decode(c_ab, s_a)
        x_bab = self.gen_b.decode(c_ba, s_b)

        (_, fake_c_a_logit), s_a_prime = self.gen_a.encode(x_b)
        (_, fake_c_b_logit), s_b_prime = self.gen_b.encode(x_a)

        fake_ba_outs_a, fake_ba_cam_logits_a = self.dis_a(x_ba)
        fake_ba_outs_b, fake_ba_cam_logits_b = self.dis_b(x_ba)
        # fake_ba_outs = fake_ba_outs_a + fake_ba_outs_b  # list
        # fake_ba_cam_logits = fake_ba_cam_logits_a + fake_ba_cam_logits_b
        real_a_outs, real_a_cam_logits = self.dis_a(x_a)

        fake_ab_outs_b, fake_ab_cam_logits_b = self.dis_b(x_ab)
        fake_ab_outs_a, fake_ab_cam_logits_a = self.dis_a(x_ab)
        # fake_ab_outs = fake_ab_outs_b + fake_ab_outs_a
        # fake_ab_cam_logits = fake_ab_cam_logits_b + fake_ab_cam_logits_a
        real_b_outs, real_b_cam_logits = self.dis_b(x_b)

        self.train()
        return x_ab, x_ba, (
            s_a, s_b,
            c_a, c_a_logit, s_a_prime,
            c_b, c_b_logit, s_b_prime,
            x_a_recon, x_b_recon,
            c_ba, s_ba, c_ab, s_ab, x_aba, x_bab,
            fake_c_a_logit, fake_c_b_logit), (
            fake_ba_outs_a, fake_ba_cam_logits_a, fake_ba_outs_b, fake_ba_cam_logits_b, real_a_outs, real_a_cam_logits,
            fake_ab_outs_b, fake_ab_cam_logits_b, fake_ab_outs_a, fake_ab_cam_logits_a, real_b_outs, real_b_cam_logits)

    # def sample(self, x_a, x_b):
    #     self.eval()
    #     # with style init: batch size = 1
    #     s_a1 = torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda()
    #     s_b1 = torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda()
    #     # random
    #     s_a2 = torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda()
    #     s_b2 = torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda()
    #
    #     x_a_recon, x_b_recon, x_ba1, x_ba2, x_ab1, x_ab2 = [], [], [], [], [], []
    #
    #     for i in range(x_a.size(0)):
    #         (c_a, _), s_a_fake = self.gen_a.encode(x_a[i].unsqueeze(0))
    #         (c_b, _), s_b_fake = self.gen_b.encode(x_b[i].unsqueeze(0))
    #
    #         x_a_recon.append(self.gen_a.decode(c_a, s_a_fake))
    #         x_b_recon.append(self.gen_b.decode(c_b, s_b_fake))
    #
    #         x_ba1.append(self.gen_a.decode(c_b, s_a1[i].unsqueeze(0)))  #
    #         x_ba2.append(self.gen_a.decode(c_b, s_a2[i].unsqueeze(0)))
    #
    #         x_ab1.append(self.gen_b.decode(c_a, s_b1[i].unsqueeze(0)))  #
    #         x_ab2.append(self.gen_b.decode(c_a, s_b2[i].unsqueeze(0)))
    #
    #     x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
    #     x_ba1, x_ba2 = torch.cat(x_ba1), torch.cat(x_ba2)
    #     x_ab1, x_ab2 = torch.cat(x_ab1), torch.cat(x_ab2)
    #     self.train()
    #
    #     return x_a, x_a_recon, x_ab1, x_ab2, x_b, x_b_recon, x_ba1, x_ba2
    def sample(self, x_a, x_b):
        self.eval()
        # random
        s_a = torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda(self.main_device)
        s_b = torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda(self.main_device)
        s_a2 = torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda(self.main_device)
        s_b2 = torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda(self.main_device)

        x_a_recon, x_b_recon, x_aba, x_bab = [], [], [], []
        x_ba, x_ab = [], []
        x_ba2, x_ab2 = [], []

        for i in range(x_a.size(0)):
            (c_a, _), s_a_prime = self.gen_a.encode(x_a[i].unsqueeze(0))
            (c_b, _), s_b_prime = self.gen_b.encode(x_b[i].unsqueeze(0))

            # rec
            x_a_recon.append(self.gen_a.decode(c_a, s_a_prime))
            x_b_recon.append(self.gen_b.decode(c_b, s_b_prime))

            # generate
            ba = self.gen_a.decode(c_b, s_a[i].unsqueeze(0))
            ba2 = self.gen_a.decode(c_b, s_a2[i].unsqueeze(0))
            x_ba.append(ba)  #
            x_ba2.append(ba2)

            ab = self.gen_b.decode(c_a, s_b[i].unsqueeze(0))
            ab2 = self.gen_b.decode(c_a, s_b2[i].unsqueeze(0))
            x_ab.append(ab)  #
            x_ab2.append(ab2)

            (c_b_recon, _), s_a_recon = self.gen_a.encode(ba)
            (c_a_recon, _), s_b_recon = self.gen_b.encode(ab)
            aba = self.gen_a.decode(c_a_recon, s_a_prime)
            bab = self.gen_b.decode(c_b_recon, s_b_prime)
            x_aba.append(aba)
            x_bab.append(bab)

        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba, x_ab = torch.cat(x_ba), torch.cat(x_ab)
        # x_ba2, x_ab2 = torch.cat(x_ba2), torch.cat(x_ab2)
        x_aba, x_bab = torch.cat(x_aba), torch.cat(x_bab)
        self.train()

        # return x_a, x_a_recon, x_ab, x_ab2, x_b, x_b_recon, x_ba, x_ba2
        return x_a, x_a_recon, x_ab, x_aba, x_ab2, x_b, x_b_recon, x_ba, x_bab, x_ba2
