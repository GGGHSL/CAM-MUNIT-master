from models.generator import Generator
from models.discriminator import Discriminator
from models.backbones import VGG19
from utils import weights_init, get_model_list, get_scheduler, get_transform, get_mean_image, load_vgg16
import torch
import torch.nn as nn
# import numpy as np
import os


# from PIL import Image


class Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(Trainer, self).__init__()
        self.hyperparameters = hyperparameters
        lr = hyperparameters['lr']
        # Initiate the networks
        self.gen_a = Generator(hyperparameters['input_dim_a'], hyperparameters['gen'])  # auto-encoder for domain a
        self.gen_b = Generator(hyperparameters['input_dim_b'], hyperparameters['gen'])  # auto-encoder for domain b
        self.dis_a = Discriminator(hyperparameters['input_dim_a'], hyperparameters['dis'])  # discriminator for domain a
        self.dis_b = Discriminator(hyperparameters['input_dim_b'], hyperparameters['dis'])  # discriminator for domain b

        self.style_dim = hyperparameters['gen']['style_dim']

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
        gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()

        # fix the noise used in sampling
        display_size = int(hyperparameters['display_size'])
        self.s_a = torch.randn(display_size, self.style_dim, 1, 1).cuda()
        self.s_b = torch.randn(display_size, self.style_dim, 1, 1).cuda()

        # Load VGG16 model
        # self.vgg16 = load_vgg16(hyperparameters['vgg_model_path'])
        # self.vgg16.eval()
        # for param in self.vgg16.parameters():
        #     param.requires_grad = False

        # Load VGG19 model
        self.vgg19 = VGG19(init_weights=hyperparameters['vgg_model_path'] + 'vgg19.pth', feature_mode=True)
        self.vgg19.eval()

        # self.instance_norm = nn.InstanceNorm2d(512, affine=False)
        # self.mean_a, self.mean_b = self.get_mean_tensor(hyperparameters)

        # Network weight initialization
        self.apply(weights_init(hyperparameters['init']))
        self.dis_a.apply(weights_init('gaussian'))
        self.dis_b.apply(weights_init('gaussian'))

    # def network_init(self, hyperparameters, dis_only=False):
    # if not dis_only:
    # self.apply(weights_init(hyperparameters['init']))
    # self.dis_a.apply(weights_init('gaussian'))
    # self.dis_b.apply(weights_init('gaussian'))

    def forward(self, x_a, x_b):
        self.eval()
        s_a = self.s_a
        s_b = self.s_b
        (c_a, _), s_a_fake = self.gen_a.encode(x_a)
        (c_b, _), s_b_fake = self.gen_b.encode(x_b)
        x_ba = self.gen_a.decode(c_b, s_a)
        x_ab = self.gen_b.decode(c_a, s_b)
        self.train()
        return x_ab, x_ba

    """
    # 1. Pre-train Generator:
    """

    # def get_mean_tensor(self, hyperparameters):
    #     # numpy array
    #     np_a = get_mean_image(os.path.join(hyperparameters['data_root'], 'trainA'))
    #     np_b = get_mean_image(os.path.join(hyperparameters['data_root'], 'trainB'))
    #     # PIL image
    #     img_a = Image.fromarray(np_a.astype(np.uint8)).convert('RGB')
    #     img_b = Image.fromarray(np_b.astype(np.uint8)).convert('RGB')
    #     # torch tensor
    #     transform = get_transform(train=True, new_size=256, height=256, width=256, crop=True)
    #     tensor_a = transform(img_a).unsqueeze(0).cuda()  # [1, 3, 256, 256]
    #     tensor_b = transform(img_b).unsqueeze(0).cuda()
    #     return tensor_a, tensor_b

    # def calc_content_elastic_loss(self, vgg, cent_img, cent_tar, alpha=0.2):
    #     """
    #     Calculate content loss to pre-train Generator
    #     :param cent_img: centralized generated img
    #     :param cent_tar: centralized real img
    #     :param alpha: l2 / l1 ratio
    #     :return: loss = l1 + alpha * l2
    #     """
    #     # 1. centralization
    #     # 2. extract features
    #     img_feat = vgg(cent_img)
    #     tar_feat = vgg(cent_tar)
    #     # 3. calculate loss
    #     diff = img_feat - tar_feat
    #     loss = torch.mean(torch.abs(diff))  # l1 loss
    #     if alpha > 0:
    #         loss += alpha * torch.mean(diff ** 2)  # l2 loss
    #     return loss

    def gen_pre_train_update(self, x_a, x_b, hyperparameters):
        s_a = torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda()
        s_b = torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda()
        a_feat = self.vgg19((x_a + 1) / 2)
        b_feat = self.vgg19((x_b + 1) / 2)

        self.gen_opt.zero_grad()
        # encode
        (c_a, c_a_logit), s_a_prime = self.gen_a.encode(x_a)
        (c_b, c_b_logit), s_b_prime = self.gen_b.encode(x_b)
        # decode (cross domain)
        x_ba = self.gen_a.decode(c_b, s_a)
        x_ab = self.gen_b.decode(c_a, s_b)

        ba_feat = self.vgg19((x_ba + 1) / 2)
        ab_feat = self.vgg19((x_ab + 1) / 2)

        # content loss
        diff_a = a_feat - ba_feat
        diff_b = b_feat - ab_feat

        alpha = hyperparameters['alpha']
        loss_gen_c_a = torch.mean(torch.abs(diff_a)) + alpha * torch.mean(diff_a ** 2)
        loss_gen_c_b = torch.mean(torch.abs(diff_b)) + alpha * torch.mean(diff_b ** 2)
        self.loss_gen_c = hyperparameters['vgg_w'] * (loss_gen_c_a + loss_gen_c_b)

        self.loss_gen_c.backward()
        self.gen_opt.step()

    """
    # 2. Main Training
    """

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def dis_update(self, x_a, x_b, hyperparameters):
        s_a = torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda()
        s_b = torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda()

        self.dis_opt.zero_grad()
        # encode
        (c_a, _), _ = self.gen_a.encode(x_a)
        (c_b, _), _ = self.gen_b.encode(x_b)

        # decode (cross domain)
        x_ba = self.gen_a.decode(c_b, s_a)
        x_ab = self.gen_b.decode(c_a, s_b)

        # loss
        self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
        self.loss_dis_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b)
        self.loss_dis_total = hyperparameters['adv_w'] * (self.loss_dis_a + self.loss_dis_b)

        self.loss_dis_total.backward()
        self.dis_opt.step()

    def gen_update(self, x_a, x_b, hyperparameters):
        s_a = torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda()
        s_b = torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda()

        self.gen_opt.zero_grad()
        # encode
        (c_a, c_a_logit), s_a_prime = self.gen_a.encode(x_a)
        (c_b, c_b_logit), s_b_prime = self.gen_b.encode(x_b)

        # decode (within domain)
        x_a_recon = self.gen_a.decode(c_a, s_a_prime)
        x_b_recon = self.gen_b.decode(c_b, s_b_prime)

        # decode (cross domain)
        x_ba = self.gen_a.decode(c_b, s_a)
        x_ab = self.gen_b.decode(c_a, s_b)

        # loss
        # --- 1. adv loss
        loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
        loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)
        self.loss_gen_adv = hyperparameters['adv_w'] * (loss_gen_adv_a + loss_gen_adv_b)

        # --- 2. reconstruction loss
        (c_b_recon, _), s_a_recon = self.gen_a.encode(x_ba)
        loss_gen_recon_c_b = self.recon_criterion(c_b_recon, c_b)
        loss_gen_recon_s_a = self.recon_criterion(s_a_recon, s_a)

        (c_a_recon, _), s_b_recon = self.gen_b.encode(x_ab)
        loss_gen_recon_c_a = self.recon_criterion(c_a_recon, c_a)
        loss_gen_recon_s_b = self.recon_criterion(s_b_recon, s_b)

        loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)

        self.loss_gen_recon_x = hyperparameters['rec_w'] * (loss_gen_recon_x_a + loss_gen_recon_x_b)
        self.loss_gen_recon_c = hyperparameters['rec_c_w'] * (loss_gen_recon_c_a + loss_gen_recon_c_b)
        self.loss_gen_recon_s = hyperparameters['rec_s_w'] * (loss_gen_recon_s_a + loss_gen_recon_s_b)

        # --- 3. cyc loss
        # x_aba = self.gen_a.decode(c_a_recon, s_a) if hyperparameters['cyc_w'] > 0 else None
        # x_bab = self.gen_b.decode(c_b_recon, s_b) if hyperparameters['cyc_w'] > 0 else None
        x_aba = self.gen_a.decode(c_a_recon, s_a_prime) if hyperparameters['cyc_w'] > 0 else None
        x_bab = self.gen_b.decode(c_b_recon, s_b_prime) if hyperparameters['cyc_w'] > 0 else None
        loss_gen_cyc_x_a = self.recon_criterion(x_aba, x_a) if hyperparameters['cyc_w'] > 0 else 0
        loss_gen_cyc_x_b = self.recon_criterion(x_bab, x_b) if hyperparameters['cyc_w'] > 0 else 0
        self.loss_gen_cyc = hyperparameters['cyc_w'] * (loss_gen_cyc_x_a + loss_gen_cyc_x_b) if hyperparameters[
                                                                                                    'cyc_w'] > 0 else 0

        # --- 4. domain-invariant perceptual loss
        self.loss_gen_vgg = 0
        if hyperparameters['vgg_w'] > 0:
            a_feat = self.vgg19((x_a + 1) / 2)
            b_feat = self.vgg19((x_b + 1) / 2)

            ba_feat = self.vgg19((x_ba + 1) / 2)
            ab_feat = self.vgg19((x_ab + 1) / 2)

            diff_a = a_feat - ab_feat  #
            diff_b = b_feat - ba_feat

            alpha = hyperparameters['alpha']
            loss_gen_vgg_a = torch.mean(torch.abs(diff_a)) + alpha * torch.mean(diff_a ** 2)
            loss_gen_vgg_b = torch.mean(torch.abs(diff_b)) + alpha * torch.mean(diff_b ** 2)
            self.loss_gen_vgg = hyperparameters['vgg_w'] * (loss_gen_vgg_a + loss_gen_vgg_b)

        # --- 5. cam loss
        (_, fake_c_a_logit), s_a_prime = self.gen_a.encode(x_b)
        (_, fake_c_b_logit), s_b_prime = self.gen_b.encode(x_a)
        loss_gen_cam_a = self.bce_loss(c_a_logit, torch.ones_like(c_a_logit, requires_grad=False).cuda()) + \
                         self.bce_loss(fake_c_a_logit, torch.zeros_like(fake_c_a_logit, requires_grad=False).cuda())
        loss_gen_cam_b = self.bce_loss(c_b_logit, torch.ones_like(c_b_logit, requires_grad=False).cuda()) + \
                         self.bce_loss(fake_c_b_logit, torch.zeros_like(fake_c_b_logit, requires_grad=False).cuda())
        self.loss_gen_cam = hyperparameters['cam_w'] * (loss_gen_cam_a + loss_gen_cam_b)

        # total loss
        self.loss_gen_total = self.loss_gen_adv + \
                              self.loss_gen_recon_x + \
                              self.loss_gen_recon_s + \
                              self.loss_gen_recon_c + \
                              self.loss_gen_cyc + \
                              self.loss_gen_cam + \
                              self.loss_gen_vgg
        self.loss_gen_total.backward()
        self.gen_opt.step()

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    """
    # 3. Functional methods
    """

    def sample(self, x_a, x_b):
        self.eval()
        # random
        s_a = torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda()
        s_b = torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda()
        # s_a2 = torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda()
        # s_b2 = torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda()

        x_a_recon, x_b_recon, x_aba, x_bab = [], [], [], []
        x_ba, x_ab = [], []
        # x_ba2, x_ab2 = [], []

        for i in range(x_a.size(0)):
            (c_a, _), s_a_prime = self.gen_a.encode(x_a[i].unsqueeze(0))
            (c_b, _), s_b_prime = self.gen_b.encode(x_b[i].unsqueeze(0))

            # rec
            x_a_recon.append(self.gen_a.decode(c_a, s_a_prime))
            x_b_recon.append(self.gen_b.decode(c_b, s_b_prime))

            # generate
            ba = self.gen_a.decode(c_b, s_a[i].unsqueeze(0))
            # ba2 = self.gen_a.decode(c_b, s_a2[i].unsqueeze(0))
            x_ba.append(ba)  #
            # x_ba2.append(ba2)

            ab = self.gen_b.decode(c_a, s_b[i].unsqueeze(0))
            # ab2 = self.gen_b.decode(c_a, s_b2[i].unsqueeze(0))
            x_ab.append(ab)  #
            # x_ab2.append(ab2)

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
        return x_a, x_a_recon, x_ab, x_aba, x_b, x_b_recon, x_ba, x_bab

    def resume(self, checkpoint_dir, hyperparameters, train=True, set_iteration=None):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen", set_iteration)
        state_dict = torch.load(last_model_name)
        self.gen_a.load_state_dict(state_dict['a'])
        self.gen_b.load_state_dict(state_dict['b'])
        iterations = int(last_model_name[-11:-3])

        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis", set_iteration)
        state_dict = torch.load(last_model_name)
        self.dis_a.load_state_dict(state_dict['a'])
        self.dis_b.load_state_dict(state_dict['b'])

        if train:
            # Load optimizers
            state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
            self.dis_opt.load_state_dict(state_dict['dis'])
            self.gen_opt.load_state_dict(state_dict['gen'])

            # Reinitilize schedulers
            self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
            self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)

        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations, is_pre_train=True):
        # Save generators, discriminators, and optimizers
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        if is_pre_train:
            gen_name = os.path.join(snapshot_dir, 'a_gen_%08d.pt' % (iterations + 1))
            dis_name = os.path.join(snapshot_dir, 'a_dis_%08d.pt' % (iterations + 1))
        else:
            gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
            dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))

        torch.save({'a': self.gen_a.state_dict(), 'b': self.gen_b.state_dict()}, gen_name)
        torch.save({'a': self.dis_a.state_dict(), 'b': self.dis_b.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)
