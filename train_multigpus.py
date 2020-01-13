import os
import sys
import tensorboardX
import shutil
import argparse
import time
import random
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from utils import str2bool
from utils import get_all_data_loaders
from utils import prepare_sub_folder
from utils import get_model_list
from utils import get_scheduler
from utils import get_config
from utils import write_html, write_loss, write_2images
from models.backbones import VGG19
from models.trainer_multigpus import Trainer

try:
    from itertools import izip as zip
except ImportError:  # will be 3.x series
    pass


# 1. Training arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/selfie2anime.yaml', help='Path to the config file.')
parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Set gpu mode; [cpu, cuda]')
parser.add_argument('--gpu_ids', type=str, default='0', help='Set visible gpu ids')
parser.add_argument('--output_path', type=str, default='./results', help="outputs path")
parser.add_argument('--phase', type=str, default='pre-train', choices=['main-train', 'pre-train'],
                    help="main-train: training without pre-training | pre-train: with pre-training")
parser.add_argument('--resume', type=str2bool, default=False)
parser.add_argument('--iteration', type=str, default=None, help="start from a specific number of iteration")
opts = parser.parse_args()

cudnn.benchmark = True

use_multi_gpus = False
device_list = []
if opts.gpu_ids is not None:
    # os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_ids
    if len(opts.gpu_ids.split(',')) > 1:
        use_multi_gpus = True
        device_list = [int(_) for _ in opts.gpu_ids.split(',')]

# 2. Load experiment setting
config = get_config(opts.config)
display_size = config['display_size']

main_device = None
multi_mode_on = False
if opts.device == 'cuda' and torch.cuda.is_available():
    if torch.cuda.device_count() > 1 and use_multi_gpus:
        print("Using devices: ", device_list)
        config['batch_size'] *= len(device_list)
        main_device = device_list[0]
        trainer = torch.nn.DataParallel(Trainer(config, main_device), device_ids=device_list).cuda(main_device)
        multi_mode_on = True

    else:
        main_device = torch.cuda.current_device()
        trainer = Trainer(config)
        trainer = trainer.cuda(device=main_device)
else:
    trainer = Trainer(config)

train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_all_data_loaders(config)
N = train_loader_b.dataset.__len__()
n = test_loader_a.dataset.__len__()

# 3. Setup logger and output folders
output_path = opts.output_path
if not os.path.exists(output_path):
    os.mkdir(output_path)

model_name = os.path.splitext(os.path.basename(opts.config))[0]
train_writer = tensorboardX.SummaryWriter(os.path.join(output_path + "/logs", model_name))
output_directory = os.path.join(output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))

# 4. Setup the optimizers
beta1 = config['beta1']
beta2 = config['beta2']
lr = config['lr']

dis_params = list(trainer.module.dis_a.parameters()) + list(trainer.module.dis_b.parameters())
gen_params = list(trainer.module.gen_a.parameters()) + list(trainer.module.gen_b.parameters())
dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                           lr=lr, betas=(beta1, beta2), weight_decay=config['weight_decay'])
gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                           lr=lr, betas=(beta1, beta2), weight_decay=config['weight_decay'])
dis_scheduler = get_scheduler(dis_opt, config)
gen_scheduler = get_scheduler(gen_opt, config)


bce_loss = torch.nn.BCEWithLogitsLoss()
l1_loss = torch.nn.L1Loss()


def l1_criterion(fake, target):
    return torch.mean(torch.abs(fake - target))


def l2_criterion(fake, target):
    return torch.mean((fake - target) ** 2)


def calc_dis_loss(fake_outs, fake_cam_logits, real_outs, real_cam_logits):
    # calculate the loss to train D
    loss = 0
    gan_type = config['dis']['gan_type']
    for it, (fake_out, fake_cam_logit, real_out, real_cam_logit) in enumerate(
            zip(fake_outs, fake_cam_logits, real_outs, real_cam_logits)):
        if gan_type == 'lsgan':
            loss += torch.mean((fake_out - 0) ** 2) + torch.mean((real_out - 1) ** 2)
            # cam
            loss += torch.mean((fake_cam_logit - 0) ** 2) + torch.mean((real_cam_logit - 1) ** 2)
        elif gan_type == 'nsgan':
            all0 = torch.zeros_like(fake_out.data, requires_grad=False).cuda()
            all1 = torch.ones_like(real_out.data, requires_grad=False).cuda()
            loss += torch.mean(F.binary_cross_entropy(F.sigmoid(fake_out), all0) +
                               F.binary_cross_entropy(F.sigmoid(real_out), all1))
            # cam
            loss += torch.mean((fake_cam_logit - 0) ** 2) + torch.mean((real_cam_logit - 1) ** 2)
        else:
            assert 0, "Unsupported GAN type: {}".format(gan_type)
    return loss


def calc_gen_loss(fake_x_outs_x, fake_x_cam_logits_x, fake_x_outs_y, fake_x_cam_logits_y):
    # calculate the loss to train G
    # fake_ba_outs_a, fake_ba_cam_logits_a, fake_ba_outs_b, fake_ba_cam_logits_b
    loss = 0
    gan_type = config['dis']['gan_type']
    for _, (fake_x_out_x, fake_x_out_y, fake_x_cam_logit_x, fake_x_cam_logit_y) in enumerate(zip(
            fake_x_outs_x, fake_x_outs_y, fake_x_cam_logits_x, fake_x_cam_logits_y)):
        if gan_type == 'lsgan':
            loss += torch.mean((fake_x_out_x - 1) ** 2) + torch.mean((fake_x_cam_logit_x - 1) ** 2) + \
                    torch.mean((fake_x_out_y - 0) ** 2) + torch.mean((fake_x_cam_logit_y - 0) ** 2)
        elif gan_type == 'nsgan':
            all1 = torch.ones_like(fake_x_out_x.data, requires_grad=False).cuda()
            loss += torch.mean(F.binary_cross_entropy(F.sigmoid(fake_x_out_x), all1))
            loss += torch.mean((fake_x_cam_logit_x - 1) ** 2)
        else:
            assert 0, "Unsupported GAN type: {}".format(gan_type)
    return loss


def save(snapshot_dir, iterations, is_pre_train=True):
    # Save generators, discriminators, and optimizers
    opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
    if is_pre_train:
        gen_name = os.path.join(snapshot_dir, 'a_gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'a_dis_%08d.pt' % (iterations + 1))
    else:
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))

    torch.save({'a': trainer.module.gen_a.state_dict(), 'b': trainer.module.gen_b.state_dict()}, gen_name)
    torch.save({'a': trainer.module.dis_a.state_dict(), 'b': trainer.module.dis_b.state_dict()}, dis_name)
    torch.save({'gen': gen_opt.state_dict(), 'dis': dis_opt.state_dict()}, opt_name)


def resume(checkpoint_dir, set_iteration=None):
    # Load generators
    last_model_name = get_model_list(checkpoint_dir, "gen", set_iteration)
    state_dict = torch.load(last_model_name)
    trainer.module.gen_a.load_state_dict(state_dict['a'])
    trainer.module.gen_b.load_state_dict(state_dict['b'])
    iterations = int(last_model_name[-11:-3])

    # Load discriminators
    last_model_name = get_model_list(checkpoint_dir, "dis", set_iteration)
    state_dict = torch.load(last_model_name)
    trainer.module.dis_a.load_state_dict(state_dict['a'])
    trainer.module.dis_b.load_state_dict(state_dict['b'])

    # Load optimizers
    state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
    dis_opt.load_state_dict(state_dict['dis'])
    gen_opt.load_state_dict(state_dict['gen'])

    print('Resume from iteration %d' % iterations)
    return iterations


vgg19 = VGG19(init_weights=config['vgg_model_path'] + 'vgg19.pth', feature_mode=True).cuda(main_device)
vgg19.eval()

# 5. Start pre-training
batch_size = config['batch_size']
if opts.phase == 'pre-train':
    max_iter = int(config['max_pre_iter'] / batch_size)
    iterations = 0
    if opts.resume:
        iterations = resume(checkpoint_directory, set_iteration=opts.iteration)
        # Reinitilize schedulers
        dis_scheduler = get_scheduler(dis_opt, config, iterations)
        gen_scheduler = get_scheduler(gen_opt, config, iterations)

    print('Generator pre-training start !')
    start_time = time.time()
    iters = max_iter - iterations
    for it, (x_a, x_b) in enumerate(zip(train_loader_a, train_loader_b)):
        x_a, x_b = x_a.cuda(device=main_device).detach(), x_b.cuda(device=main_device).detach()

        # 1. Pre-training code
        """
        # Pre-train Generator:
        """
        # trainer.module.gen_pre_train_update(images_a, images_b, config)
        x_ab, x_ba, _, _ = trainer(x_a, x_b)

        # content loss
        a_feat = vgg19((x_a + 1) / 2)
        b_feat = vgg19((x_b + 1) / 2)
        ba_feat = vgg19((x_ba + 1) / 2)
        ab_feat = vgg19((x_ab + 1) / 2)

        diff_a = a_feat - ba_feat
        diff_b = b_feat - ab_feat

        alpha = config['alpha']
        loss_gen_c_a = torch.mean(torch.abs(diff_a)) + alpha * torch.mean(diff_a ** 2)
        loss_gen_c_b = torch.mean(torch.abs(diff_b)) + alpha * torch.mean(diff_b ** 2)
        loss_gen_c = config['vgg_w'] * (loss_gen_c_a + loss_gen_c_b)
        del loss_gen_c_a, loss_gen_c_b

        loss_gen_c.backward()
        gen_opt.step()

        #
        torch.cuda.synchronize(device=torch.cuda.current_device())

        """
        # Update learning rate:
        """
        # trainer.module.update_learning_rate()
        if dis_scheduler is not None:
            dis_scheduler.step()
        if gen_scheduler is not None:
            gen_scheduler.step()

        # 2. Show real-time pre-training loss
        print(
            "[%5d/%5d] time: %4.4f pre_g_loss: %.8f" %
            (iterations, max_iter, time.time() - start_time,
             loss_gen_c.item()))
        del loss_gen_c

        # 3. Write images
        if (iterations + 1) % config['image_display_iter'] == 0:
            sample = random.sample(range(N), display_size)
            train_display_images_a = torch.stack([train_loader_a.dataset[i] for i in sample]).cuda(main_device)
            train_display_images_b = torch.stack([train_loader_b.dataset[i] for i in sample]).cuda(main_device)

            with torch.no_grad():
                image_outputs = trainer.module.sample(train_display_images_a, train_display_images_b)
                del train_display_images_a, train_display_images_b
            write_2images(image_outputs, display_size, image_directory, 'pre_train_%08d' % (iterations + 1))
            del image_outputs

        # 4. Save network weights
        if (iterations + 1) % (config['snapshot_save_iter'] / 5) == 0:
            save(checkpoint_directory, iterations)

        iterations += 1

        if it >= iters:
            break
    print('Finish pre-training')

# 6. Start training
max_iter = int(config['max_iter'] / batch_size)
iterations = 0
if opts.resume:
    iterations = resume(checkpoint_directory, set_iteration=opts.iteration)
    # Reinitilize schedulers
    dis_scheduler = get_scheduler(dis_opt, config, iterations)
    gen_scheduler = get_scheduler(gen_opt, config, iterations)

print('training start !')
start_time = time.time()
while True:
    for it, (x_a, x_b) in enumerate(zip(train_loader_a, train_loader_b)):
        x_a, x_b = x_a.cuda(device=main_device).detach(), x_b.cuda(device=main_device).detach()

        # Main training code
        """
        # Discriminator update:
        """
        # trainer.module.dis_update(images_a, images_b, config)
        dis_opt.zero_grad()

        _, _, _, (
            fake_ba_outs_a, fake_ba_cam_logits_a, fake_ba_outs_b, fake_ba_cam_logits_b, real_a_outs, real_a_cam_logits,
            fake_ab_outs_b, fake_ab_cam_logits_b, fake_ab_outs_a, fake_ab_cam_logits_a, real_b_outs, real_b_cam_logits
        ) = trainer(x_a, x_b)

        # loss
        fake_ba_outs = fake_ba_outs_a + fake_ba_outs_b  # list
        fake_ba_cam_logits = fake_ba_cam_logits_a + fake_ba_cam_logits_b
        fake_ab_outs = fake_ab_outs_b + fake_ab_outs_a
        fake_ab_cam_logits = fake_ab_cam_logits_b + fake_ab_cam_logits_a

        loss_dis_a = calc_dis_loss(fake_ba_outs, fake_ba_cam_logits, real_a_outs, real_a_cam_logits)
        loss_dis_b = calc_dis_loss(fake_ab_outs, fake_ab_cam_logits, real_b_outs, real_b_cam_logits)
        loss_dis_total = config['adv_w'] * (loss_dis_a + loss_dis_b)
        del loss_dis_a, loss_dis_b, fake_ba_outs, fake_ba_cam_logits, real_a_outs, real_a_cam_logits, fake_ab_outs, fake_ab_cam_logits, real_b_outs, real_b_cam_logits

        loss_dis_total.backward()
        dis_opt.step()

        """
        # Generator update
        """
        # trainer.module.gen_update(images_a, images_b, config)
        gen_opt.zero_grad()

        x_ab, x_ba, (
            s_a, s_b,
            c_a, c_a_logit, s_a_prime,
            c_b, c_b_logit, s_b_prime,
            x_a_recon, x_b_recon,
            c_ba, s_ba, c_ab, s_ab, x_aba, x_bab,
            fake_c_a_logit, fake_c_b_logit), (
            fake_ba_outs_a, fake_ba_cam_logits_a, fake_ba_outs_b, fake_ba_cam_logits_b, _, _,
            fake_ab_outs_b, fake_ab_cam_logits_b, fake_ab_outs_a, fake_ab_cam_logits_a, _, _) = trainer(x_a, x_b)

        # loss
        # --- 1. adv loss
        loss_gen_adv_a = calc_gen_loss(fake_ba_outs_a, fake_ba_cam_logits_a, fake_ba_outs_b, fake_ba_cam_logits_b)
        loss_gen_adv_b = calc_gen_loss(fake_ab_outs_b, fake_ab_cam_logits_b, fake_ab_outs_a, fake_ab_cam_logits_a)
        loss_gen_adv = config['adv_w'] * (loss_gen_adv_a + loss_gen_adv_b)
        del loss_gen_adv_a, loss_gen_adv_b, fake_ba_outs_a, fake_ba_cam_logits_a, fake_ba_outs_b, fake_ba_cam_logits_b, \
            fake_ab_outs_b, fake_ab_cam_logits_b, fake_ab_outs_a, fake_ab_cam_logits_a

        # --- 2. reconstruction loss
        loss_gen_recon_c_b = l1_criterion(c_ba, c_b)
        loss_gen_recon_c_a = l1_criterion(c_ab, c_a)

        # loss_gen_recon_s_a = recon_criterion(s_ba, s_a_prime) - recon_criterion(s_ba, s_b_prime)
        # loss_gen_recon_s_a = recon_criterion(s_ba, s_a)
        # loss_gen_recon_s_a = recon_criterion(s_ba, s_a_prime)
        loss_gen_recon_s_a = l1_criterion(s_ba, s_a) + config['alpha'] * torch.mean((s_ba - s_a_prime) ** 2)

        # loss_gen_recon_s_b = recon_criterion(s_ab, s_b_prime) - recon_criterion(s_ba, s_a_prime)
        # loss_gen_recon_s_b = recon_criterion(s_ab, s_b)
        # loss_gen_recon_s_b = recon_criterion(s_ab, s_b_prime)
        loss_gen_recon_s_b = l1_criterion(s_ab, s_b) + config['alpha'] * torch.mean((s_ab - s_b_prime) ** 2)

        loss_gen_recon_x_a = l1_criterion(x_a_recon, x_a)
        loss_gen_recon_x_b = l1_criterion(x_b_recon, x_b)

        loss_gen_recon_x = config['rec_w'] * (loss_gen_recon_x_a + loss_gen_recon_x_b)
        loss_gen_recon_c = config['rec_c_w'] * (loss_gen_recon_c_a + loss_gen_recon_c_b)
        loss_gen_recon_s = config['rec_s_w'] * (loss_gen_recon_s_a + loss_gen_recon_s_b)
        # loss_gen_recon_s = loss_gen_recon_s_a + loss_gen_recon_s_b  #

        del loss_gen_recon_x_a, loss_gen_recon_x_b, x_a_recon, x_b_recon, \
            loss_gen_recon_c_a, loss_gen_recon_c_b, c_ab, c_ba, c_a, c_b, \
            loss_gen_recon_s_a, loss_gen_recon_s_b, s_ba, s_ab, s_a_prime, s_b_prime

        # --- 3. cyc loss
        loss_gen_cyc_x_a = l1_criterion(x_aba, x_a) if config['cyc_w'] > 0 else 0
        loss_gen_cyc_x_b = l1_criterion(x_bab, x_b) if config['cyc_w'] > 0 else 0
        loss_gen_cyc = config['cyc_w'] * (loss_gen_cyc_x_a + loss_gen_cyc_x_b)
        del loss_gen_cyc_x_a, loss_gen_cyc_x_b, x_aba, x_bab

        # --- 4. domain-invariant perceptual loss
        loss_gen_vgg = 0
        if config['vgg_w'] > 0:
            a_feat = vgg19((x_a + 1) / 2)
            b_feat = vgg19((x_b + 1) / 2)

            ba_feat = vgg19((x_ba + 1) / 2)
            ab_feat = vgg19((x_ab + 1) / 2)
            del x_ab, x_ba, x_a, x_b

            # content should be the same
            diff_a = a_feat - ab_feat
            diff_b = b_feat - ba_feat

            alpha = config['alpha']
            # loss_gen_vgg_a = torch.mean(torch.abs(diff_a)) + alpha * torch.mean(diff_a ** 2)
            # loss_gen_vgg_b = torch.mean(torch.abs(diff_b)) + alpha * torch.mean(diff_b ** 2)
            loss_gen_vgg_a = torch.mean(torch.abs(diff_a)) + torch.mean(diff_a ** 2)
            loss_gen_vgg_b = torch.mean(torch.abs(diff_b)) + torch.mean(diff_b ** 2)
            loss_gen_vgg = config['vgg_w'] * (loss_gen_vgg_a + loss_gen_vgg_b)
            del loss_gen_vgg_a, loss_gen_vgg_b, diff_a, a_feat, ba_feat, diff_b, b_feat, ab_feat

        # --- 5. cam loss
        loss_gen_cam_a = bce_loss(c_a_logit, torch.ones_like(c_a_logit, requires_grad=False).cuda(main_device)) + \
                         bce_loss(fake_c_a_logit,
                                  torch.zeros_like(fake_c_a_logit, requires_grad=False).cuda(main_device))
        loss_gen_cam_b = bce_loss(c_b_logit, torch.ones_like(c_b_logit, requires_grad=False).cuda(main_device)) + \
                         bce_loss(fake_c_b_logit,
                                  torch.zeros_like(fake_c_b_logit, requires_grad=False).cuda(main_device))
        loss_gen_cam = config['cam_w'] * (loss_gen_cam_a + loss_gen_cam_b)
        del loss_gen_cam_a, loss_gen_cam_b, c_a_logit, fake_c_a_logit, c_b_logit, fake_c_b_logit

        # total loss
        loss_gen_total = loss_gen_adv + \
                         loss_gen_recon_x + \
                         loss_gen_recon_s + \
                         loss_gen_recon_c + \
                         loss_gen_cyc + \
                         loss_gen_cam + \
                         loss_gen_vgg

        if not (iterations > 40000 and loss_gen_total.item() > 100):
            loss_gen_total.backward()
            gen_opt.step()

        torch.cuda.synchronize(device=torch.cuda.current_device())

        """
        # Update learning rate:
        """
        # trainer.module.update_learning_rate()
        if dis_scheduler is not None:
            dis_scheduler.step()
        if gen_scheduler is not None:
            gen_scheduler.step()

        # Show real-time training loss
        message = "[%5d/%5d] time: %4.4f d_loss: %.4f, g_loss: %.4f, " % (
            iterations, max_iter, time.time() - start_time, loss_dis_total.item(), loss_gen_total.item()) + \
                  "adv_loss: %.4f, rec_loss: %.4f, rec_s_loss: %.4f, rec_c_loss: %.4f, cyc_loss: %.4f, cam_loss: %.4f, vgg_loss: %.4f" % (
                      loss_gen_adv.item(), loss_gen_recon_x.item(), loss_gen_recon_s.item(), loss_gen_recon_c.item(),
                      loss_gen_cyc.item(), loss_gen_cam.item(), loss_gen_vgg.item())
        if iterations > 40000 and loss_gen_total.item() > 100:
            message += "  # GD passed!"

        print(message)
        del loss_dis_total, loss_gen_total, \
            loss_gen_recon_x, loss_gen_cyc, loss_gen_cam, loss_gen_vgg

        # Dump training stats in log file
        if (iterations + 1) % config['log_iter'] == 0:
            print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
            write_loss(iterations, trainer, train_writer)

        # Write images
        if (iterations + 1) % config['image_save_iter'] == 0:
            train_sample = random.sample(range(N), display_size)
            test_sample = random.sample(range(n), display_size)

            train_display_images_a = torch.stack([train_loader_a.dataset[i] for i in train_sample]).cuda(main_device)
            train_display_images_b = torch.stack([train_loader_b.dataset[i] for i in train_sample]).cuda(main_device)
            test_display_images_a = torch.stack([test_loader_a.dataset[i] for i in test_sample]).cuda(main_device)
            test_display_images_b = torch.stack([test_loader_b.dataset[i] for i in test_sample]).cuda(main_device)

            with torch.no_grad():
                test_image_outputs = trainer.module.sample(test_display_images_a, test_display_images_b)
                train_image_outputs = trainer.module.sample(train_display_images_a, train_display_images_b)
            write_2images(test_image_outputs, display_size, image_directory, 'test_%08d' % (iterations + 1))
            write_2images(train_image_outputs, display_size, image_directory, 'train_%08d' % (iterations + 1))
            # HTML
            write_html(output_directory + "/index.html", iterations + 1, config['image_save_iter'], 'images')

        if (iterations + 1) % config['image_display_iter'] == 0:
            sample = random.sample(range(N), display_size)
            train_display_images_a = torch.stack([train_loader_a.dataset[i] for i in sample]).cuda(main_device)
            train_display_images_b = torch.stack([train_loader_b.dataset[i] for i in sample]).cuda(main_device)

            with torch.no_grad():
                image_outputs = trainer.module.sample(train_display_images_a, train_display_images_b)
            write_2images(image_outputs, display_size, image_directory, 'train_current')

        # Save network weights
        if (iterations + 1) % config['snapshot_save_iter'] == 0:
            save(checkpoint_directory, iterations, is_pre_train=False)

        iterations += 1
        if iterations >= max_iter:
            sys.exit('Finish training')
