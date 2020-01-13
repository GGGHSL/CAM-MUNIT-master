from utils import get_all_data_loaders, prepare_sub_folder, write_html, write_loss, get_config, write_2images
import argparse
from models.trainer import Trainer
import torch.backends.cudnn as cudnn
import torch
import os
import tensorboardX
import shutil

# 1. Training arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/selfie2anime.yaml', help='Path to the config file.')
parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Set gpu mode; [cpu, cuda]')
parser.add_argument('--display_nums', type=int, default=10, help='number of images displayed during test phase')
parser.add_argument('--load_path', type=str, default='./result', help="outputs path")
parser.add_argument('--iteration', type=str, default=None, help="start from a specific number of iteration")
parser.add_argument('--output_path', type=str, default='./test', help="outputs path")
opts = parser.parse_args()

cudnn.benchmark = True

# 2. Load experiment setting
config = get_config(opts.config)
display_size = config['display_size']

if opts.device == 'cuda' and torch.cuda.is_available():
    trainer = Trainer(config)
    trainer = trainer.cuda()
else:
    trainer = Trainer(config)

_, _, test_loader_a, test_loader_b = get_all_data_loaders(config)
n = test_loader_a.dataset.__len__()

# 3. Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0]
checkpoint_directory = os.path.join(opts.load_path + "/outputs", model_name) + "/checkpoints"
iterations = trainer.resume(checkpoint_directory, hyperparameters=config, set_iteration=opts.iteration)

output_path = opts.output_path
output_path += '_' + str(iterations)
if not os.path.exists(output_path):
    os.mkdir(output_path)

output_directory = os.path.join(output_path, model_name)
_, image_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))

# 4. Start testing
trainer.eval()
print('Testing start !')
for i in range(n // opts.display_nums):
    idx = list(range(i*opts.display_nums, min((i+1)*opts.display_nums, n)))

    # Write images
    test_display_images_a = torch.stack([test_loader_a.dataset[_] for _ in idx]).cuda()
    test_display_images_b = torch.stack([test_loader_b.dataset[_] for _ in idx]).cuda()

    with torch.no_grad():
        test_image_outputs = trainer.sample(test_display_images_a, test_display_images_b)
    write_2images(test_image_outputs, display_size, image_directory, 'test_display_%08d' % (i + 1))
print('Finished!')
