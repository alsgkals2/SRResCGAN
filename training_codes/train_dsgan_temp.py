import argparse
import os
import torch.utils.data
from models import DSGAN
from utils import utils_dsgan as utils
from PIL import Image
import torchvision.transforms.functional as TF
import loss
import torch.optim as optim
from tensorboardX import SummaryWriter
import argparse
import os
import torch.optim as optim
import torch.utils.data
import torchvision.utils as tvutils
import loss
import utils
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Apply the trained model to create a dataset')
parser.add_argument('--checkpoint', default=None, type=str, help='checkpoint model to use')
parser.add_argument('--artifacts', default='', type=str, help='selecting different artifacts type')
parser.add_argument('--name', default='', type=str, help='additional string added to folder path')
parser.add_argument('--dataset', default='df2k', type=str, help='selecting different datasets')
parser.add_argument('--track', default='train', type=str, help='selecting train or valid track')
parser.add_argument('--num_res_blocks', default=8, type=int, help='number of ResNet blocks')
parser.add_argument('--cleanup_factor', default=2, type=int, help='downscaling factor for image cleanup')
parser.add_argument('--upscale_factor', default=4, type=int, choices=[4], help='super resolution upscale factor')
opt = parser.parse_args()

# define input and target directories
input_target_hr_dir = 'datasets/DIV2K/DIV2K_train_HR/' # HR_target
input_target_lr_dir = 'datasets/DIV2K/DIV2K_train_LR/X4' # LR_target
target_hr_files = [os.path.join(input_target_hr_dir, x) for x in os.listdir(input_target_hr_dir) if utils.is_image_file(x)]
target_lr_files = [os.path.join(input_target_lr_dir, x) for x in os.listdir(input_target_lr_dir) if utils.is_image_file(x)]

tdsr_hr_dir = 'datasets/DF2K/generated/HR/'
tdsr_lr_dir = 'datasets/DF2K/generated/LR/'

if not os.path.exists(tdsr_hr_dir):
    os.makedirs(tdsr_hr_dir)
if not os.path.exists(tdsr_lr_dir):
    os.makedirs(tdsr_lr_dir)

# prepare neural networks
model_path = 'pretrained_nets/DSGAN/300_G.pth'
model_g = DSGAN.Generator(n_res_blocks=opt.num_res_blocks)
model_g.load_state_dict(torch.load(model_path), strict=True)
model_g.eval()
model_g = model_g.cuda()
print('# generator parameters:', sum(param.numel() for param in model_g.parameters()))


#여기부터 학습하려고 코드 가져옴(github  https://github.com/ManuelFritsche/real-world-sr/blob/master/dsgan/train.py  )
model_d = DSGAN.Discriminator(kernel_size=opt.kernel_size, gaussian=opt.gaussian, wgan=opt.wgan, highpass=opt.highpass)
print('# discriminator parameters:', sum(param.numel() for param in model_d.parameters()))

g_loss_module = loss.GeneratorLoss(**vars(opt))

# filters are used for generating validation images
filter_low_module = DSGAN.FilterLow(kernel_size=opt.kernel_size, gaussian=opt.gaussian, include_pad=False)
filter_high_module = DSGAN.FilterHigh(kernel_size=opt.kernel_size, gaussian=opt.gaussian, include_pad=False)
if torch.cuda.is_available():
    model_g = model_g.cuda()
    model_d = model_d.cuda()
    filter_low_module = filter_low_module.cuda()
    filter_high_module = filter_high_module.cuda()

# define optimizers
optimizer_g = optim.Adam(model_g.parameters(), lr=opt.learning_rate, betas=[opt.adam_beta_1, 0.999])
optimizer_d = optim.Adam(model_d.parameters(), lr=opt.learning_rate, betas=[opt.adam_beta_1, 0.999])
start_decay = opt.num_epochs - opt.num_decay_epochs
scheduler_rule = lambda e: 1.0 if e < start_decay else 1.0 - max(0.0, float(e - start_decay) / opt.num_decay_epochs)
scheduler_g = optim.lr_scheduler.LambdaLR(optimizer_g, lr_lambda=scheduler_rule)
scheduler_d = optim.lr_scheduler.LambdaLR(optimizer_d, lr_lambda=scheduler_rule)

# load/initialize parameters
if opt.checkpoint is not None:
    checkpoint = torch.load(opt.checkpoint)
    start_epoch = checkpoint['epoch'] + 1
    iteration = checkpoint['iteration'] + 1
    model_g.load_state_dict(checkpoint['model_g_state_dict'])
    model_d.load_state_dict(checkpoint['models_d_state_dict'])
    optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
    optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
    scheduler_g.load_state_dict(checkpoint['scheduler_g_state_dict'])
    scheduler_d.load_state_dict(checkpoint['scheduler_d_state_dict'])
    print('Continuing training at epoch %d' % start_epoch)
else:
    start_epoch = 1
    iteration = 1

# prepare tensorboard summary
summary_path = ''
if opt.saving:
    if opt.save_path is None:
        save_path = ''
    else:
        save_path = '/' + opt.save_path
    dir_index = 0
    while os.path.isdir('runs/' + save_path + '/' + str(dir_index)):
        dir_index += 1
    summary_path = 'runs' + save_path + '/' + str(dir_index)
    writer = SummaryWriter(summary_path)
    print('Saving summary into directory ' + summary_path + '/')

# training iteration
for epoch in range(start_epoch, opt.num_epochs + 1):
    train_bar = tqdm(train_loader, desc='[%d/%d]' % (epoch, opt.num_epochs))
    model_g.train()
    model_d.train()

    for input_img, disc_img in train_bar:
        iteration += 1
        if torch.cuda.is_available():
            input_img = input_img.cuda()
            disc_img = disc_img.cuda()

        # Estimate scores of fake and real images
        fake_img = model_g(input_img)
        if opt.ragan:
            real_tex = model_d(disc_img, fake_img)
            fake_tex = model_d(fake_img, disc_img)
        else:
            real_tex = model_d(disc_img)
            fake_tex = model_d(fake_img)

        # Update Discriminator network
        if iteration % opt.disc_freq == 0:
            # calculate gradient penalty
            if opt.wgan:
                rand = torch.rand(1).item()
                sample = rand * disc_img + (1 - rand) * fake_img
                gp_tex = model_d(sample)
                gradient = torch.autograd.grad(gp_tex.mean(), sample, create_graph=True)[0]
                grad_pen = 10 * (gradient.norm() - 1) ** 2
            else:
                grad_pen = None
            # update discriminator
            model_d.zero_grad()
            d_tex_loss = loss.discriminator_loss(real_tex, fake_tex, wasserstein=opt.wgan, grad_penalties=grad_pen)
            d_tex_loss.backward(retain_graph=True)
            optimizer_d.step()
            # save data to tensorboard
            if opt.saving:
                writer.add_scalar('loss/d_tex_loss', d_tex_loss, iteration)
                if opt.wgan:
                    writer.add_scalar('disc_score/gradient_penalty', grad_pen.mean().data.item(), iteration)

        # Update Generator network
        if iteration % opt.gen_freq == 0:
            # update discriminator
            model_g.zero_grad()
            g_loss = g_loss_module(fake_tex, fake_img, input_img)
            assert not torch.isnan(g_loss), 'Generator loss returns NaN values'
            g_loss.backward()
            optimizer_g.step()
            # save data to tensorboard
            if opt.saving:
                writer.add_scalar('loss/perceptual_loss', g_loss_module.last_per_loss, iteration)
                writer.add_scalar('loss/color_loss', g_loss_module.last_col_loss, iteration)
                writer.add_scalar('loss/g_tex_loss', g_loss_module.last_tex_loss, iteration)
                writer.add_scalar('loss/g_overall_loss', g_loss, iteration)

        # save data to tensorboard
        rgb_loss = g_loss_module.rgb_loss(fake_img, input_img)
        mean_loss = g_loss_module.mean_loss(fake_img, input_img)
        if opt.saving:
            writer.add_scalar('loss/rgb_loss', rgb_loss, iteration)
            writer.add_scalar('loss/mean_loss', mean_loss, iteration)
            writer.add_scalar('disc_score/real', real_tex.mean().data.item(), iteration)
            writer.add_scalar('disc_score/fake', fake_tex.mean().data.item(), iteration)
        train_bar.set_description(desc='[%d/%d]' % (epoch, opt.num_epochs))

    scheduler_d.step()
    scheduler_g.step()
    if opt.saving:
        writer.add_scalar('param/learning_rate', torch.Tensor(scheduler_g.get_lr()), epoch)

    # validation step
    if epoch % opt.val_interval == 0 or epoch % opt.val_img_interval == 0:
        val_bar = tqdm(val_loader, desc='[Validation]')
        model_g.eval()
        val_images = []
        with torch.no_grad():
            # initialize variables to estimate averages
            mse_sum = psnr_sum = rgb_loss_sum = mean_loss_sum = 0
            per_loss_sum = col_loss_sum = tex_loss_sum = 0

            # validate on each image in the val dataset
            for index, (input_img, disc_img, target_img) in enumerate(val_bar):
                if torch.cuda.is_available():
                    input_img = input_img.cuda()
                    target_img = target_img.cuda()
                fake_img = torch.clamp(model_g(input_img), min=0, max=1)

                mse = ((fake_img - target_img) ** 2).mean().data
                mse_sum += mse
                psnr_sum += -10 * torch.log10(mse)
                rgb_loss_sum += g_loss_module.rgb_loss(fake_img, target_img)
                mean_loss_sum += g_loss_module.mean_loss(fake_img, target_img)
                per_loss_sum += g_loss_module.perceptual_loss(fake_img, target_img)
                col_loss_sum += g_loss_module.color_loss(fake_img, target_img)

                # generate images
                if epoch % opt.val_img_interval == 0 and epoch != 0:
                    blur = filter_low_module(fake_img)
                    hf = filter_high_module(fake_img)
                    val_image_list = [
                        utils.display_transform()(target_img.data.cpu().squeeze(0)),
                        utils.display_transform()(fake_img.data.cpu().squeeze(0)),
                        utils.display_transform()(disc_img.squeeze(0)),
                        utils.display_transform()(blur.data.cpu().squeeze(0)),
                        utils.display_transform()(hf.data.cpu().squeeze(0))]
                    n_val_images = len(val_image_list)
                    val_images.extend(val_image_list)

            if opt.saving and len(val_loader) > 0:
                # save validation values
                writer.add_scalar('val/mse', mse_sum/len(val_set), iteration)
                writer.add_scalar('val/psnr', psnr_sum / len(val_set), iteration)
                writer.add_scalar('val/rgb_error', rgb_loss_sum / len(val_set), iteration)
                writer.add_scalar('val/mean_error', mean_loss_sum / len(val_set), iteration)
                writer.add_scalar('val/perceptual_error', per_loss_sum / len(val_set), iteration)
                writer.add_scalar('val/color_error', col_loss_sum / len(val_set), iteration)

                # save image results
                if epoch % opt.val_img_interval == 0 and epoch != 0:
                    val_images = torch.stack(val_images)
                    val_images = torch.chunk(val_images, val_images.size(0) // (n_val_images * 5))
                    val_save_bar = tqdm(val_images, desc='[Saving results]')
                    for index, image in enumerate(val_save_bar):
                        image = tvutils.make_grid(image, nrow=n_val_images, padding=5)
                        out_path = 'val/target_fake_tex_disc_f-wav_t-wav_' + str(index)
                        writer.add_image('val/target_fake_crop_low_high_' + str(index), image, iteration)

    # save model parameters
    if opt.saving and epoch % opt.save_model_interval == 0 and epoch != 0:
        path = './checkpoints/' + save_path + '/iteration_' + str(iteration) + '.tar'
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        state_dict = {
            'epoch': epoch,
            'iteration': iteration,
            'model_g_state_dict': model_g.state_dict(),
            'models_d_state_dict': model_d.state_dict(),
            'optimizer_g_state_dict': optimizer_g.state_dict(),
            'optimizer_d_state_dict': optimizer_d.state_dict(),
            'scheduler_g_state_dict': scheduler_g.state_dict(),
            'scheduler_d_state_dict': scheduler_d.state_dict(),
        }
        torch.save(state_dict, path)
        path = './checkpoints' + save_path + '/last_iteration.tar'
        torch.save(state_dict, path)
"""

# generate the noisy images
idx = 0
with torch.no_grad():
    for file_hr, file_lr in zip(target_hr_files, target_lr_files):
        idx +=1
        print('Image No.:', idx)
        # load HR image
        input_img_hr = Image.open(file_hr)
        input_img_hr = TF.to_tensor(input_img_hr)

        # Save input_img as HR image for TDSR
        path = os.path.join(tdsr_hr_dir, os.path.basename(file_hr))
        TF.to_pil_image(input_img_hr).save(path, 'PNG')

        # load LR image
        input_img_lr = Image.open(file_lr)
        input_img_lr = TF.to_tensor(input_img_lr)

        # Apply model to generate the noisy resize_img
        if torch.cuda.is_available():
            input_img_lr = input_img_lr.unsqueeze(0).cuda()

        resize_noisy_img = model_g(input_img_lr).squeeze(0).cpu()

        # Save resize_noisy_img as LR image for TDSR
        path = os.path.join(tdsr_lr_dir, os.path.basename(file_lr))
        TF.to_pil_image(resize_noisy_img).save(path, 'PNG')
        #break
"""