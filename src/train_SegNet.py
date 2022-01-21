import time
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils import data
from models.models import create_model
import os

from options.train_options import TrainOptions
from data.dataprocess import DataProcess

if __name__ == '__main__':
    opt = TrainOptions().parse()
    dataset = DataProcess(opt)
    train_loader = data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.num_workers)
    model = create_model(opt)
    model.print_networks()
    log_dir = os.path.join(opt.log_dir, opt.name).replace('\\', '/')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir, comment=opt.name)

    total_steps = 0
    start_time = time.time()
    for epoch in range(1, opt.n_epochs + opt.n_epochs_decay + 1):
        for samples in train_loader:
            iter_start_time = time.time()
            total_steps += opt.batchSize
            model.set_input(samples)
            model.optmize_parameters()

            if total_steps % opt.print_freq == 0:
                loss = model.get_current_loss()
                errors = model.get_statistic_errors()
                writer.add_scalar('G', loss['G'], total_steps + 1)
                for k, v in loss.items():
                    loss[k] = round(v, 3)
                for k, v in errors.items():
                    errors[k] = round(v, 3)
                print('Epoch->', epoch, 'Total_steps->', total_steps, 'Loss->', loss)
                print('Epoch->', epoch, 'Total_steps->', total_steps, 'Errors->', errors)
                print('-------------------*****-------------------TIME-->', (time.time()-start_time)/3600., 'hour')

            if total_steps % opt.display_freq == 0:
                visuals = model.get_current_visuals()
                images = torch.cat((visuals['real_DI'],
                                    visuals['real_SI'],
                                    visuals['real_SS'],
                                    visuals['fake_SS']), 2)
                grid = torchvision.utils.make_grid(images)
                writer.add_image('Epoch_(%d)_(%d)' % (epoch, total_steps + 1), grid, total_steps + 1)

        if epoch % opt.save_epoch_freq == 0:
            model.save_networks(epoch)

        model.update_learning_rate()
    writer.close()




