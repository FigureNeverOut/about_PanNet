# This is a pytorch version for the work of PanNet
# YW Jin, X Wu, LJ Deng(UESTC);
# 2020-09;
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import Dataset_Pro
from model import Dicnn,summaries
import numpy as np

import shutil
from torch.utils.tensorboard import SummaryWriter

###################################################################
# ------------------- Pre-Define Part----------------------
###################################################################
# ============= 1) Pre-Define =================== #
SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
# cudnn.benchmark = True  ###自动寻找最优算法
cudnn.deterministic = True
cudnn.benchmark = False

# ============= 2) HYPER PARAMS(Pre-Defined) ==========#
lr = 0.0002  #学习率
epochs = 1000 # 450
ckpt = 50#每50步保存一次
batch_size = 32
model_path = "Weights/1000/.pth"

# ============= 3) Load Model + Loss + Optimizer + Learn_rate_update ==========#
model = Dicnn().cpu()
if os.path.isfile(model_path):
    model.load_state_dict(torch.load(model_path))   ## Load the pretrained Encoder
    print('PANnet is Successfully Loaded from %s' % (model_path))

summaries(model, grad=True)    ## Summary the Network
criterion = nn.MSELoss(size_average=True).cuda()  ## Define the Loss function L2Loss

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0)   ## optimizer 1: Adam
#lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.1)   # learning-rate update

#optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-7)  ## optimizer 2: SGD
#lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=180, gamma=0.1)  # learning-rate update: lr = lr* 1/gamma for each step_size = 180

# ============= 4) Tensorboard_show + Save_model ==========#
# if os.path.exists('train_logs'):  # for tensorboard: copy dir of train_logs  ## Tensorboard_show: case 1
#   shutil.rmtree('train_logs')  # ---> console (see tensorboard): tensorboard --logdir = dir of train_logs
lr_scheduler=torch.optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=200,gamma=0.5)
writer = SummaryWriter('./train_logs/50')    ## Tensorboard_show: case 2

def save_checkpoint(model, epoch):  # save model function（保存模型函数
    model_out_path = 'Weights' + '/' + "{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)

###################################################################
# ------------------- Main Train (Run second)----------------------
###################################################################
def train(training_data_loader, validate_data_loader,start_epoch=0):
    print('Start training...')
    # epoch 450, 450*550 / 2 = 123750 / 8806 = 14/per imgae

    for epoch in range(start_epoch, epochs, 1):

        epoch += 1
        epoch_train_loss, epoch_val_loss = [], []

        # ============Epoch Train=============== #
        model.train()

        for iteration, batch in enumerate(training_data_loader, 1): # 100  3
            # gt Nx8x64x64
            # lms Nx8x64x64
            # ms_hp Nx8x16x16
            # pan_hp Nx1x64x64
            gt, lms, ms_hp, pan_hp = batch[0].cpu(), batch[1].cpu(), batch[2].cpu(), batch[3].cpu()

            optimizer.zero_grad()  # fixed

            hp_sr = model(ms_hp, pan_hp)  # call model
            sr = lms + hp_sr  # output:= lms + hp_sr

            loss = criterion(sr, gt)  # compute loss
            epoch_train_loss.append(loss.item())  # save all losses into a vector for one epoch

            loss.backward()   # fixed
            optimizer.step()  # fixed
            lr_scheduler.step()  # if update_lr, activate here!

            for name, layer in model.named_parameters():
                # writer.add_histogram('torch/'+name + '_grad_weight_decay', layer.grad, epoch*iteration)
                writer.add_histogram('net/'+name + '_data_weight_decay', layer, epoch*iteration)

        #lr_scheduler.step()  # if update_lr, activate here!

        t_loss = np.nanmean(np.array(epoch_train_loss))  # compute the mean value of all losses, as one epoch loss
        writer.add_scalar('mse_loss/t_loss', t_loss, epoch)  # write to tensorboard to check
        print('Epoch: {}/{} training loss: {:.7f}'.format(epochs, epoch, t_loss))  # print loss for each epoch

        if epoch % ckpt == 0:  # if each ckpt epochs, then start to save model
            save_checkpoint(model, epoch)

        # ============Epoch Validate=============== #
        model.eval()# fixed
        with torch.no_grad():  # fixed
            for iteration, batch in enumerate(validate_data_loader, 1):
                gt, lms, ms_hp, pan_hp = batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[3].cuda()

                hp_sr = model(ms_hp, pan_hp)
                sr = lms + hp_sr

                loss = criterion(sr, gt)
                epoch_val_loss.append(loss.item())

        if epoch % 10 == 0:
            v_loss = np.nanmean(np.array(epoch_val_loss))
            writer.add_scalar('val/v_loss', v_loss, epoch)
            print('             validate loss: {:.7f}'.format(v_loss))

    writer.close()  # close tensorboard


###################################################################
# ------------------- Main Function (Run first) -------------------
###################################################################
if __name__ == "__main__":
    train_set = Dataset_Pro('./training_data/train_small.h5')  # creat data for training   # 100
    training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=batch_size, shuffle=True,  # 3 32
                                      pin_memory=True, drop_last=True)  # put training data to DataLoader for batches

    validate_set = Dataset_Pro('./training_data/valid_small.h5')  # creat data for validation
    validate_data_loader = DataLoader(dataset=validate_set, num_workers=0, batch_size=batch_size, shuffle=True,
                                      pin_memory=True, drop_last=True)  # put training data to DataLoader for batches

    train(training_data_loader, validate_data_loader)  # call train function (call: Line 66)
