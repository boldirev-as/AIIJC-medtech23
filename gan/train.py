import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from gan import Discriminator, Generator, weights_init
from dataset import DatasetECG
import torch.autograd as autograd
import json
import os

beta1 = 0.5
beta2 = 0.999
p_coeff = 10
n_critic = 5
lr = 0.00002 
workers = 2
batch_size = 8
nc = 1
nz = 100
ngf = 64 
ndf = 64
epoch_num = 32
ngpu = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
matplotlib.use('TkAgg')

save_path = f'./gan/nets/version_2048maxConv_ep{epoch_num}_nz{nz}_nf{ngf}_nc{nc}/'
#os.mkdir(save_path)
os.chdir('./gan/')
def main():
    # load training data
    trainset = DatasetECG("../train_annotations.csv", "./../transformed_train")

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=False
    )

    real_batch = next(iter(trainloader))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Example of second lead of ecg")
    #plt.plot(real_batch[0][0][2])
    plt.plot(real_batch[0][0])
    plt.show(block=False)

    # init netD and netG
    netD = Discriminator(nc, ndf, ngpu).to(device)
    netD.apply(weights_init)

    netG = Generator(nz, nc,ngf,ngpu).to(device)
    netG.apply(weights_init)

    # used for visualizing training process
    fixed_noise = torch.randn(1, nz, 1, device=device)

    # optimizers
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, beta2))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, beta2))
    #optimizerD = optim.RMSprop(netD.parameters(), lr=lr)
    #optimizerG = optim.RMSprop(netG.parameters(), lr=lr)

    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    for epoch in range(epoch_num):
        for step, (data, _j) in enumerate(trainloader):
            # training netD
            real_cpu = torch.unsqueeze(data, dim=1).to(device)
            b_size = real_cpu.size(0)
            netD.zero_grad()
            
            noise = torch.randn(b_size, nz, 1, device=device)

            fake = netG(noise)

            # gradient penalty
            eps = torch.Tensor(b_size, 1, 1, ).uniform_(0, 1)
            eps = eps.to(device)
            #print("EPS shape", eps.shape)
            #print("real_cpu shape", real_cpu.shape)
            x_p = eps * real_cpu + (1 - eps) * fake
            grad = autograd.grad(netD(x_p).mean(), x_p, create_graph=True, retain_graph=True)[
                0].view(b_size, -1)
            grad_norm = torch.norm(grad, 2, 1)
            grad_penalty = p_coeff * torch.pow(grad_norm - 1, 2)

            loss_D = torch.mean(netD(fake) - netD(real_cpu))
            loss_D.backward()
            optimizerD.step()

            for p in netD.parameters():
                p.data.clamp_(-0.01, 0.01)

            if step % n_critic == 0:
                # training netG
                noise = torch.randn(b_size, nz, 1, device=device)

                netG.zero_grad()
                fake = netG(noise)
                loss_G = -torch.mean(netD(fake))

                netD.zero_grad()
                netG.zero_grad()
                loss_G.backward()
                optimizerG.step()
            # Output training stats
            if step % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\t'
                      % (epoch, epoch_num, step, len(trainloader),
                         loss_D.item(), loss_G.item()))

            G_losses.append(loss_G.item())
            D_losses.append(loss_D.item())
            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == epoch_num-1) and (i == len(trainloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                #img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                img_list.append(fake[0])

    # save model
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    #plt.savefig(save_path+"loss.png", tight=True)
    plt.show(block=False)

    fig = plt.figure(figsize=(40, 20))
    plt.axis("off")

    plt.title("examples of generated ecg 2 lead")
    for i, img in enumerate(img_list[::int(len(img_list)/6)]):
        sub = fig.add_subplot(4, 2, i + 1, )
        sub.title.set_text(i)
        sub.plot(img.squeeze())
    #plt.savefig(save_path+"examples.png", tight=True)
    plt.show(block=False)

    params = {
        "beta1":beta1,
        "beta2":beta2,
        "p_coeff":p_coeff,
        "n_critic":n_critic,
        "lr":lr,
        "batch_size":batch_size,
        "nc":nc,
        "nz":nz,
        "ngf":ngf,
        "ndf":ndf,
        "epoch_num":epoch_num,
    }

    json.dump(params, open("./nets/params.json", 'w+'))
    torch.save(netG, './nets/wgan_gp_netG.pkl')
    torch.save(netD, './nets/wgan_gp_netD.pkl')
    plt.show()


if __name__ == '__main__':
    main()
