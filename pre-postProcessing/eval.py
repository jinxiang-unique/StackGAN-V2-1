import torch
from torch.autograd import Variable
from torch.optim import Adam
import numpy as np
import time, cv2, models
from dataset import BirdsDataset, DataLoader
import config as cfg

# WIP
# todo: validation, scoring, seed

epochs = 10

# load in the model
generator = models.G_NET().cuda()
generator.load_state_dict(torch.load("models/stackGAN-V2_Generator0.pyt"))

# get evaluator
discriminator = models.eval256().cuda()
discriminator.apply(models.weights_init)

# Loss function and optimizer
loss = torch.nn.BCELoss().cuda()
disOptimizer = Adam(discriminator.parameters(), lr=cfg.discriminatorLR, betas=(0.5, 0.999))

# load datasets
trainData = DataLoader(BirdsDataset(computer = "pc"), batch_size=cfg.batchSize, shuffle=True, drop_last=True, num_workers=0)

# Adversarial ground truths
real = Variable(torch.Tensor(cfg.batchSize).fill_(1)).cuda()
fake = Variable(torch.Tensor(cfg.batchSize).fill_(0)).cuda()

# Sample noise as generator input
noise = torch.Tensor(cfg.batchSize, cfg.zDim).cuda()

# updates per epoch
batchs = trainData.__len__()
evalBatchs = batchs * cfg.batchSize

# print a progressbar
def progressbar(epoch, epochs, batch, batchs, start):

    out = f"{epoch + 1} / {epochs} epoch, :"
    per = round(batch / batchs * 100, 1)
    num = int(per / 2)
    for h in range(num):
        out += "#"
    for s in range(50 - num):
        out += " "
    timer = round(time.time() - start, 1)
    print (f"{out}: {per}%, {timer}s", end="\r", flush=True)

# train
for epoch in range(epochs):

    # varables to get averages
    totalDisLoss = 0.0
    totalImg = 0.0
    totalGen = 0.0

    start = time.time()

    for batch, data in enumerate(trainData):

        # Get Training Data
        ####################

        # get training data
        images = data[0].cuda()
        embeddings = data[1].cuda()

        # generate normal noise
        noise.data.normal_(0, 1)

        # Generate a batch of images
        genImgs, mu, logvar = generator(noise, embeddings)

        #noise = torch.rand(cfg.batchSize, 3, 256, 256).cuda() * 2 - 1
        
        # Train Evaluator
        ######################

        discriminator.zero_grad()

        # real
        logits = discriminator(images)
        realLoss = loss(logits, real)

        for i in range(len(logits)):
            totalImg += logits[i].detach().item()

        # fake
        logits = discriminator(genImgs[2].detach()) # genImgs[2].detach()
        fakeLoss = loss(logits, fake)
        
        for i in range(len(logits)):
            totalGen += logits[i].detach().item()

        # calulate loss mean
        disLoss = realLoss + fakeLoss
        totalDisLoss += disLoss

        # optimize
        disLoss.backward()
        disOptimizer.step()

        # print progress bar
        progressbar(epoch, epochs, batch, batchs, start)

    # get end time
    end = time.time()
    duration = round(end - start, 1)

    # print the average losses and the duration
    print (f"{epoch + 1} / {epochs} epoch, dis loss: {totalDisLoss / batchs}, gen eval: {totalGen / evalBatchs} / 0, real eval: {totalImg / evalBatchs} / 1, duration: {duration}s")
