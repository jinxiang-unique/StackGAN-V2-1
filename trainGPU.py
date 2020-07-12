import torch
from torch.autograd import Variable
from torch.optim import Adam
import numpy as np
import time, cv2, models
from dataset import BirdsDataset, DataLoader
import config as cfg

class StackGANv2():
    def __init__(self):
     
        # get the gpu
        if torch.cuda.is_available():
            torch.cuda.set_device(torch.device("cuda:0"))
        else:
            raise Exception("Get cuda you loser")

        # get generator
        self.generator = models.G_NET().cuda()

        # get discriminator
        self.discriminator = []
        if cfg.StageNum > 0:
            self.discriminator.append(models.D_NET64().cuda())
        if cfg.StageNum > 1:
            self.discriminator.append(models.D_NET128().cuda())
        if cfg.StageNum > 2:
            self.discriminator.append(models.D_NET256().cuda())
        if cfg.StageNum > 3:
            self.discriminator.append(models.D_NET512().cuda())
        if cfg.StageNum > 4:
            self.discriminator.append(models.D_NET1024().cuda())

        # Initialize weights
        self.generator.apply(models.weights_init)

        for i in range(len(self.discriminator)):
            self.discriminator[i].apply(models.weights_init)

        # Loss function
        self.loss = torch.nn.BCELoss().cuda()

        # Optimizers
        self.genOptimizer = Adam(self.generator.parameters(), lr=cfg.generatorLR, betas=(0.5, 0.999))

        self.disOptimizer = []
        for i in range(len(self.discriminator)):
            opt = Adam(self.discriminator[i].parameters(), lr=cfg.discriminatorLR, betas=(0.5, 0.999))
            self.disOptimizer.append(opt)
            
        # load datasets
        self.trainData = DataLoader(BirdsDataset(computer = "vm"), batch_size=cfg.batchSize, shuffle=True, drop_last=True, num_workers=0)

    def train(self, epochs, batchSize, saveInterval):

        # updates per epoch
        batchs = self.trainData.__len__()

        # Sample noise as generator input
        noise = Variable(torch.Tensor(batchSize, cfg.zDim)).cuda()
        fixedNoise = Variable(torch.Tensor(cfg.rowsColums * cfg.rowsColums, cfg.zDim).normal_(0, 1)).cuda()

        # fixed data for output
        fixedData = self.trainData.dataset.getFixedData(cfg.rowsColums)

        # Adversarial ground truths
        real = Variable(torch.Tensor(batchSize).fill_(1)).cuda()
        fake = Variable(torch.Tensor(batchSize).fill_(0)).cuda()

        # sizes to downscale to for the discriminator
        sizes = [64, 128, 256]

        # train
        for epoch in range(epochs):

            # varables to get averages
            totalGenLoss = 0.0
            totalKLloss = 0.0
            totalDisLoss = 0.0

            # get start time
            start = time.time()

            for batch, data in enumerate(self.trainData):
                
                # Get Training Data
                ####################

                # get training data
                images = data[0].cuda()
                embeddings = data[1].cuda()

                # generate normal noise
                noise.data.normal_(0, 1)

                # Train Discriminator
                ######################

                # Generate a batch of images
                genImgs, mu, logvar = self.generator(noise, embeddings)

                mean = mu.detach()

                # discriminate
                for i in range(len(self.discriminator)):
                    self.discriminator[i].zero_grad()
                    imgs = torch.nn.functional.interpolate(images, size=sizes[i], mode="nearest")

                    # real
                    logits, uncondLogits = self.discriminator[i](imgs, mean)
                    realLoss = self.loss(logits, real) + self.loss(uncondLogits, real)

                    # wrong
                    logits, uncondLogits = self.discriminator[i](torch.roll(imgs, 1, 0), mean)
                    wrongLoss = self.loss(logits, fake) + self.loss(uncondLogits, real)

                    # fake
                    logits, uncondLogits = self.discriminator[i](genImgs[i].detach(), mean)
                    fakeLoss = self.loss(logits, fake) + self.loss(uncondLogits, fake)
                  
                    # calulate loss mean
                    disLoss = realLoss + wrongLoss + fakeLoss
                    totalDisLoss += disLoss

                    # optimize
                    disLoss.backward()
                    self.disOptimizer[i].step()

                # Train Generator
                ##################

                self.generator.zero_grad()

                # generate
                genLoss = 0
                for i in range(len(self.discriminator)):

                    logits = self.discriminator[i](genImgs[i], mean)
                    genLoss += self.loss(logits[0], real) + self.loss(logits[1], real)

                totalGenLoss += genLoss

                # optimize
                KLloss = models.KLloss(mu, logvar) * cfg.KL
                totalKLloss += KLloss
                genLoss = genLoss + KLloss
                genLoss.backward()
                self.genOptimizer.step()

                # print progress bar
                self.progressbar(epoch, epochs, batch, batchs, start)

            # get end time
            end = time.time()
            duration = round(end - start, 1)

            # print the average losses and the duration
            print (f"{epoch+1} / {epochs-1} epoch, dis loss: {totalDisLoss / batchs}, gen loss: {totalGenLoss / batchs}, {totalKLloss / batchs}, duration: {duration}s")

            # if at save interval save generated image samples and generator
            if epoch % saveInterval == 0:
                self.sampleImages(epoch, fixedNoise, fixedData)
                torch.save(self.generator.state_dict(), f"models/stackGAN-V2_Generator{epoch}.pyt")

        print ("FINISHED!!")

    # print a progressbar
    def progressbar(self, epoch, epochs, batch, batchs, start):

        out = f"{epoch + 1} / {epochs - 1} epoch, :"
        per = round(batch / batchs * 100, 1)
        num = per // 2
        for h in range(num):
            out += "#"
        for s in range(50 - num):
            out += " "
        timer = round(time.time() - start, 1)
        print (f"{out}: {per}%, {timer}s", end="\r", flush=True)

    # save images every number of epochs
    def sampleImages(self, epoch, noise, data):

        # rows and columns
        rc = cfg.rowsColums

        # generate images
        genImgs, mu, logvar = self.generator(noise, data[1])

        for i in range(cfg.StageNum):
            genImgs[i] = genImgs[i].detach()

        # save generated images to disk
        self.saveImages(genImgs, data[0], rc, "Train_", epoch)

    # post processes the images for saving to disk
    def postprocessing(self, tensor):

        # Rescale images from -1, 1 to 0, 1
        tensor.add_(1).div_(2)

        # scale to 0, 255
        tensor = tensor.mul_(255).type(torch.uint8)

        # make channels last
        tensor = tensor.permute(0, 2, 3, 1)

        return tensor

    # saves the images to disk in a grid
    def saveImages(self, genImgs, trainImgs, rc, name, epoch):

        # gap between images
        gap = 10

        res = cfg.stage3Res

        # canvas to draw the images on
        canvasSizeY = res * rc + (rc * gap)
        canvasSizeX = canvasSizeY * 3 + (res + gap) + gap
        canvas = torch.zeros((canvasSizeY + gap, canvasSizeX, 3), dtype=torch.uint8).cuda()

        # upscale the stage1 and stage2 images to stage3 size
        genImgs[0] = torch.nn.functional.interpolate(genImgs[0], scale_factor=4, mode="nearest")
        genImgs[1] = torch.nn.functional.interpolate(genImgs[1], scale_factor=2, mode="nearest")

        # postprocess
        for i in range(cfg.StageNum):
            genImgs[i] = self.postprocessing(genImgs[i])

        # draw all the images on the canvas
        gapX = gap
        gapY = gap
        cnt = 0
        for i in range(rc):
            # draw real image
            canvas[gapY:gapY+res, gapX:gapX+res] = trainImgs[i * rc]

            for j in range(rc):

                # draw stage 1, 2 and 3
                for l in range(cfg.StageNum):

                    gapX += res + gap
                    canvas[gapY:gapY+res, gapX:gapX+res] = genImgs[l][cnt]

                cnt += 1

            gapY += res + gap
            gapX = gap

        # save image to disk, images generated are RGB
        cv2.imwrite(f"images/{name}{epoch}.png", canvas.cpu().numpy()) # cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

if __name__ == "__main__":
    
    import os
    os.mkdir("data")
    os.mkdir("images")
    os.mkdir("models")
    
    gan = StackGANv2()
    gan.train(cfg.epochs + 1, cfg.batchSize, cfg.saveInterval)
