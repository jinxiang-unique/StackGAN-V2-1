import torch
import numpy as np
from dataset import BirdsDataset, DataLoader
import cv2, random, pickle
import models
import config as cfg

if __name__ == '__main__':

    # load in generator
    generator = models.G_NET().cuda()
    generator.load_state_dict(torch.load(r"models/Birds 1024 embeddings/stackGAN-V2_Generator.pyt"))
    generator.eval()

    batchs = 16
    
    noise = torch.Tensor(batchs, cfg.zDim).cuda()

    # load embeddings
    pickleIn = open(r"C:\Users\Hayden\Downloads\AIBirds\data\char-CNN-RNN-embeddings.pickle", "rb")
    embeddings = np.array(pickle.load(pickleIn, encoding="latin1"))

    idx2txt = np.load("data/Idx2Txt.npy", allow_pickle=True)

    def prepEmbeddings(embeddings):

        idx = random.randint(0, embeddings.shape[0] - 1)
        embedding = embeddings[idx]
        return torch.Tensor(embedding)

    # post processes the images for saving to disk
    def postprocessing(tensor):

        # Rescale images from -1, 1 to 0, 1
        tensor.add_(1).div_(2)

        # scale to 0, 255 and make type uint8
        tensor = tensor.mul_(255).type(torch.uint8)

        # make channels last
        tensor = tensor.permute(0, 2, 3, 1)

        return tensor

    while True:

        # get random index
        idx = random.randint(0, embeddings.shape[0] - 1)

        # get the text embedings
        emb = prepEmbeddings(embeddings[idx])

        emb = emb.repeat(batchs, 1).cuda()

        # generate normal noise
        noise.data.normal_(0, 1)

        # Generate images
        with torch.no_grad():
            genImgs, mu, logvar = generator(noise, emb)

        # print the text
        with open(idx2txt[idx], "r") as f:
            descriptions = f.read().splitlines()
            print (descriptions[random.randint(0, 9)])

        # convert gpu tensor to img tensor array
        imgs = postprocessing(genImgs[2])

        # gap between images
        gap = 10
        rc = 4
        res = cfg.stage3Res

        # canvas to draw the images on
        canvasSizeY = res * rc + (rc * gap)
        canvasSizeX = canvasSizeY + gap
        canvas = torch.zeros((canvasSizeY + gap, canvasSizeX, 3), dtype=torch.uint8).cuda()

        # draw all the images on the canvas
        gapX = gap
        gapY = gap
        cnt = 0
        for i in range(rc):

            for j in range(rc):

                canvas[gapY:gapY+res, gapX:gapX+res] = imgs[cnt]
                gapX += res + gap

                cnt += 1

            gapY += res + gap
            gapX = gap

        #for i in range (batchs):

        cv2.imshow("img", cv2.cvtColor(canvas.cpu().numpy(), cv2.COLOR_BGR2RGB))
        cv2.waitKey()
        cv2.destroyAllWindows()
    