import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from os.path import join
import pickle, random
import config as cfg

class BirdsDataset(Dataset):

    def __init__(self, computer="pc", type="train"):

        # what direcorty the data set is at
        if computer == "pc":
            path = join(r"D:\Misc\Datasets\birds", type)
        elif computer == "vm":
            path = join("birds", type)
        elif computer == "cb":
            path = join(r"drive/My Drive/Colab Notebooks/birds", type)

        # get images
        #self.Images = np.memmap("data/birdImages", dtype="uint8", mode="r", shape=(11788, 304, 304, 3))
        self.Images = np.memmap("data/hr_images", dtype="uint8", mode="r", shape=(8855, 304, 304, 3))

        # load the embeddings
        #self.Embeddings = np.load("data/CNN_RNN_Embeddings.npy", allow_pickle=True)
        pickleIn = open("data/char-CNN-RNN-embeddings.pickle", "rb")
        self.Embeddings = np.array(pickle.load(pickleIn, encoding="latin1"))

    def prepEmbeddings(self, embeddings):

        idx = random.randint(0, embeddings.shape[0] - 1)
        embedding = embeddings[idx]
        return torch.Tensor(embedding)

    # oringal way the researches got the embeddings
    def prepEmbeddings2(self, embeddings, sample_num = 4):

        embedding_num, _ = embeddings.shape

        # Take every sample_num captions to compute the mean vector
        sampled_embeddings = []

        randix = np.random.choice(embedding_num, sample_num, replace=False)

        e_sample = embeddings[randix, :]
        e_mean = np.mean(e_sample, axis=0)
        sampled_embeddings.append(e_mean)

        sampled_embeddings_array = np.array(sampled_embeddings)
        sampled_embeddings_array = torch.Tensor(np.squeeze(sampled_embeddings_array))

        return sampled_embeddings_array

    # prepares the image
    def prepImage(self, image):

        # random crop
        transformedImage = np.zeros((cfg.stage3Res, cfg.stage3Res, cfg.channels), dtype=np.uint8)
        oriSize = image.shape[0]

        h1 = np.floor((oriSize - cfg.stage3Res) * np.random.random())
        w1 = np.floor((oriSize - cfg.stage3Res) * np.random.random())
        croppedImage = image[int(w1):int(w1 + cfg.stage3Res), int(h1):int(h1 + cfg.stage3Res), :]

        # random flip horizontally
        if random.random() > 0.5:
            transformedImage = np.fliplr(croppedImage)
        else:
            transformedImage = croppedImage

        # Rescale -1 to 1
        transformedImage = transformedImage / 127.5 - 1

        # convert to tensors and swap to channels first
        transformedImage = torch.Tensor(transformedImage).permute(2, 0, 1)

        return transformedImage

    def __getitem__(self, i):

        return self.prepImage(self.Images[i]), self.prepEmbeddings(self.Embeddings[i])

    # gets length of the dataset
    def __len__(self):

        return self.Images.shape[0]

    def getFixedData(self, rc):

        assert rc == 4, "There's gonna be a problem here"

        imgs = []
        embs = []

        # researches dataset: 1992, 5881, 7561, 1225
        nums = [744, 744, 744, 744, # blue bird
                8201, 8201, 8201, 8201, # red bird
                2666, 2666, 2666, 2666, # yellow bird
                887, 887, 887, 887] # rainbow bird

        for i in range(rc * rc):

            # get and crop the image to 256, 256
            imgs.append(self.Images[nums[i], 48:304, 48:304, :])

            # get the embeddings
            embs.append(self.prepEmbeddings(self.Embeddings[nums[i]]))

        # convert images to uint8 tensor
        imgs = torch.Tensor(imgs)
        imgs = imgs.type(torch.uint8).cuda()

        # tile and convert embeddings to tensor
        embs = torch.stack(embs).cuda()
        embs = self.tile(embs[:rc * rc], rc)

        return imgs, embs

    # make sure every row with n column have the same embeddings
    def tile(self, x, n):

        for i in range(n):
            for j in range(1, n):
                x[i * n + j] = x[i * n]
        return x

# add noise for data augmentation, todo: implement
if __name__ == "__main__":

    import cv2

    imageDir = r"C:\Users\Hayden\Downloads\eee.jpg"
    img = cv2.imread(imageDir, cv2.IMREAD_COLOR)

    for i in range(1, 10):

        noise = np.random.normal(0, 2, img.shape)
        print(f"std: {i}, max: {noise.max()}, min: {noise.min()}")

        noisy_img = img + noise
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

        cv2.imshow("img", noisy_img)
        cv2.waitKey()
        cv2.destroyAllWindows()
