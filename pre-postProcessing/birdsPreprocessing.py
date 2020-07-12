import numpy as np
import cv2, os
import CNN_RNN_Embeddings

if __name__ == '__main__':

    # set the class that will be converting the text into embeddings
    embeder = CNN_RNN_Embeddings.Embeddings()

    # set train data varables
    Images = np.memmap("data/birdImages", dtype="uint8", mode="w+", shape=(11788, 304, 304, 3))
    Embeddings = []

    # paths
    imgDir = r"D:\Misc\Datasets\birds\CUB_200_2011\images"
    txtDir = r"D:\Misc\Datasets\birds\text_c10"

    # get all image paths
    imgPaths = []
    for path, subdirs, files in os.walk(imgDir):
        for name in files:
            imgPaths.append(os.path.join(path, name))

    # get all text description paths
    embPaths = []
    for path, subdirs, files in os.walk(txtDir):
        for name in files:
            embPaths.append(os.path.join(path, name))

    assert len(imgPaths) == len(embPaths), "They should be the same length."
    
    # gets the name from the path, method: cpu > memory
    def getName(path):
        path = path.split("\\")
        return path[len(path) - 1][:-4]

    # check if all the files match up
    for i in range(len(imgPaths)):
        assert getName(imgPaths[i]) == getName(embPaths[i]), "arrays missmatch"

    updateProg = len(imgPaths) // 1_000

    # preprocess
    for i in range(len(imgPaths)):

        if i % updateProg == 0:
            per = round(i / len(imgPaths) * 100, 1)
            print (f"{per}%", end="\r", flush=True)

        # load image
        image = cv2.imread(imgPaths[i], cv2.IMREAD_COLOR)
        
        # resize
        image = cv2.resize(image, (304, 304), interpolation=cv2.INTER_CUBIC) # INTER_NEAREST

        # write image to page file
        Images[i] = image

        # load sentence descriptions from the text file
        with open(embPaths[i], "r") as f:
            txt = f.read().splitlines()

            # convert the sentence descriptions into text embedings
            sentenceEmbeddings = embeder.embedSentenceBatch(txt)
            Embeddings.append(sentenceEmbeddings)

    np.save("data/CNN_RNN_Embeddings", Embeddings)
    print ("saved and done")
