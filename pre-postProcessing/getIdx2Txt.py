import numpy as np
import pickle, os

if __name__ == '__main__':

    # get all the file names
    with open(r"birds\train\filenames.pickle", 'rb') as f:
        filenames = pickle.load(f)

    # get all text description paths
    embPaths = []
    for path, subdirs, files in os.walk("text_c10"):
        for name in files:
            embPaths.append(os.path.join(path, name))

    def getName(path):
        path = path.split("\\")
        return path[len(path) - 1][:-4]

    filenames.sort()
    embPaths.sort()

    Idx2Txt = []
    count = 0
    for i in range(len(embPaths)):

        if getName(embPaths[i]) == filenames[count].split("/")[1]:

            Idx2Txt.append(embPaths[i])
            count += 1

    assert len(filenames) == len(Idx2Txt), "it's never that easy"

    np.save("data/Idx2Txt", Idx2Txt)
    print ("saved and done")
