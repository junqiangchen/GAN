from GAN.mnist_model import GANModel, WGANModel
import numpy as np
import pandas as pd
import cv2


def gantrain():
    '''
    Preprocessing for dataset
    '''
    # Read  data set (Train data from CSV file)
    csvimagedata = pd.read_csv('../data/mnist_train.csv')
    images = csvimagedata.iloc[:, :].values
    images = images[:, 1:]
    imagedata = images.astype(np.float)
    # shuffle imagedata and maskdata together
    perm = np.arange(len(csvimagedata))
    np.random.shuffle(perm)
    imagedata = imagedata[perm]
    # Extracting images and labels from given data
    imagedata = imagedata.astype(np.float)
    # Normalize from [0:255] => [-1.0:1.0]
    train_images = (imagedata / 255.)

    srcnn = GANModel(28, 28)
    srcnn.train(train_images,
                "model\\gan",
                "log", 0.0005, train_epochs=10000, batch_size=128)


def ganpredict():
    srcnn = GANModel(28, 28, 1, 100)
    predictimage = srcnn.prediction("model\\gan", 50)
    for i in range(len(predictimage)):
        cv2.imwrite(str(i + 1) + "srcnn.jpg", predictimage[i])


def wgantrain():
    '''
    Preprocessing for dataset
    '''
    # Read  data set (Train data from CSV file)
    csvimagedata = pd.read_csv('../data/mnist_train.csv')
    images = csvimagedata.iloc[:, :].values
    images = images[:, 1:]
    imagedata = images.astype(np.float)
    # shuffle imagedata and maskdata together
    perm = np.arange(len(csvimagedata))
    np.random.shuffle(perm)
    imagedata = imagedata[perm]
    # Extracting images and labels from given data
    imagedata = imagedata.astype(np.float)
    # Normalize from [0:255] => [-1.0:1.0]
    train_images = (imagedata / 255.)
    train_images = np.reshape(train_images, (-1, 28, 28, 1))

    srcnn = WGANModel(28, 28)
    srcnn.train(train_images,
                "model\\wgan-mnist",
                "log", 0.0001, train_epochs=500000)


def wganpredict():
    srcnn = WGANModel(28, 28, 1, 100)
    predictimage = srcnn.prediction("model\\wgan-mnist", 5)
    for i in range(len(predictimage)):
        cv2.imwrite(str(i + 1) + "srcnn.jpg", predictimage[i])


def main(argv):
    if argv == 1:
        gantrain()
    if argv == 2:
        ganpredict()
    if argv == 3:
        wgantrain()
    if argv == 4:
        wganpredict()


if __name__ == "__main__":
    main(3)