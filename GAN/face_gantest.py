from GAN.face_model import WGAN_GPModel
import numpy as np
import pandas as pd
import cv2


def srcnn_wgan_train():
    '''
       Preprocessing for dataset
       '''
    # Read  data set (Train data from CSV file)
    csvimagedata = pd.read_csv('face_data.csv')
    imagedata = csvimagedata.iloc[:, :].values
    # shuffle imagedata and maskdata together
    perm = np.arange(len(csvimagedata))
    np.random.shuffle(perm)
    imagedata = imagedata[perm]
    images = []
    for i in range(len(imagedata)):
        image = cv2.imread(imagedata[i][0], cv2.IMREAD_COLOR)
        images.append(image)
    # Extracting images and labels from given data
    imagesdata=np.asarray(images)
    imagesdata = imagesdata.astype(np.float)
    # Normalize from [0:255] => [0.0:1.0]
    train_images = imagesdata / 255.

    train_images = np.reshape(train_images, (-1, 96, 96, 3))
    srcnn = WGAN_GPModel(96, 96, 3, 128, 64)
    srcnn.train(train_images,
                "model\\ganface",
                "test\\log", learning_rate=0.0001, train_epochs=50000)


def srcnn_wgan_predict():
    true_img = cv2.imread("../1.bmp")
    true_img = cv2.cvtColor(true_img, cv2.COLOR_BGR2GRAY)
    true_img = cv2.resize(true_img, (256, 256))
    scale_factor = 2
    init_width, init_height = true_img.shape[0], true_img.shape[1]
    srcnn = WGAN_GPModel(init_height * scale_factor, init_width * scale_factor)
    predictimage, intermediate_img = srcnn.prediction(
        "model\\gansr", true_img, scale_factor)
    cv2.imwrite("srcnn_gan.jpg", predictimage)
    cv2.imwrite("intermediate_img.jpg", intermediate_img)


def main(argv):
    if argv == 3:
        srcnn_wgan_train()
    if argv == 4:
        srcnn_wgan_predict()


if __name__ == "__main__":
    main(3)
