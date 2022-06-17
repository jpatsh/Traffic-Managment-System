import os
import random
import shutil

# training images and XML files directory
TRAIN_DIR = '/content/drive/MyDrive/Colab Notebooks/Final_AI/train'
# validation images and XML files directory
VALID_DIR = '/content/drive/MyDrive/Colab Notebooks/Final_AI/valid'

TEST_DIR = '/content/drive/MyDrive/Colab Notebooks/Final_AI/test'


def get_data():
    DATA_DIR = '/content/drive/MyDrive/Colab Notebooks/Final_AI/data'
    XML_DIR = '/content/drive/MyDrive/Colab Notebooks/Final_AI/xlms'
    listdir = os.listdir(DATA_DIR)
    for file in listdir:
        num = random.random()


        if num >= 0 and num <.75:    #train
            shutil.copy(DATA_DIR + '/' + file, TRAIN_DIR + '/' + file)
            xfile = file[:-4] + '.xml'
            shutil.copy(XML_DIR + '/' + xfile, TRAIN_DIR + '/' + xfile)
        elif num >.75 and num < .9:    #val
            shutil.copy(DATA_DIR + '/' + file, VALID_DIR + '/' + file)
            xfile = file[:-4] + '.xml'
            shutil.copy(XML_DIR + '/' + xfile, VALID_DIR + '/' + xfile)
        else:     #test
            shutil.copy(DATA_DIR + '/' + file, TEST_DIR + '/' + file)
            xfile = file[:-4] + '.xml'
            shutil.copy(XML_DIR + '/' + xfile, TEST_DIR + '/' + xfile)



get_data()
