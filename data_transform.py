from xml.etree import ElementTree as ET
import os
from PIL import Image



path = os.path.expanduser("/content/drive/MyDrive/Colab Notebooks/Final_AI/xlms")
listdir = os.listdir(path)

for file in listdir:

    tree = ET.parse(path + '/' + file)
    root = tree.getroot()
    

    for member in root.findall('object'):

        if member.find('name').text == 'licence' or member.find('name').text =='license-plate' or member.find('name').text =='LP':
            member.find('name').text = "license"
            print(member.find('name').text) # for debugging only
    tree.write(path + '/' + file)



path = os.path.expanduser("/content/drive/MyDrive/Colab Notebooks/Final_AI/data")
listdir = os.listdir(path)
for file in listdir:
    if file[-4:] == '.png':
        im = Image.open(path + '/' + file)
        rgb_im = im.convert('RGB')
        rgb_im.save(path + '/' + file[:-4] + '.jpg')
        os.remove(path + '/' + file) 
