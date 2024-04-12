# Kevin Wang Levy's Lab Bonus Quest B
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#Change imgDir based on the path your images are stored VVV
imgDir = r"C:\Users\KKwan\Desktop\Levy's Lab\PreliminaryGenerativeHistoPath-master\cyto2label_public\test"
imgFiles = sorted(os.listdir(imgDir))

def calcNCratio(mask):
    nucleus, cytoplasm, total = 0, 0, 0
    for pixel in mask:
        if pixel == (0, 0, 255):
            nucleus += 1
        elif pixel == (0, 255, 0):
            cytoplasm += 1
        total += 1
    if total != 0:
        return nucleus / total
    return 0

NCratios = []
for imgFile in imgFiles:
    imgPath = os.path.join(imgDir, imgFile)
    image = Image.open(imgPath)
    width, _ = image.size                       
    mask = list(image.getdata())
    #retrive full img size, divide by two to only use mask

    NCratios.append(calcNCratio(mask))

avgNCratio = sum(NCratios) / len(NCratios)

threshold = 0.3
if avgNCratio > threshold:
    print("The population is most likely cancerous.")
else:
    print("The population is not likely cancerous. ")
