import cv2
import numpy as np
from modeling import FaceNet
import argparse

my_parser = argparse.ArgumentParser()


args = my_parser.parse_args()

width = height = 224

model = FaceNet()
model(np.zeros((1, width, height, 3)))
model.load_weights("FaceNet.h5")

img = cv2.imread('test.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img / 255.0
img = cv2.resize(img, (width, height))
img = img.reshape(1, width, height, 3)

result = model.predict(img)

pred = np.argmax(result)

faces= ["Ali Khamenei","Angelina Jolie","Barak Obama","Behnam Bani","Donald Trump","Emma Watson","Han Hye Jin","Kim Jong Un",
"Leyla Hatami","Lionel Messi","Michelle Obama","Morgan Freeman","Queen Elizabeth","Scarlett Johanson"]

print('This is ',faces[pred])