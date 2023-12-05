import sys
import pandas as pd
import cv2 as cv
from utils.wrappers import *

img_dir = sys.argv[1]

model = Wrapper('instances/optimized_model.h5')

img = cv.imread(img_dir, cv.IMREAD_GRAYSCALE)
img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
img = cv.resize(img, (224, 224), interpolation=cv.INTER_AREA)

pred = model.predict(img)

for key, value in pred.items():
    print(f'{key}: {value:.4f}')
