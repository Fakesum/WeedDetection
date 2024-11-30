import os, sys
from PIL import Image
import matplotlib.pyplot as plt
import shutil

try:
    shutil.rmtree("runs")
except FileNotFoundError:
    pass

file_name = sys.argv[1]
os.system(f"yolo predict model='best.onnx' source={file_name}")

plt.imshow(Image.open(f"./runs/detect/predict/{".".join(os.path.basename(file_name).split(".")[:-1])}.jpg"))
plt.axis('off') 
plt.show()