from PIL import Image
import os

frames_dir = '/home/as4296/palmer_scratch/vit/paf/train/frames'
# print(os.listdir(frames_dir))
fname = os.listdir(frames_dir)[3]               # pick the first .png
img = Image.open(os.path.join(frames_dir, fname))
w, h = img.size                               # PIL gives (width, height)
print(f"Image size: width={w}, height={h}")