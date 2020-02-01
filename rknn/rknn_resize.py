import glob
import os
from PIL import Image

if __name__ == "__main__":
    images_path = glob.glob(os.path.join("data/", '*.jpg'))
    #print(images_path)
    for filename in images_path:
        temp = filename.find("_416x416")
        if temp >= 0:
            continue
        temp = filename.find("_608x608")
        if temp >= 0:
            continue
        print(filename)
        prefix = filename.split('.jpg')[0]
        #print(prefix)
        img = Image.open(filename)
        im = img.resize((416, 416), Image.ANTIALIAS)
        im.save(prefix + "_416x416.jpg")
        im = img.resize((608, 608), Image.ANTIALIAS)
        im.save(prefix + "_608x608.jpg")
