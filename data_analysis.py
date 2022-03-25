from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
import IPython.terminal.debugger as Debug

# root dataset directory
data_dir = "datasets/nyu/training/basement_0001a"

# load all files
color_files = os.listdir(data_dir)
depth_files = os.listdir(data_dir + '/depth')

# filter all images
color_img_names = sorted(list(filter(lambda elem: ".jpg" in elem, color_files)))
depth_img_names = sorted(list(filter(lambda elem: ".png" in elem, depth_files)))

# load the images
for idx, (c_n, d_n) in enumerate(zip(color_img_names, depth_img_names)):
    # load the image
    color_img = Image.open(data_dir + "/" + c_n)
    depth_img = Image.open(data_dir + "/depth/" + d_n)

    # convert to array
    color_img_arr = np.array(color_img)
    depth_img_arr = np.array(depth_img) / 5000  # todo: unclear why the data is divided by 5000?: it is a ratio.

    # show the data
    fig, arr = plt.subplots(1, 2)
    arr[0].set_title(f"{idx}-RGB")
    arr[0].imshow(color_img_arr)
    arr[1].set_title(f"{idx}-depth")
    arr[1].imshow(depth_img_arr)
    print(depth_img_arr.min(), depth_img_arr.max())
    plt.show()

    if idx == 10:
        break



