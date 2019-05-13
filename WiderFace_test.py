"""
Keras RFCN
Copyright (c) 2018
Licensed under the MIT License (see LICENSE for details)
Written by parap1uie-s@github.com
"""

import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from KerasRFCN.Model.Model import RFCN_Model
from keras.preprocessing import image
from WiderFace import RFCNNConfig


def Test(model, loadpath, savepath):
    assert not loadpath == savepath, "loadpath should'n same with savepath"

    model_path = model.find_last()[1]
    # Load trained weights (fill in path to trained weights here)
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    if os.path.isdir(loadpath):
        for idx, imgname in enumerate(os.listdir(loadpath)):
            if not imgname.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
                continue
            print(imgname)
            imageoriChannel = np.array(plt.imread(os.path.join(loadpath, imgname))) / 255.0
            img = image.img_to_array(image.load_img(os.path.join(loadpath, imgname)))
            TestSinglePic(img, imageoriChannel, model, savepath=savepath, imgname=imgname)

    elif os.path.isfile(loadpath):
        if not loadpath.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
            print("not image file!")
            return
        print(loadpath)
        imageoriChannel = np.array(plt.imread(loadpath)) / 255.0
        img = image.img_to_array(image.load_img(loadpath))
        filename = os.path.basename(loadpath)
        TestSinglePic(img, imageoriChannel, model, savepath=savepath, imgname=filename)


def TestSinglePic(image, image_ori, model, savepath, imgname):
    r = model.detect([image], verbose=1)[0]
    print(r)

    def get_ax(rows=1, cols=1, size=8):
        _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
        return ax

    ax = get_ax(1)

    assert not savepath == "", "empty save path"
    assert not imgname == "", "empty image file name"

    for box in r['rois']:
        y1, x1, y2, x2 = box
        p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                              alpha=0.7, linestyle="solid",
                              edgecolor="lime", facecolor='none')
        ax.add_patch(p)
    ax.imshow(image_ori)
    plt.savefig(os.path.join(savepath, imgname), bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    ROOT_DIR = os.getcwd()
    parser = argparse.ArgumentParser()

    parser.add_argument('--loadpath', required=False,
                        default="images/",
                        metavar="evaluate images loadpath",
                        help="evaluate images loadpath")
    parser.add_argument('--savepath', required=False,
                        default="result/",
                        metavar="evaluate images savepath",
                        help="evaluate images savepath")

    config = RFCNNConfig()
    args = parser.parse_args()
    model = RFCN_Model(mode="inference", config=config,
                       model_dir=os.path.join(ROOT_DIR, "logs"))
    if not os.path.isdir(args.savepath):
        os.makedirs(args.savepath)
    Test(model, args.loadpath, args.savepath)
