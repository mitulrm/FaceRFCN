import argparse
import os
import glob

from KerasRFCN.Model.Model import RFCN_Model
from keras.preprocessing import image
from WiderFace import RFCNNConfig


def loadModel(config, modelPath):
    model = RFCN_Model(mode="inference", config=config,
                       model_dir=os.path.join(ROOT_DIR, "logs"))
    if not modelPath:
        modelPath = model.find_last()[1]
    print("Loading weights from: {}".format(modelPath))
    if modelPath and os.path.isfile(modelPath):
        # Load trained weights
        model.load_weights(modelPath, by_name=True)
    else:
        raise AssertionError("Model weight file does not exists")
    return model


def performPrediction(model, imageDir, saveDir, saveImage=False):
    """
    Prediction format:
    1. Create a folder same as the folder of image
    2. Create a text file with same name as image file (replace extension with txt)
    3. Content of file
        0_Parade_marchingband_1_20 -> Name of image file minus the extension
        2 -> Number of face
        541 354 36 46 1.000 -> [x, y, width, height, confidence] of a face
        100 242 20 35 0.98 -> [x, y, width, height, confidence] of a face
        ...

    """
    if not os.path.isdir(imageDir):
        raise AssertionError("Image directory does not exists")
    for directory in os.listdir(imageDir):
        curSavDir = os.path.join(saveDir, directory)
        if not os.path.isdir(curSavDir):
            os.makedirs(curSavDir)
        curImgDir = os.path.join(imageDir, directory)
        imagePathList = glob.glob(os.path.join(curImgDir, "*.jpg"))
        for idx, imagePath in enumerate(imagePathList):
            print("-" * 80)
            print("Processing image [{}/{}]: {}".format(idx + 1, len(imagePathList), imagePath))
            filename, ext = os.path.splitext(os.path.basename(imagePath))
            txtFilePath = os.path.join(curSavDir, filename + ".txt")
            with open(txtFilePath, "w") as fp:
                fp.write(filename)
                fp.write("\n")
                img = image.img_to_array(image.load_img(imagePath))
                prediction = model.detect([img], verbose=0)[0]
                faceList = prediction["rois"]
                scoreList = prediction["scores"]
                noOfFaces = len(faceList)
                print("Found {} faces".format(noOfFaces))
                fp.write("{}\n".format(noOfFaces))
                for face, score in zip(faceList, scoreList):
                    y1, x1, y2, x2 = face
                    width = x2 - x1
                    height = y2 - y1
                    fp.write("{} {} {} {} {}\n".format(x1, y1, width, height, score))


if __name__ == "__main__":
    ROOT_DIR = os.path.dirname(__file__)
    parser = argparse.ArgumentParser()
    parser.add_argument("--saveDir", default="prediction",
                        help="Directory where predictions should be stored")
    parser.add_argument("--imageDir", required=True,
                        help="Directory containing the images to be evaluated")
    parser.add_argument("--modelPath", default=None,
                        help="Path to model weights file (h5)")
    args = parser.parse_args()

    config = RFCNNConfig()
    model = loadModel(config, args.modelPath)
    if not os.path.isdir(args.saveDir):
        os.makedirs(args.saveDir)

    performPrediction(model, args.imageDir, args.saveDir)
