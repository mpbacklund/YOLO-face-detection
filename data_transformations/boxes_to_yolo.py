import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def main():
    face_images_db = np.load('archive/face_images_updated.npz')['face_images']
    bb_df = pd.read_csv('bb_csv.csv')

    # get the image height, image width, and number of images
    (imHeight, imWidth, numImages) = face_images_db.shape

    split_index = int(0.8 * numImages)

    # do the training dataset
    for index in range(split_index):
        path = "../yolo/datasets/train/"
        saveImage(index, face_images_db, path)
        bb_list = convertToYoloFormat(index, bb_df, imHeight, imWidth)
        saveToTxt(index, bb_list, path)

    # do the validation dataset
    for index in range(split_index, numImages):
        path = "../yolo/datasets/test/"
        saveImage(index, face_images_db, path)
        bb_list = convertToYoloFormat(index, bb_df, imHeight, imWidth)
        saveToTxt(index, bb_list, path)

def saveImage(imageNumber, face_db, path):
    image = face_db[:,:,imageNumber] 

    fig, ax = plt.subplots(figsize=(6,6))

    # Display the image
    ax.imshow(image, cmap='gray')

    ax.axis('off')

    imageName = "img" + str(imageNumber) + ".png"
    imagePath = os.path.join(path, imageName)
    plt.savefig(imagePath)
    plt.close(fig)

def convertToYoloFormat(imageNumber, bb_df, imWidth, imHeight):
    leftEye = f"0 {bb_df.at[imageNumber, 'left_eye_x'] / imWidth} {bb_df.at[imageNumber, 'left_eye_y'] / imHeight} {bb_df.at[imageNumber, 'left_eye_width'] / imWidth} {bb_df.at[imageNumber, 'left_eye_height'] / imHeight}\n"
    rightEye = f"1 {bb_df.at[imageNumber, 'right_eye_x'] / imWidth} {bb_df.at[imageNumber, 'right_eye_y'] / imHeight} {abs(bb_df.at[imageNumber, 'right_eye_width'] / imWidth)} {bb_df.at[imageNumber, 'right_eye_height'] / imHeight}\n"
    nose = f"2 {bb_df.at[imageNumber, 'nose_x'] / imWidth} {bb_df.at[imageNumber, 'nose_y'] / imHeight} {bb_df.at[imageNumber, 'nose_width'] / imWidth} {bb_df.at[imageNumber, 'nose_height'] / imHeight}\n"
    mouth = f"3 {bb_df.at[imageNumber, 'mouth_x'] / imWidth} {bb_df.at[imageNumber, 'mouth_y'] / imHeight} {bb_df.at[imageNumber, 'mouth_width'] / imWidth} {bb_df.at[imageNumber, 'mouth_height'] / imHeight}\n"

    return [leftEye, rightEye, nose, mouth]

def saveToTxt(imageNumber, bb_list, path):
    imageName = "img" + str(imageNumber) + ".txt"
    imagePath = os.path.join(path, imageName)
    
    with open(imagePath, 'w') as file:
        file.writelines(bb_list)

if __name__ == "__main__":
    main()