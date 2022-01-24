import os
import pathlib
import numpy as np
import pandas as pd
import cv2 as cv
import imutils
from PIL import Image
import tensorflow as tf
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from tqdm import tqdm


DATASET_PATH = "/home/plavy/.keras/datasets/celeb_faces"
DATASET_PATH = "/home/student_proj/celeb_dataset"
DATASET_PATH = "/home/plavy/celeb_dataset"
CLASSES = ["image_id", "Male", "Eyeglasses", "Bald", "Smiling", "No_Beard", "Attractive"]
TEST_SIZE = 0.2
EPOCHS = 160
img_height = 180
img_width = 180


# Haar Cascade, center face in an image
# https://github.com/opencv/opencv/tree/master/data/haarcascades
def face_extractor(origin, destination):
    fc = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    img = cv.imread(origin, 1)
    if img is not None:
        img = imutils.resize(img, width=img_width)
        H, W, _ = img.shape
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        face_coord = fc.detectMultiScale(gray, 1.2, 10, minSize=(50, 50))
        if len(face_coord) == 1:
            X, Y, w, h = face_coord[0]
        elif len(face_coord) == 0:
            return None
        else:
            max_val = 0
            max_idx = 0
            for idx in range(len(face_coord)):
                _, _, w_i, h_i = face_coord[idx]
                if w_i * h_i > max_val:
                    max_idx = idx
                    max_val = w_i * h_i
                else:
                    pass

                X, Y, w, h = face_coord[max_idx]
        img_cp = img[
                 max(0, Y - int(0.15 * h)): min(Y + int(1.15 * h), H),
                 max(0, X - int(w * 0.15)): min(X + int(1.15 * w), W)
                 ].copy()
        cv.imwrite(destination, img_cp)


if __name__ == "__main__":

    # Make numpy values easier to read.

    np.set_printoptions(precision=3, suppress=True)

    # Read the dataset
    # Facial attribute detection using Deep learning
    # https://towardsdatascience.com/real-time-multi-facial-attribute-detection-using-transfer-learning-and-haar-cascades-with-fastai-47ff59e36df0

    dataset_csv = pd.read_csv(
        DATASET_PATH + "/list_attr_celeba.csv")

    dataset_csv = dataset_csv[CLASSES]

    data_dir = pathlib.Path(DATASET_PATH + "/img_align_celeba/img_align_celeba")

    images = list(data_dir.glob('*.jpg'))
    NUM_OF_IMAGES = len(images)
    NUM_OF_IMAGES = 24000

    dataset_csv = dataset_csv.iloc[0:NUM_OF_IMAGES]

    dataset_csv = dataset_csv.replace(-1,0)

    # Run Haar Cascade

    print("Running Haar Cascade...")
    for i in tqdm(range(NUM_OF_IMAGES)):
        IMAGE_PATH = DATASET_PATH + "/img_align_celeba/img_align_celeba/" + dataset_csv["image_id"][i]
        if not os.path.isfile(IMAGE_PATH.split(".")[0] + "_face.jpg"):
            face_extractor(IMAGE_PATH, IMAGE_PATH.split(".")[0] + "_face.jpg")

    # Load images
    # Multi-Label Image Classification Model
    # https://www.analyticsvidhya.com/blog/2019/04/build-first-multi-label-image-classification-model-python/

    images = []
    print("Loading images...")
    for i in tqdm(range(NUM_OF_IMAGES)):
        IMAGE_PATH = DATASET_PATH + "/img_align_celeba/img_align_celeba/" + dataset_csv["image_id"][i]
        if os.path.isfile(IMAGE_PATH.split(".")[0] + "_face.jpg"):
            IMAGE_PATH = IMAGE_PATH.split(".")[0] + "_face.jpg"
        img = image.load_img(IMAGE_PATH, target_size=(img_height, img_width, 3))
        img = image.img_to_array(img)
        img = img/255   # if you zero center the data, the model converges faster
        images.append(img)
    images = np.array(images)

    labels = np.array(dataset_csv.drop(['image_id'], axis=1))

    train_images, test_images, train_labels, test_labels = \
        train_test_split(images, labels, random_state=42, test_size=TEST_SIZE)

    # Train the model

    print("Training the model...")
    model = tf.keras.Sequential()
    model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=(img_height,img_width,3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(6, activation='sigmoid'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, validation_data=(test_images, test_labels), batch_size=32, epochs=EPOCHS)

    print("Training completed")

    # Save the model

    print("Saving the model...")
    model.save("saved_model")

    print("Model saved. ")
    print("You can now run predict.py")

