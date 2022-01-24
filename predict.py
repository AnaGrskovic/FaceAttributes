import numpy as np
import pandas as pd
from keras.preprocessing import image
from tensorflow import keras
from train import CLASSES, img_height, img_width, face_extractor


IMAGES = ["rock.jpg", "melinda.jpg", "brunette.jpg", "blonde.jpg", "man_with_glasses.jpg",
          "dark_haired_woman.jpg", "smiling_man_with_beard.jpg", "mark.jpg"]


if __name__ == "__main__":
    print("Loading the model...")
    model = keras.models.load_model("saved_model")

    for IMAGE_NAME in IMAGES:
        face_extractor(IMAGE_NAME, IMAGE_NAME.split(".")[0] + "_face.jpg")
        img = image.load_img(IMAGE_NAME.split(".")[0] + "_face.jpg", target_size=(img_height,img_width,3))
        img = image.img_to_array(img)
        img = img/255

        classes = np.array(CLASSES[1:])
        probabilities = model.predict(img.reshape(1,img_height,img_width,3))
        indexes = np.argsort(probabilities[0])
        print(f"\n{IMAGE_NAME.split('.')[0]}_face.jpg")
        for i in range(len(indexes)):
            print("{}".format(classes[indexes[i]])+" ({:.3})".format(probabilities[0][indexes[i]]))
