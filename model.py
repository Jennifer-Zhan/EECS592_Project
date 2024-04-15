# reference: https://www.hackersrealm.net/post/extract-features-from-image-python
# reference: https://github.com/yuukicammy/mit-adobe-fivek-dataset
from torch.utils.data.dataloader import DataLoader
from dataset.fivek import MITAboveFiveK
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

def main():
    data_loader = DataLoader(
        MITAboveFiveK(root="path-to-dataset-root", split="debug", download=True, experts=["a"]),
        batch_size=None)

    VGG_model = VGG16()
    VGG_model = Model(inputs=VGG_model.inputs, outputs=VGG_model.layers[-2].output)

    CLIP_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    CLIP_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    emotion_list = ["image evokes amusement", "image evokes awe", "image evokes contentment", "image evokes excitement", "image evokes anger", "image evokes disgust", "image evokes fear", "image evokes sadness"]

    original_features = []
    edited_features = []
    for item in data_loader:
        original_image_path = item["files"]["dng"]
        original_img = load_img(original_image_path, target_size=(224, 224))
        original_img = img_to_array(original_img)
        original_img = original_img.reshape((1, original_img.shape[0], original_img.shape[1], original_img.shape[2]))
        original_img = preprocess_input(original_img)
        # extract original feature using pretrained VGG16 model
        original_feature = VGG_model.predict(original_img, verbose=0)
        original_features.append(original_feature)

        edited_image_path = item["files"]["tiff16"]["a"]
        edited_img = load_img(edited_image_path, target_size=(224, 224))
        edited_img = img_to_array(edited_img)
        edited_img = edited_img.reshape((1, edited_img.shape[0], edited_img.shape[1], edited_img.shape[2]))
        edited_img = preprocess_input(edited_img)
        # extract edtied feaure using pretrained VGG16 model
        edited_feature = VGG_model.predict(edited_img, verbose=0)
        edited_features.append(edited_feature)


    X = np.array(original_features)
    print(X)
    y = np.array(edited_features)
    print(y)

    # build model with fully connected layer
    fully_connected_model = Sequential([
        Dense(64, activation='relu', input_shape=(1, 4096)),
        Dense(32, activation='relu'),
        Dense(4096, activation='linear')
    ])

    print(fully_connected_model.summary())
    fully_connected_model.compile(optimizer='adam', loss='mean_squared_error')
    fully_connected_model.fit(X, y, validation_split=0.2, epochs=10, batch_size=32)


if __name__ == "__main__":
    main()