# reference: https://www.hackersrealm.net/post/extract-features-from-image-python
# reference: https://github.com/yuukicammy/mit-adobe-fivek-dataset
#from torch.utils.data.dataloader import DataLoader
#from dataset.fivek import MITAboveFiveK
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Reshape, Conv2D, MaxPooling2D, Flatten, Input, Conv2DTranspose
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import tensorflow
import matplotlib.pyplot as plt
import os
import cv2

def CNN_model():
  model = Sequential([
      Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
      MaxPooling2D((2,2)),
      Conv2D(64, (3, 3), activation='relu'),
      MaxPooling2D((2,2)),
      Conv2D(64, (3, 3), activation='relu'),
      Flatten(),
      Dense(64, activation='relu'),
      Reshape((224, 224))
  ])
  return model

def fully_connected_model():
    model = Sequential([
       Dense(64, activation='relu', input_shape=(224, 224)),
       Dense(32, activation='relu'),
       Dense(4096, activation='linear'),
       Reshape((224,224))
    ])
    return model

def CNN_model_2():
    input_tensor = Input(shape=(224, 224, 3))
    conv2D1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_tensor)
    conv2D2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2D1)

    tranpose1 = Conv2DTranspose(64, (3, 3), strides=(1, 1), activation='relu', padding='same')(conv2D2)
    output_tensor = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(tranpose1)
    model = Model(inputs=input_tensor, outputs=output_tensor)
    return model

def main():
    # data_loader = DataLoader(
        # MITAboveFiveK(root="path-to-dataset-root", split="debug", download=False, experts=["a"]),
        # batch_size=None)

    # read from train dataset
    path_edited = "train/expert/"
    path_original = "train/original/"
    VGG_model = VGG16()
    VGG_model = Model(inputs=VGG_model.inputs, outputs=VGG_model.layers[-2].output)

    CLIP_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    CLIP_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    emotion_list = ["amusement", "awe", "contentment", "excitement", "anger", "disgust", "fear", "sadness"]

    # {"amusement":[value1, value2], "awe":[value1, value2]}
    color_shift_values = {}
    original_images_dict = {}
    # initialize the color_shift_values
    for emotion in emotion_list:
        color_shift_values[emotion] = []
        original_images_dict[emotion] = []
    edited_list = os.listdir(path_edited)
    original_list = os.listdir(path_original)
    num = len(original_list)
    for i in range(10):
        original_img = cv2.imread(path_original+original_list[i])
        original_img = cv2.resize(original_img, (224, 224))
        # classification using CLIP
        inputs_clip = CLIP_processor(text=["image evokes amusement", "image evokes awe", "image evokes contentment", "image evokes excitement", "image evokes anger", "image evokes disgust", "image evokes fear", "image evokes sadness"], images=original_img, return_tensors="pt", padding=True)
        outputs_clip = CLIP_model(**inputs_clip)
        logits_per_image = outputs_clip.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        # get the highest prob in probs list
        highest_prob = torch.max(probs)
        highest_prob_index = 0
        #print(probs)
        for i in range(8):
            if (probs[0][i].item()==highest_prob):
                highest_prob_index = i
        image_class = emotion_list[highest_prob_index]
        original_images_dict[image_class].append(original_img)

        edited_img = cv2.imread(path_edited+edited_list[i])
        edited_img = cv2.resize(edited_img, (224, 224))
        #original_img = img_to_array(original_img)
        #edited_img = img_to_array(edited_img)
        original_color = cv2.cvtColor(original_img, cv2.COLOR_BGR2LAB)
        #print("original_color")
        #print(original_color)
        edited_color = cv2.cvtColor(edited_img, cv2.COLOR_BGR2LAB)
        #print("edited_color")
        #print(edited_color)
        # get color_shift_value
        color_diff = cv2.subtract(edited_color, original_color)
        color_shift_values[image_class].append(color_diff)

        #original_img = img_to_array(original_img)
        #original_img = original_img.reshape((1, original_img.shape[0], original_img.shape[1], original_img.shape[2]))
        #original_img = preprocess_input(original_img)
        # extract original feature using pretrained VGG16 model
        #original_feature = VGG_model.predict(original_img, verbose=0)
        #original_features[image_class].append(original_feature)

        #edited_img = img_to_array(edited_img)
        #edited_img = edited_img.reshape((1, edited_img.shape[0], edited_img.shape[1], edited_img.shape[2]))
        #edited_img = preprocess_input(edited_img)
        # extract edtied feaure using pretrained VGG16 model
        #edited_feature = VGG_model.predict(edited_img, verbose=0)
        #edited_features[image_class].append(edited_feature)

    model_dict = {}
    for emotion in emotion_list:
        if(len(original_images_dict[emotion])!=0):
            #print("len")
            #print(len(original_images_dict[emotion]))
            X = np.array(original_images_dict[emotion])
            print(X.shape)
            #print(X)
            y = np.array(color_shift_values[emotion])
            print(y.shape)
            #print(y)

            # build model with fully connected layer
            model_dict[emotion] = CNN_model_2()

            #print(model_dict[emotion].summary())
            model_dict[emotion].compile(optimizer='adam', loss='mean_squared_error')
            model_dict[emotion].fit(X, y, epochs=10, batch_size=32)
        
    # get input from test dataset
    test_img = cv2.imread("test.png")
    test_img = cv2.resize(test_img, (224, 224))
    # classify using CLIP
    logits_per_image = outputs_clip.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    # get the highest prob in probs list
    highest_prob = torch.max(probs)
    #print(probs)
    highest_prob_index = 0
    for i in range(8):
        if (probs[0][i].item()==highest_prob):
            highest_prob_index = i
    image_class = emotion_list[highest_prob_index]
    if(image_class in model_dict.keys()):
        # get features
        #test_img = img_to_array(test_img)
        #test_img = test_img.reshape((1, test_img.shape[0], test_img.shape[1], test_img.shape[2]))
        #test_img = preprocess_input(test_img)
        # extract original feature using pretrained VGG16 model
        # test_feature = VGG_model.predict(test_img, verbose=0)
        #test_X = np.array(test_img)
        input_img = np.expand_dims(test_img, axis=0)
        predicted_color_adjustment_value = model_dict[image_class].predict(input_img)
        print(predicted_color_adjustment_value)
        test_color = cv2.cvtColor(test_img, cv2.COLOR_BGR2LAB)
        predicted_color_adjustment_value = np.reshape(predicted_color_adjustment_value, (1, 224, 224, 3))
        edited_color = test_color+predicted_color_adjustment_value
        edited_color = np.clip(edited_color, 0, 255)
        print("edited_color")
        print(edited_color)
        print("shape")
        print(edited_color.shape)
        new_edited_img = cv2.cvtColor(edited_color[0], cv2.COLOR_LAB2BGR)
        #predicted_feature = tensorflow.reshape(np.array(predicted_feature), shape=(1,1,4096))
        #decoder = Sequential([
            #Dense(256, activation='relu', input_shape=(1,4096)),
            #Dense(512, activation='relu'),
            #Dense(224*224, activation='sigmoid'),
            #Reshape((224, 224))
        #])
        #decode_image = decoder.predict(predicted_feature)
        plt.imshow(new_edited_img)
        plt.show()
    else:
        print("need larger dataset")


if __name__ == "__main__":
    main()