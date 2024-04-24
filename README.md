# EECS 592 Project

## Introduction

Our project is to explore the correlation between the emotions evoked in images and corresponding color adjustment applied, aiming to recommend an appropriate color adjustment option for image based on their emotional content. Our project contains two parts: images classification based on emotional content and CNN models for predicting color adjustment for photo editing.

## Dataset

MIT-Adobe FiveK Dataset: https://data.csail.mit.edu/graphics/fivek/

## Files

`EECS592_model.ipynb`: This is an ipynb file that includes all of our implementation. The file includes how we construct and train CNN models for each emotion type, how we test the model, and how we evaluate the model using SSIM and PSNR. The CNN models we trained are at the CNN Test section. The GAN test section is just a try.

`model.py`: This is the file that includes CNN model implementation, but we already add the code in this file to `EECS592_model.ipynb` for larger dataset training.

`classification.py`: This is the file for getting CLIP model classification results. The code in this file has been added to CNN model implementation.

## Reference

https://www.hackersrealm.net/post/extract-features-from-image-python

https://github.com/yuukicammy/mit-adobe-fivek-dataset
