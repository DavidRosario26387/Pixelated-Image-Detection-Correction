
# Pixelated Image Detection and Correction

This project focuses on detecting and restoring pixelated images using machine learning models. The detection model leverages the MobileNetV2 architecture fine-tuned for binary classification, while the restoration model employs the ESPCN (Efficient Sub-Pixel Convolutional Neural Network) for super-resolution.


## Key Features

__High Accuracy Detection:__ MobileNetV2 architecture, fine-tuned for binary classification, achieving 98.25% validation accuracy.

__Effective Restoration:__ ESPCN model enhances image quality by reducing pixelation artifacts.

__Balanced Dataset:__ Used DIV2K dataset and Wide Screen Image Dataset from Kaggle. Pixelated versions were created by downscaling and upscaling the images by 5x or 6x using nearest neighbor or bilinear interpolation randomly.

__Optimized Training:__ Adam optimizer, early stopping, trained over 13 epochs with batch size 32.

__Evaluation Metrics:__ Evaluated with accuracy, F1-score, PSNR, and SSIM. F1 score evaluation was done on an independent dataset, Caltech 256 Image Dataset, apart from the one used in training.
## Introduction

Pixelation can degrade the visual quality of images, making them appear blocky and unclear. This project addresses the challenge of detecting and restoring pixelated images to enhance visual quality and clarity.

## Datasets used

Our dataset consists of 1,140 high-resolution images for which we have used Used DIV2K dataset and 'Wide Screen Image' Dataset from Kaggle. To create a balanced dataset for the binary classifier, we generated pixelated versions of each image. This involved downsizing the original images by factors of 5x or 6x and subsequently upscaling them by the same factors using either nearest neighbor or bilinear interpolation methods.
## Model Architecture

__Pixelated Image Detection__

We utilized the MobileNetV2 architecture, pre-trained on ImageNet, and fine-tuned it with additional layers for binary classification.

![architecture](https://github.com/DavidRosario26387/Pixelated-Image-Detection/assets/116174510/20f4e3c5-1ac5-452b-81dc-20cbf7f9b9e1)

__Image Restoration__

For image restoration, we used the ESPCN model, which is designed to perform single-image super-resolution.
## Preprocessing

__Pixelated Image Detection__
Each image was preprocessed to fit the model's input size of 224x224 pixels. Instead of resizing, which could distort the image and affect the detection task, we opted for cropping to preserve the original texture and details essential for accurate pixelation detection, as for detection the complete image is not needed; rather, the texture is what is important.

__Image Restoration__
Images were downscaled by a factor of 4 to reduce pixelation artifacts and then fed into the ESPCN model for super-resolution.
## Training

The detection model was built using the __MobileNetV2 architecture__ pre-trained on the ImageNet dataset. It was configured with an Adam optimizer (learning rate set to 1e-4) and trained using binary cross-entropy loss. We used an early stopping mechanism to prevent overfitting, training the model over 13 epochs with a batch size of 32. The dataset was split into training and validation sets using an 80-20 split, with stratification to maintain class balance.

![training2](https://github.com/DavidRosario26387/Pixelated-Image-Detection/assets/116174510/b01ad792-0023-47f9-bce9-e1c4e6b653ec)

![training](https://github.com/DavidRosario26387/Pixelated-Image-Detection/assets/116174510/f16ae209-6bbb-4646-affe-a7f83af58e2e)

## Evaluation

__Pixelated Image Detection__

The detection model achieved a peak validation accuracy of 98.25%. 

![validation accuracy](https://github.com/DavidRosario26387/Pixelated-Image-Detection/assets/116174510/b474a51c-3f1d-4c40-8280-ee9d29a1a558)

The __F1 score__ is 0.96, evaluated on 1,200 images from the Caltech 256 Image Dataset (600 images in both classes). 

![F1 score](https://github.com/DavidRosario26387/Pixelated-Image-Detection/assets/116174510/7b0aec03-c66c-45d0-b7e8-de8ed6b21562)
<br>

Confusion Matrix

![Confusion matrix](https://github.com/DavidRosario26387/Pixelated-Image-Detection/assets/116174510/f340b979-67b0-45fd-91bf-2e8d9f4f1383)

The model processed images at an average __FPS__ of 41.00.

![FPS](https://github.com/DavidRosario26387/Pixelated-Image-Detection/assets/116174510/01b48e5b-cf4a-4007-808e-ccd5b3482172)

---

__Pixelated Image Correction__

The ESPCN model effectively reduced pixelation artifacts and improved visual clarity. 
The average __PSNR__ improved from 25.61 dB to 27.92 dB for a sample of 20 images

<img src="https://github.com/DavidRosario26387/Pixelated-Image-Detection-Correction/assets/116174510/328b39e7-3e3a-4c32-879b-35653b6ef83d" width="750" height="400">
<img src="https://github.com/DavidRosario26387/Pixelated-Image-Detection-Correction/assets/116174510/9303fc03-7325-4e14-8732-8c3c5f305b37" width="750" height="400">
<br>

The mean __SSIM__ increased from 0.7997 to 0.8315 for a sample of 20 images. 

<img src="https://github.com/DavidRosario26387/Pixelated-Image-Detection-Correction/assets/116174510/83fc2b69-8ade-4ea1-b57c-d0acaec1d639" width="750" height="400">
<br>
The ESPCN model processed 1920x1080 images at an average __FPS__ of 5.59, indicating limitations in real-time image restoration tasks.

![fpsespn](https://github.com/DavidRosario26387/Pixelated-Image-Detection-Correction/assets/116174510/3664829d-4a0f-4aa0-b4ea-ba3952256f65)

## Sample output

__Pixelated Image Detection:__

![detection](https://github.com/DavidRosario26387/Pixelated-Image-Detection-Correction/assets/116174510/4be072e6-ce6a-437c-871c-1c100e838c8b)

__Pixelated Image Correction:__

![res1](https://github.com/DavidRosario26387/Pixelated-Image-Detection-Correction/assets/116174510/f1dd1f9c-759e-4d11-9019-788414c6ebc9)
Zoomed-Out Comparison of Cropped Sections: 
![res2](https://github.com/DavidRosario26387/Pixelated-Image-Detection-Correction/assets/116174510/5f87cb7c-165a-4772-b7e5-dae041041243)
![res3](https://github.com/DavidRosario26387/Pixelated-Image-Detection-Correction/assets/116174510/2242f08d-acf1-4e2d-99ce-60ac549b15b8)

---

![res4](https://github.com/DavidRosario26387/Pixelated-Image-Detection-Correction/assets/116174510/d793d6c4-5043-4042-89a7-607c32985877)
Zoomed-Out Comparison of Cropped Sections: 
![res5](https://github.com/DavidRosario26387/Pixelated-Image-Detection-Correction/assets/116174510/7075c8ae-41eb-4b7d-8105-e35d44320cf9)
![res6](https://github.com/DavidRosario26387/Pixelated-Image-Detection-Correction/assets/116174510/1c10b64f-8e95-47a7-96ee-02c042efeda1)

---

![res7](https://github.com/DavidRosario26387/Pixelated-Image-Detection-Correction/assets/116174510/e6062a1d-c4be-4726-b880-888806ac124d)
Zoomed-Out Comparison of Cropped Sections: 
![res8](https://github.com/DavidRosario26387/Pixelated-Image-Detection-Correction/assets/116174510/c8481163-c2c5-4c15-be7a-af8c28095779)

---

## Acknowledgement

 For Image Correction ESPCN Model weight downloaded from: https://github.com/fannymonori/TF-ESPCN<br>
 Author: Fanny Monori<br>
 TensorFlow ESPCN implementation for super-resolution<br>

## Conclusion
This project demonstrates the high accuracy of the MobileNetV2 model in detecting pixelation and the effectiveness of the ESPCN model in Correcting pixelated images.
