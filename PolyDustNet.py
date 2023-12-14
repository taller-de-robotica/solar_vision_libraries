"""
Code adapted from poly_dust_detector.py
https://github.com/taller-de-robotica/1er-taller-pc/blob/main/paneles/poly_dust_detector.py
@author: crat2
"""

#LIBRARIES
import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify, unpatchify
import cv2


from termcolor import cprint

import segmentation_models as sm
from keras.models import load_model

_MC = 'yellow'
_image_size = (1792, 1024)

class Unet_Model:

    def __init__(self, model_path) -> None:
        cprint("Loading model...", _MC)
        self.model = load_model(model_path, compile=False)

    #UNET_PREDICTION FUNCTION
    def unet_prediction(self, image):
        """
        Receives an RGB image and returns a 1D numpy array of ints where
        0 - Panel
        1 - Background
        2 - Dust
        """
        patch_size = 256
        unet_model = self.model

        #Patches preprocessing
        cprint("Creating patches...", _MC)
        small_patches = []
        patches_img = patchify(image, (patch_size, patch_size, 3), step=patch_size) ##perform patchify
        for i in range(patches_img.shape[0]): 
            for j in range(patches_img.shape[1]): 
                single_patch_img = patches_img[i,j,:,:]
                small_patches.append(single_patch_img)
        small_patches =  np.array(small_patches)
        
        #Unet4 SM preprocessing
        cprint("Preprocessing with resnet50...", _MC)
        BACKBONE = 'resnet50' #define the backbone
        preprocess_input = sm.get_preprocessing(BACKBONE)
        norm_img = preprocess_input(small_patches)
        norm_img =  np.array(norm_img)
        norm_img = np.squeeze(norm_img, 1)
        
        #Prediction
        cprint("Predicting...", _MC)
        y_pred_unet = unet_model.predict(norm_img) #make the prediction
        prediction_unet = np.argmax(y_pred_unet, axis=3)[:,:,:] #from prob to int
        
        #Reconstructed image
        cprint("Reconstructing image...", _MC)
        patched_prediction = np.reshape(prediction_unet, [patches_img.shape[0], patches_img.shape[1], 
                                                        patches_img.shape[3], patches_img.shape[4]])
        reconstructed_image = unpatchify(patched_prediction, (image.shape[0], image.shape[1]))
        cprint("Finish..", _MC)
        return reconstructed_image
    
    #OUTPUT: Plotting original and GT and prediction
    def show_images(original_image, dust_image, show = False):
        """
        Displays the original image and image with detected regions.
        """
        cprint("Plotting image...", _MC)
        scale = 5
        new_size = (_image_size[0]//scale, _image_size[1]//scale)
        original = cv2.resize(original_image.astype(np.uint8), new_size)
        original = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
        
        reconstructed = cv2.resize(dust_image.astype(np.uint8), new_size) * 100
        reconstructed = cv2.merge((reconstructed,reconstructed,reconstructed))
        twins = np.concatenate((original, reconstructed), axis=1)

        if show:
            ## Funciona en Ubuntu, debe funcionar también en Windows
            cprint('Max value: ' + str(np.max(dust_image)), _MC)
            cprint("Press any key on the image to exit", _MC)
            cv2.namedWindow("Image vs Unet prediction", cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Image vs Unet prediction', twins)
            cv2.waitKey(500)
            cv2.destroyAllWindows()

        return twins