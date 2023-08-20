# ------------------------- Imports Libraries ------------------------------

from fastapi import FastAPI, File, UploadFile 
from fastapi.responses import JSONResponse
import tensorflow as tf
import uvicorn
import json 
import os
import base64
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.metrics import OneHotMeanIoU
from tensorflow.keras import backend as K

# --------------------------------------------------------------------------




# ----------------------------- Functions ----------------------------------

def dice_coeff(y_true, y_pred):
    """ Dice coefficient

    :param y_true : true values
    :param y_pred : predicted values

    :return score : return the Dice coefficient """

    smooth = 0.001
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    return score

def dice_loss(y_true, y_pred):
    """ Dice loss metric

    :param y_true : true values
    :param y_pred : predicted values

    :return loss : return the score of the Dice loss metric """

    loss = 1 - dice_coeff(y_true, y_pred)

    return loss

def total_loss(y_true, y_pred):
    """ total loss function

    :param y_true : true values
    :param y_pred : predicted values

    :return loss : return the score of the total loss function """

    loss = categorical_crossentropy(y_true, y_pred) + (3*dice_loss(y_true, y_pred))

    return loss

def IoU(y_true, y_pred):
    """ IOU metric

    :param y_true : true values
    :param y_pred : predicted values

    :return result : return the score of the IOU metric """

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.math.sigmoid(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred) - intersection
    result = intersection / denominator

    return result

def mask_categories_transformation(image):
    """ mask_categories_transformation function

    :param image : original image

    :return label : return transform 35 categories in 8 categories """

    categories = {'void': [0, 1, 2, 3, 4, 5, 6],
                  'flat': [7, 8, 9, 10],
                  'construction': [11, 12, 13, 14, 15, 16],
                  'object': [17, 18, 19, 20],
                  'nature': [21, 22],
                  'sky': [23],
                  'human': [24, 25],
                  'vehicle': [26, 27, 28, 29, 30, 31, 32, 33, -1]}

    label = np.zeros((image.shape[0], image.shape[1], 8))
    
    for i in range(-1, 34):
        if i in categories['void']:   
            label[:, :, 0] = np.logical_or(label[:, :, 0], (image == i))
        elif i in categories['flat']:
            label[:, :, 1] = np.logical_or(label[:, :, 1], (image == i))
        elif i in categories['construction']:
            label[:, :, 2] = np.logical_or(label[:, :, 2], (image == i))
        elif i in categories['object']:
            label[:, :, 3] = np.logical_or(label[:, :, 3], (image == i))
        elif i in categories['nature']:
            label[:, :, 4] = np.logical_or(label[:, :, 4], (image == i))
        elif i in categories['sky']:
            label[:, :, 5] = np.logical_or(label[:, :, 5], (image == i))
        elif i in categories['human']:
            label[:, :, 6] = np.logical_or(label[:, :, 6], (image == i))
        elif i in categories['vehicle']:
            label[:, :, 7] = np.logical_or(label[:, :, 7], (image == i))
    return label

def segmentation_color(real_mask):
    """ segmentation_color function

    :param real_mask : real mask

    :return real_mask : return transform 35 categories in 8 categories """
    labels_color = np.array([[0, 0, 0], [206, 26, 26], [2237, 165, 21], [132, 132, 132], [31, 161, 135], [255, 0, 255], [98, 200, 122], [187, 7, 247]])
    real_mask = labels_color[real_mask]
    real_mask = real_mask.astype(np.uint8)
    return real_mask

# --------------------------------------------------------------------------




# --------------------------- API Creation ---------------------------------

app = FastAPI()

# --------------------------------------------------------------------------




# ------------------------------ Model ------------------------------------

model = load_model("unet_mini_not_augmented.h5", custom_objects={'dice_coeff' : dice_coeff,
'mean_iou' : OneHotMeanIoU(num_classes=8, name='mean_iou'), 'IoU' : IoU}, compile = True)

# --------------------------------------------------------------------------




# --------------------------- POST Request ---------------------------------

@app.post("/predict_mask/")
async def predict_mask(file: UploadFile = File(...)):

    # Get the original image with (1, 256, 256, 3) shape
    contents = await file.read()
    conversion_np_array = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(conversion_np_array, cv2.IMREAD_COLOR)
    img_resized = cv2.resize(img, (256,256))
    img = img_resized.reshape(1,256,256,3)
    
    # Predict mask with model and assign color to it
    predicted_mask = model.predict(img)
    predicted_mask = predicted_mask.reshape(1,256,256,8)[0,:,:,:]
    predicted_mask = np.array(np.argmax(predicted_mask, axis=2), dtype='uint8')
    predicted_mask = segmentation_color(predicted_mask)

    # Transform image dimensions of (1, 256, 256, 3) to (256, 256, 3)
    img = np.squeeze(img)

    # Create image/predicted mask with alpha parameter
    alpha_image = cv2.addWeighted(predicted_mask.astype(np.float64), 0.5, img.astype(np.float64), 1, 0 )
    
    # Encode original image to PNG format
    _, buffer_image = cv2.imencode('.png', img)

    # Encode original image in base64 format
    original_image = base64.b64encode(buffer_image).decode("utf-8")
    
    # Encode original image/predicted mask to PNG format
    _, buffer_mask = cv2.imencode('.png', alpha_image)

    # Encode original image/predicted mask in base64 format
    mask = base64.b64encode(buffer_mask).decode("utf-8")

    # Get the path of the orignal mask
    filename = file.filename
    basename = os.path.basename(filename) 
    parts = os.path.splitext(basename)[0].split("_")
    result = "_".join(parts[:3]) 
    filname_mask = result + "_gtFine_labelIds.png"
    image_path = "img_p8/" + filname_mask

    # Assign label color to the original mask and create image/original mask with alpha parameter
    original_mask = image.img_to_array(image.load_img(image_path, color_mode='grayscale', target_size=(256, 256)))
    original_mask = np.squeeze(original_mask)
    original_mask = mask_categories_transformation(original_mask)
    original_mask = np.array(np.argmax(original_mask, axis=2), dtype='uint8')
    original_mask = segmentation_color(original_mask) 
    original_mask = cv2.addWeighted(original_mask.astype(np.float64), 0.5, img.astype(np.float64), 1, 0 )

    # Encode original mask to PNG format
    _, buffer_original_mask = cv2.imencode('.png', original_mask)

    # Encode original mask in base64 format
    original_mask = base64.b64encode(buffer_original_mask).decode("utf-8")

    # Return a Response in JSON format
    return JSONResponse(content={"prediction": mask, "image" : original_image, "real_mask": original_mask})

# --------------------------------------------------------------------------




# ----------------------- Application's Running ----------------------------

if __name__ == '__main__': 
    uvicorn.run(app, port=8000)
    
# --------------------------------------------------------------------------