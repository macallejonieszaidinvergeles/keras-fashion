from django.http import HttpResponse
from django.template import Template, Context
from django.template.loader import get_template
from django.shortcuts import render
import os
import pickle
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import h5py
import cv2
from django.core.files.storage import FileSystemStorage



def infer_prec(img, img_size):
    img = tf.expand_dims(img, -1)       # from 28 x 28 to 28 x 28 x 1 
    img = tf.divide(img, 255)           # normalize 
    img = tf.image.resize(img,          # resize acc to the input
             [img_size, img_size])
    img = tf.reshape(img,               # reshape to add batch dimension 
            [1, img_size, img_size, 1])
    return img 


def inicio(request):


    model_keras = keras.models.load_model("./my_model.h5")
    # print("model keras:",model_keras)

    if request.POST:

        upload = request.FILES['file']
        fss = FileSystemStorage()
        file = fss.save(upload.name, upload)
        file_url = fss.url(file)

        # file = request.POST["file"]
        
        # image_path = file_url
        image_path = "chanclas-unisex-adidas-adilette-35543_jcFFoGe.jpg"
        print("la ruta del archivo:",image_path)

        # # GRIS
        img = cv2.imread(image_path, 0)   # read image as gray scale    
        print("img shape",img.shape)   # (300, 231)
        img = infer_prec(img, 28)  # call preprocess function 
        y_pred = model_keras.predict(img)
        print(y_pred)
        prediction = tf.argmax(y_pred, axis=-1).numpy()
        print("la prediccion es:",prediction)
        

    return render(request, "inicio.html")
