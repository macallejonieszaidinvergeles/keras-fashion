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
        
        print("file:",file)
        print("file_url:",file_url[1:])
        image_path = file_url[1:]
        print("la ruta del archivo:",image_path)

        img = cv2.imread(image_path, 0)  # read image as gray scale    
        img = cv2.bitwise_not(img)             # < ----- bitwise_not
        print(img.shape)   # (300, 231)

        img = infer_prec(img, 28)  # call preprocess function 
        print(img.shape)   # (1, 28, 28, 1)
        y_pred = model_keras.predict(img)
        print(y_pred)
        prediction = tf.argmax(y_pred, axis=-1).numpy()
        print("la prediccion es:",prediction)
        print("0: T-shirt/top 1: Trouser 2: Pullover 3: Dress 4: Coat 5: Sandal 6: Shirt 7: Sneaker 8: Bag 9: \n Ankle boot /n0: camiseta/top 1: pantalón 2: jersey 3: vestido 4: abrigo 5: sandalia 6: camisa 7: tenis 8: bolso 9: botín")

        dict_items = {0: "camiseta/top" ,1: "pantalón", 2: "jersey", 3: "vestido", 4: "abrigo", 5: "sandalia",
         6: "camisa", 7: "tenis", 8: "bolso", 9: "botín"}

        prediction = str(prediction)
        prediction = int(prediction[1:2])

        for item in dict_items:
            if prediction == item:
                result = dict_items[item]
        print("result:",result)
     
        return render(request, "inicio.html",{"prediction":result,'file_url': file_url[1:]})
        

    return render(request, "inicio.html")
