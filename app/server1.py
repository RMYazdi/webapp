

import keras
import tensorflow as tf
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Lambda, MaxPool2D, BatchNormalization
from keras.utils import np_utils
from keras.utils import model_to_dot
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras import models, layers, optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils import class_weight
from keras.optimizers import SGD, RMSprop, Adam, Adagrad, Adadelta, RMSprop
from keras.models import Sequential, model_from_json
from keras.layers import Activation,Dense, Dropout, Flatten, Conv2D, MaxPool2D,MaxPooling2D,AveragePooling2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.layers.merge import concatenate
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
import os
from glob import glob
import random
import cv2
import pandas as pd
import numpy as np
import itertools
import sklearn
import scipy
import skimage
from skimage.transform import resize
from sklearn import model_selection
from sklearn.model_selection import train_test_split, learning_curve,KFold,cross_val_score,StratifiedKFold
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
from keras.applications.inception_v3 import decode_predictions









from starlette.applications import Starlette
from starlette.responses import HTMLResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
from pathlib import Path
import uvicorn, aiohttp, asyncio
import sys, numpy as np

path = Path(__file__).parent
model_file_url="https://dl.boxcloud.com/d/1/b1!v-jnNeLciccMYG_VCP3qfMbZpko9lPnC2u4MVHzYW4g7s0MgamYSt7YOB2fZ6L9FsdletonblqOzIQUT08YmloLIMaeMNWIVwAww6bTvUXYtv8B6hk1ohlHsmXfS8oIPQBnSscaBUeJHPBk0tnv8PaQH49C1wh4S1CmZso4dZJY00bkcvfvhKTnIj3VMK2p_0MhwWl8IBAFZ2Ya6ztJTXm866vEMp3F3ku3Y7qNzrzboqClpXZHTv8OsJT0uo-Ho_K0gH5bC0YMl0jYqzmZLrqp35oiU47cOitRf-MlDmM2DoQSVgB-MW_d3vDBXCOqdLK2BJ4mNvqNYGL1LLBu0AdEJ7R2ZOuzk96TGySEigBsQzQbZw1yPc9tQQk14afLD5JcPTDNxmlLcU-0cCzdwq1Hn8KnINoE5r-AQbHpCphfDj3tiOgZv1anPY3DU-r2ck1mSC5w6mF6OIUeXHbw439tKI3MVHDxRz8lLrEuB5tlNcE3aobz57FOifjpMd48VASFJqA713JYeaM2smsbSHSgNinQu16HfX8odYKYHPX8v5iaBVp6Jjeli4L3CH-ZLaWZN_pFA8uiki_BSg3Ycivt4drtBD7Sixo1qe_RZsKpIX6xdgn-hYwS_B9m1KjKXQsqAkFAljvY7dCgHGQW121rK2-aKlPp5rRj_FHIVgu56E7WC1wVgb-GxAqd7ayz6TiviWxptg6V7yWIpDr4CDv8WFGLKcIP2E_y3gfmp2HGSPioJPH83_LeODSf10Jtw3FiY8lWU62_Rr8ulTgB6sEohdfO3xBimygota6AdFQZ43OAuZ_ZgRiEi0UsrLwv6LjeWg1cnKDtV3AC8lgezINhMfblzPqCLJpEU6AXms417TSAbFuaQD9ZRVSvX8PKfnbkz9FE-l2RrXyte3g53SR9Qa_6gDUOyvv1knPjQUCf5JSCNEXb9QM0YjSiFy6B5nF7iwF3FEm3RwpSVNcS54kmYx8n1YfYD1SMTAjz4ukOgkmy4ct9tkc0dTldGZb-PFBS0EuouXnu4L_EybJK7BZ6AMZ1GpFByIVmmGDMFjGokJhPsF-zkf0PohAJDJC4DvGagshluyOPTD1lIAqF4Sa6L_KMd70QCCAUx9BV7TWDMtJ-gyp9ZtOSxR0EFjnNEsSIEpFZ44FygPKUolW5M-FEv8YC8n0LW1wC0i1lTFEgP0BdTo-YWTO-yHEHCH_MRY2IxjFScqJQRDt1hlM3uYkATvo-fuYMU94WYPD52foHtpMYrWc0W2OLOQY5E8P6OYhRxA64kMHP0ajbALKKbkRAiVCEKN8E64Y2oh7-VOfisWubMzLVWuQ../download"
model_file_name = 'model'

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

MODEL_PATH = path/'models'/f'{model_file_name}.h5'
IMG_FILE_SRC = '/tmp/saved_image.png'















# Defining Hyperparameters and model structure
image_size=224
target_dims = (image_size,image_size,3) # add channel for RGB
number_classes = 2
number_epoch=150
batch_size=32
learningRate=0.0001
patience=100
# optimizer = keras.optimizers.Adam(lr=0.001)
optimizer = keras.optimizers.RMSprop(lr=learningRate)









#Generate Model 

from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50

from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

def Model_build():
  # Importing the important libraries

  def single_branch(name_modifier):#add name_modifier cause we use a single model(like InceptionV3) mutilple times. 
    
    base_model = InceptionV3(include_top=False, weights='imagenet')
    for layer in base_model.layers:
      layer.trainable = True
    
    for layer in base_model.layers:
      layer._name = layer.name + str(name_modifier)


    x = base_model.output
    
    x = GlobalAveragePooling2D()(x)

    # x = Dense(1024, activation='relu')(x)

    # predictions = Dense(number_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=x)
    
    # Training only top layers i.e. the layers which we have added in the end
    return model






  Frontal_branch=single_branch('one')
  Lateral_branch=single_branch('two')
  Oblique_branch=single_branch('three')



  # plot_model(Oblique_branch)


  combinedInput = concatenate([Frontal_branch.output, Lateral_branch.output,Oblique_branch.output])

  predictions = Dense(number_classes, activation='softmax',name='visualize_layer')(combinedInput)



  model=Model(inputs=[Frontal_branch.input,Lateral_branch.input,Oblique_branch.input],outputs=predictions)
  return model

# model.summary()

# plot_model(model)






  # model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=["accuracy"])




  









































async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_model():
 
  
    
    #UNCOMMENT HERE FOR CUSTOM TRAINED MODEL
    await download_file(model_file_url, MODEL_PATH)
    model=Model_build()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=["accuracy"])
    model.load_weights(MODEL_PATH)

    
#     model = load_model(MODEL_PATH) # Load your Custom trained model
    # model._make_predict_function()
#     model = ResNet50(weights='imagenet') # COMMENT, IF you have Custom trained model
    return model

# Asynchronous Steps
loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_model())]
model = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route("/upload", methods=["POST"])
async def upload(request):
    data = await request.form()
    img_bytes = await (data["file"].read())
    with open(IMG_FILE_SRC_Frontal, 'wb') as f: f.write(img_bytes)
      
      
    data = await request.form()
    img_bytes = await (data["file2"].read())
    with open(IMG_FILE_SRC_Lateral, 'wb') as f: f.write(img_bytes)
      
      
    data = await request.form()
    img_bytes = await (data["file3"].read())
    with open(IMG_FILE_SRC_Oblique, 'wb') as f: f.write(img_bytes)
      
      
 
      
    return model_predict(IMG_FILE_SRC_Frontal,IMG_FILE_SRC_Lateral,IMG_FILE_SRC_Oblique, model)





def model_predict(img_path_frontal,img_path_Lateral,img_path_Oblique, model):
    result = [];     
    
    
    def img_prep(img_path):
      img = cv2.imread(img_path)
      img = skimage.transform.resize(img, (224, 224))
      img_file=img/224.0
      x = np.asarray(img_file)
      x=x.reshape((1,224,224,3))
      return(x)
    
    x=[img_prep(img_path_frontal), img_prep(img_path_Lateral),img_prep(img_path_Oblique)]
    y_pred = model.predict(x)
    
    
    Label_dict={0:"Patient",1:"Normal"}
    label=Label_dict[np.argmax(y_pred, axis=1)[0]]
    accuracy=max(y_pred[0])
    
    result.append((label,accuracy))
    result_html1 = path/'static'/'result1.html'
    result_html2 = path/'static'/'result2.html'
    result_html = str(result_html1.open().read() +str(result) + result_html2.open().read())
    return HTMLResponse(result_html)







# def model_predict(img_path, model):
#     result = []; img = image.load_img(img_path, target_size=(224, 224))
#     x = preprocess_input(np.expand_dims(image.img_to_array(img), axis=0))
#     predictions = decode_predictions(model.predict(x), top=3)[0] # Get Top-3 Accuracy
#     for p in predictions: _,label,accuracy = p; result.append((label,accuracy))
#     result_html1 = path/'static'/'result1.html'
#     result_html2 = path/'static'/'result2.html'
#     result_html = str(result_html1.open().read() +str(result) + result_html2.open().read())
#     return HTMLResponse(result_html)

@app.route("/")
def form(request):
    index_html = path/'static'/'index.html'
    return HTMLResponse(index_html.open().read())

if __name__ == "__main__":
    if "serve" in sys.argv: uvicorn.run(app, host="0.0.0.0", port=8080)
