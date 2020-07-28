# #for dive
# from __future__ import print_function
# import pickle
# import os.path
# from googleapiclient.discovery import build
# from google_auth_oauthlib.flow import InstalledAppFlow
# from google.auth.transport.requests import Request
# from apiclient.http import MediaFileUpload

# #for db
# import firebase_admin
# from firebase_admin import credentials
# from firebase_admin import db

#for model,web and android
from flask import Flask, jsonify, request, render_template
from werkzeug.utils import secure_filename
# import cv2
from flask_cors import CORS
# import numpy as np
# import os
# import base64
# import shutil
# from keras.models import load_model
# from keras.preprocessing import image
# from keras.preprocessing.image import img_to_array
# from keras import backend as K

# #for sky detector
# from detectron2.engine import DefaultPredictor
# from detectron2.config import get_cfg
# #from detectron2.utils.visualizer import Visualizer
# #from detectron2.data import MetadataCatalog
# #from detectron2.modeling import build_model


# # If modifying these scopes, delete the file token.pickle.
# SCOPES = ['https://www.googleapis.com/auth/drive']

app = Flask(__name__)
CORS(app)

model_path = "static/Model/VGG16.h5" 
model_weight = "static/Model/weights_VGG16.h5"

def detect_sky(img_path):
    im = cv2.imread(img_path)

    # Create config
    cfg = get_cfg()
    path="/home/waqas/aqi/lib/python3.6/site-packages/detectron2/model_zoo/configs/COCO-PanopticSegmentation/panoptic_fpn_R_50_1x.yaml"
    cfg.merge_from_file(path)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.WEIGHTS = "static/Model/model.pkl"

    # Create predictor
    predictor = DefaultPredictor(cfg)

    # Make prediction
    outputs = predictor(im)

    for i in outputs["panoptic_seg"][1]:
        if i['category_id'] == 40:
            if i['isthing'] == False:
                return "yes"
    return "NO"

def model_predict(img_path,filename):
    
    img = cv2.imread(img_path)
    img = cv2.resize(img, (96, 96))
    img = img.astype("float") / 255.0
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)

    model = load_model(model_path)
    model.load_weights(model_weight)
    
    prediction = model.predict(img)
    K.clear_session()
    y_classes = prediction.argmax(axis=1) 
    save(y_classes,filename)
    return y_classes

def save(value,filename):

    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentialsPersonalDrive.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    service = build('drive', 'v3', credentials=creds)
    
    folder_id = '1tBCT11Y2UAkD6esgrN_39xymzz6hyMhD' 
    names = str(value) + '.png'
    file_metadata = {
        'name': [names],
        'parents': [folder_id]
    }
    read = 'uploads/' + filename
    media = MediaFileUpload(read,
                            mimetype='image/jpeg',
                            resumable=True)
    file = service.files().create(body=file_metadata,
                                        media_body=media,
                                        fields='id').execute()

# def save_to_DB(log,lat,result):
#     # Fetch the service account key JSON file contents
    
#     if (not len(firebase_admin._apps)):
#         cred = credentials.Certificate('wedapp-266618-firebase-adminsdk-yfm27-c7edafeb13.json')
#         # Initialize the app with a service account, granting admin privileges
#         firebase_admin.initialize_app(cred, {
#             'databaseURL': 'https://wedapp-266618.firebaseio.com/'
#         })
#     # As an admin, the app has access to read and write all data, regradless of Security Rules
#     ref = db.reference('wedapp-266618')
    
#     users_ref = ref.push({'longitude':log,'latitude':lat,'result':result})
    
# @app.route('/classify', methods=['GET', 'POST'])
# def classify():
#     if request.method == 'POST':
#         encodedimage = request.values['image']
#         decodedimage = base64.b64decode(encodedimage)
#         longitude = request.values['longitude']
#         latitude = request.values['latitude']
        
        
#         current = os.getcwd()
#         new = current + '/uploads'
#         os.chdir(new)
        
        
@app.route('/predict', methods=['GET', 'POST'])
def upload():
#     if request.method == "POST":

#         if request.files:
#             output={}
#             image = request.files["image"]
#             print(request.files)
#             basepath = os.path.dirname(__file__)
#             file_path = os.path.join(
#             basepath, 'uploads', secure_filename(image.filename))
#             print(image.filename)
#             image.save(file_path)
#             # Make prediction
#             preds = model_predict(file_path,image.filename)
#         return str(preds)        
    return "Error"
    
if __name__ == '__main__':
    app.run( )#port=8080, host='0.0.0.0',threaded=False)
