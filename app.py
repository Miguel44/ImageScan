from __future__ import division, print_function
import os
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
# Flask
from flask import Flask, flash, request, redirect, render_template, jsonify, json
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
app = Flask(__name__)

app.secret_key = b'_5#y2L"ncaGDJiK'
# Folder were photos will be uploaded
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Puts restriction on image upload.
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Path were the ML model is store
# MODEL_PATH = 'models'
# model = load_model(MODEL_PATH)
# model._make_predict_function()
from keras.applications.resnet50 import ResNet50
model = ResNet50(weights='imagenet')
model.save('models')


def allowed_file(filename):
    """
    Checks that the type of file is allowed.
    :param filename: Name of the file to check.
    :return: True if file type is allowed and false if the file type is not allowed.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def model_predict(img_path, model):
    """
    Using an ML model, an image is process and is use as input for the model,
    for image classification.

    :param img_path: The path were the image is stored.
    :param model: The pre-trained model that is use for image classification.
    :return: Numpy array.
    """
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='caffe')

    preds = model.predict(x)
    return preds


def convert_json(pred):
    """
    Converts prediction list to JSON format.
    :param pred: The prediction list.
    :return: A JSON string.
    """
    objectlst = []
    emptlst = {'objects': []}
    for i in range(5):
        objectlst.append(str(pred[0][i][1]))
    emptlst['objects'] = objectlst
    jsonStr = json.dumps(emptlst)
    with open('data.txt', 'w') as outfile:
        json.dump(emptlst, outfile)
    return jsonStr


@app.route('/')
def upload_page():
    return render_template("index.html")


@app.route('/', methods=["POST"])
def upload_image():
    # Validates that the uploaded image has the proper size and file type.
    if 'image' not in request.files:
        flash('No file part')
        return redirect(request.url)
    image = request.files['image']
    if image.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if image and allowed_file(image.filename):
        filename = secure_filename(image.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(image_path)
        print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed')

        # The image is process for classification.
        prediction = model_predict(image_path, model)
        pred_class = decode_predictions(prediction, top=5)
        results = convert_json(pred_class)
        print(results)
        return render_template('index.html')
    else:
        flash('Allowed image types are -> png, jpg, jpeg')
        return redirect(request.url)


if __name__ == '__main__':
    app.run(debug=True)
