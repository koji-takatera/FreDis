import os
from flask import Flask, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename

from keras.models import Sequential, load_model
import keras,sys
import numpy as np
from PIL import Image

classes = ["freshapples","rottenapples", "freshbanana", "rottenbanana", "freshoranges", "rottenoranges"]
num_classes = len(classes)
image_size = 50

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            print('ファイルなし')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            print('ファイルなし')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            model = load_model('./fruits_cnn.h5')

            image = Image.open(filepath)
            image = image.convert('RGB')
            image = image.resize((image_size, image_size))
            data = np.asarray(image)
            X = []
            X.append(data)
            X = np.array(X)

            result = model.predict([X])[0]
            predicted = result.argmax()
            percentage = int(result[predicted] * 100)

            return "Result： " + classes[predicted] + ", Probability："+ str(percentage) + " %"

    return render_template('index.html')

            #return redirect(url_for('uploaded_file', filename=filename))
    #return '''
    #<!doctype html>
    #<html>
    #<head>
    #<meta charset="UTF-8">
    #<link rel="stylesheet" href="../static/index.css">
    #<title>Fresh Discriminator</title>
    #</head>
    #<body>
    #<header class="header">
    #        <h1 class="logo">
    #            <a href="/">Fresh Discriminator</a>
    #        </h1>
    #    <nav class="global-nav">
    #        <li class="nav-item active"><a href="/">HOME</a></li> 
    #        <li class="nav-item active"><a href="/about">ABOUT</a></li>           
    #    </nav>
    #</header>
    #<form method = post enctype = multipart/form-data>
    #<p><input type=file name=file>
    #<input type=submit value=Upload>
    #</form>
    #</body>
    #</html>
    #'''

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/about')
def about():
    return render_template('about.html')

