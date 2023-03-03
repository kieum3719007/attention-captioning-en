import io
import os

from flask import Flask, render_template, request
from PIL import Image
from core.captioner import Captioner
from core.image_captioning import create_model
from core.image_utils import preproccess_image

CAPTIONER = Captioner(create_model())
app = Flask(__name__)
image_size = (384, 384)

def transform_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert("RGB")
    image = preproccess_image(image=image, size=image_size, normalize=True)
    return image


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # read image bytes
        
        if ('image' in request.files):
            file = request.files['image']
            image_bytes = file.read()
        
            save_image(image_bytes)
            sentence = get_prediction(image_bytes)
            return render_template("result.html", sentence=sentence, bytes=image_bytes)
        else:
            return render_template('index.html')


def get_prediction(image_bytes):
    """
    Gets the prediction of a single image
    """
    
    image = transform_image(image_bytes)
    return CAPTIONER(image)

def save_image(image_bytes):
    """
    Saves an image to temporary location
    """
    image = Image.open(io.BytesIO(image_bytes))
    image = image.resize((500, 500))

    image.save(os.path.join("static", "images", "temp.png"))