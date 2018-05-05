from flask import Flask, request
import numpy as np
import os
import pickle
import base64
from keras.models import load_model
from keras.preprocessing import image as image_utils

app = Flask(__name__)


label_map = {'c_24': 17, 'c_25': 18, 'c_26': 19, 'c_27': 20, 'c_20': 13,
 'c_21': 14, 'c_22': 15, 'c_23': 16, 'c_28': 21, 'c_29': 22, 'c_1': 1, 'c_0': 0,
  'c_3': 23, 'c_2': 12, 'c_5': 33, 'c_4': 32, 'c_7': 35, 'c_6': 34, 'c_9': 37, 'c_33': 27,
   'c_32': 26, 'c_37': 31, 'c_36': 30, 'c_35': 29, 'c_34': 28, 'c_19': 11, 'c_18': 10, 'c_31': 25,
    'c_30': 24, 'c_15': 7, 'c_14': 6, 'c_17': 9, 'c_16': 8, 'c_11': 3, 'c_10': 2, 'c_13': 5, 'c_12': 4, 'c_8': 36}


def prediction():
    # building the path
    # testing for a single image
    test_image = image_utils.load_img('image.jpeg', target_size=(32, 32))
    test_image = image_utils.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict_on_batch(test_image)
    # print(result)
    predicted_label = list(label_map.keys())[list(label_map.values()).index(np.argmax(result))]
    return predicted_label


@app.route('/', methods=['GET', 'POST'])
def start():
    if request.method == 'POST':
        strng = request.values
        imageInstring = strng['image']
        imgdata = base64.b64decode(imageInstring)

        with open("image.jpeg", "wb") as fh:
            fh.write(imgdata)

        pred = prediction()
        return pred
    else:
        return "<h1>use post method</h1>"


@app.route('/hello/<username>')
def hello(username):
    return '<h1>u want soluchan %s ?</h1>' % username


if __name__ == '__main__':
	print "hellooo"
	model = load_model('plantdisease_withVal_allepoch64.h5')
	print "model loaded"
	app.run(port=8080, use_reloader=True)

