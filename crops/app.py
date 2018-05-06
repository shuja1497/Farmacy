from flask import Flask, request
import numpy as np
import os
import pickle
import base64
from keras.models import load_model
from keras.preprocessing import image as image_utils

app = Flask(__name__)


label_map = {'24': 17, '25': 18, '26': 19, '27': 20, '20': 13,
 '21': 14, '22': 15, '23': 16, '28': 21, '29': 22, '1': 1, '0': 0,
  '3': 23, '2': 12, '5': 33, 'c_4': 32, '7': 35, '6': 34, '9': 37, '33': 27,
   '32': 26, '37': 31, '36': 30, '35': 29, '34': 28, '19': 11, '18': 10, '31': 25,
    '30': 24, '15': 7, '14': 6, '17': 9, '16': 8, '11': 3, '10': 2, '13': 5, '12': 4, '8': 36}


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


@app.route('/hello/<username>/<first>/<second>')
def hello(username, first, second):
	a = first+second
	return a


if __name__ == '__main__':
	print "hellooo"
	model = load_model('plantdisease_withVal_allepoch64.h5')
	print "model loaded"
	app.run(port=8080, use_reloader=True)

