import argparse
import json
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np


def process_image(image):
    image = tf.image.resize(image, [224, 224])
    image /= 255
    return image.numpy()


def predict(image_path, model, class_names, top_k):
    image = Image.open(image_path)
    image_np_arr = np.asarray(image)
    processed_image = process_image(image_np_arr)
    image = np.expand_dims(processed_image, axis=0)
    prediction = model.predict(image)
    indices_top_k = prediction.argsort()[0][-top_k:][::-1]
    classes = [class_names[str(i + 1)] for i in indices_top_k]
    probs = [prediction[0][i] for i in indices_top_k]
    return probs, classes


parser = argparse.ArgumentParser(description='Predict parser')
parser.add_argument('image_path', help='Image path')
parser.add_argument('saved_model', help='Model path', default='model.h5')
parser.add_argument('--top_k', help='Top k', required=False, default=5)
parser.add_argument('--category_names', help='Class map json file path', required=False, default='label_map.json')
args = parser.parse_args()

with open(args.category_names, 'r') as f:
    class_names = json.load(f)

saved_model = tf.keras.models.load_model(args.saved_model, custom_objects={'KerasLayer': hub.KerasLayer}, compile=False)
probs, classes = predict(args.image_path, saved_model, class_names, int(args.top_k))
print('Classes probabilities:')
for i in range(int(args.top_k)):
    print('{}: {:.2f}'.format(classes[i], probs[i]))

# CLI usage
# python predict.py ./test_images/orange_dahlia.jpg model.h5 --top_k 4

