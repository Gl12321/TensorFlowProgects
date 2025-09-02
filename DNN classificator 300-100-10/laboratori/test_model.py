import joblib
import numpy as np
from PIL import Image
import tensorflow as tf
import sys
sys.path.append('..')
from model import DNN


model = joblib.load("../../../DNN_300_100_10.pkl")

img = Image.open("photo_2025-08-20_17-52-31/zz8.png").convert("L")
img.show()
img = img.resize((28, 28))
img_array = np.array(img)

inverted_array = 255 - img_array
binary_array = np.where(inverted_array > 127, 255, 0).astype(np.uint8)

result_img = Image.fromarray(binary_array)
result_img.show()

binary_array = binary_array / 255.0
binary_array = binary_array.reshape(1, -1).astype(np.float32)

logits = model(binary_array)
prediction = tf.argmax(logits, axis=1).numpy()[0]
print(f"Предсказанная цифра: {prediction}")