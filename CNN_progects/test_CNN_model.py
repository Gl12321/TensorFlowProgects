import numpy as np
import tensorflow as tf
from PIL import Image
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

model_path = "/home/stanislav/ml_projects/tensorflow_progects/CNN_progects/laboratori/ResNet18"
model = tf.saved_model.load(model_path)
infer = model.signatures["serving_default"]

(_, _), (X_train, _) = tf.keras.datasets.mnist.load_data()
reference_mean = (X_train / 255.0).mean()

def preprocess_png_auto(path):
    img = Image.open(path).convert("L")
    img = img.resize((28, 28), Image.LANCZOS)

    arr = np.array(img).astype(np.float32) / 255.0

    if arr.mean() > reference_mean:
        arr = 1.0 - arr

    arr_min = arr.min()
    arr_max = arr.max()
    # arr = (arr - arr_min) / (arr_max - arr_min)

    final_img = Image.fromarray((arr * 255).astype(np.uint8))
    final_img.save("final_image_for_neural_network.png")
    print("Сохранено изображение: final_image_for_neural_network.png")

    return arr.reshape(1, 28, 28, 1).astype(np.float32)

x = preprocess_png_auto("laboratori/ds/qq3.png")
outputs = infer(tf.constant(x))
logits = outputs["output"].numpy()
probs = tf.nn.softmax(logits).numpy()
print("предсказаная цифра:", np.argmax(probs[0]))
