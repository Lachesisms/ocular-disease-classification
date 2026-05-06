from flask import Flask, request, render_template, send_from_directory
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
import os
import numpy as np
import tensorflow as tf
import cv2
import base64

app = Flask(__name__, template_folder=r'D:\EyeDieaseProject\program\templates')
app.config['UPLOAD_FOLDER'] = r'D:\EyeDieaseProject\program\uploads'

MODEL_PATH = r'D:\EyeDieaseProject\model\ResNet50\ResNet50_best.keras'
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

DISEASE_CLASSES = [
    'Age-related Macular Degeneration',  # 0: A
    'Cataract',                          # 1: C
    'Diabetes',                          # 2: D
    'Glaucoma',                          # 3: G
    'Hypertension',                      # 4: H
    'Pathological Myopia',               # 5: M
    'Normal',                            # 6: N
    'Other Diseases',                    # 7: O
]


# ============================================================
# Grad-CAM: generates saliency heatmap overlaid on input image
# to visualize which regions influenced the model's prediction
# ============================================================
def generate_gradcam(model, img_array, pred_index, original_img_path):
    last_conv_layer_name = 'conv5_block3_out'

    grad_model = Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    original_img = cv2.imread(original_img_path)
    original_img = cv2.resize(original_img, (224, 224))

    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(original_img, 0.6, heatmap_colored, 0.4, 0)

    _, buffer = cv2.imencode('.jpg', superimposed)
    return base64.b64encode(buffer).decode('utf-8')


# ============================================================
# Process a single uploaded image: save, preprocess, predict, generate Grad-CAM
# ============================================================
def process_image(file):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    img = load_img(filepath, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    predicted_class = DISEASE_CLASSES[predicted_index]
    confidence = float(np.max(predictions)) * 100

    gradcam = generate_gradcam(model, img_array, predicted_index, filepath)

    return {
        'filename': file.filename,
        'result': predicted_class,
        'confidence': f"{confidence:.1f}",
        'gradcam': gradcam
    }


# ============================================================
# Routes
# ============================================================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "Model not loaded", 500

    files = request.files.getlist('file')
    files = [f for f in files if f.filename != '']

    if not files:
        return "No file uploaded", 400

    results = [process_image(f) for f in files[:2]]  # Process up to 2 images

    return render_template('result.html', results=results)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
