import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load label mapping from JSON
with open('class_labels.json', 'r') as f:
    labels = json.load(f)

def Predict_Fish(img, model_path):
    # Load the model
    model = load_model(model_path)

    # Predict
    pred = model.predict(img)
    pred_class = np.argmax(pred)

    # Convert index to string for JSON dictionary
    predicted_fish = labels[str(pred_class)]
    print("Predicted fish type:", predicted_fish)

custom_cnn_path = 'Custom CNN/custom_cnn_model.h5'
vgg_path = 'VGG16/vgg_finetuned_model.h5'
resnet_path = 'ResNet50/resnet_finetuned_model.h5'
mobilenet_path = 'MobileNet/mobilenet_finetuned_model.h5'
inception_path = 'InceptionV3/inception_finetuned_model.h5'
efficientnet_path = 'EfficientNetB0/efficientnet_finetuned_model.h5'

test_img_path = 'Dataset/data/test/fish sea_food black_sea_sprat/0I5O9H5AFIAE.jpg'

# Load and preprocess the image
img = image.load_img(test_img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# Predict for all models

Predict_Fish(img_array, custom_cnn_path)
Predict_Fish(img_array, vgg_path)
Predict_Fish(img_array, resnet_path)
Predict_Fish(img_array, mobilenet_path)
Predict_Fish(img_array, inception_path)
Predict_Fish(img_array, efficientnet_path)