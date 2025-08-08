# Fish Image Classification

Aim: To input the image of a fish and classify it into one of the 11 categories.

Data preprocessing and augmentation

* Loaded the fish dataset, which was organized into train, val, and test folders with one folder per class.
* Resized all images to 224x224 for consistency and to match model input requirements.
* Normalized pixel values to the [0, 1] range by dividing by 255.
* Applied data augmentation techniques such as random rotation, zoom, horizontal flip, and vertical flip to increase the variety of training images and improve model robustness.
* Created separate generators for training, validation, and test sets, ensuring that augmentation was only applied to the training data.

Model training

* Started with building a custom CNN from scratch to establish a baseline.
* Experimented with fine-tuning five pre-trained models: VGG16, ResNet50, MobileNet, InceptionV3, and EfficientNetB0.
* For each pre-trained model, removed the top layer, added a global average pooling layer, a dense hidden layer, and a softmax output layer matching the number of classes.
* Froze the lower layers of each pre-trained model and fine-tuned the upper layers with a small learning rate to adapt to the fish dataset.
* Trained each model on the processed dataset using categorical cross-entropy loss and the Adam optimizer.

Model evaluation and selection

* Evaluated all models using precision, recall, F1-score, and accuracy on the test set.
* MobileNet and InceptionV3 achieved the highest accuracy (~99.4%) with very balanced precision and recall across almost all classes.
* ResNet50 and EfficientNetB0 underperformed due to preprocessing mismatch and were not chosen.
* MobileNet was selected because it matched InceptionV3’s accuracy while being smaller and faster, making it more efficient for deployment.
* Observed that the rare class “animal fish bass” was poorly predicted by all models due to extreme class imbalance.

Streamlit application

* Built a Streamlit app with a simple UI for uploading a fish image and classifying it.
* The uploaded image is resized to 224x224, normalized to [0, 1], and passed into the fine-tuned MobileNet model.
* The app displays the top-1 predicted class along with its confidence score and the top-3 predicted classes with their probabilities.
* The app dynamically loads the model and class labels so updated versions can be used without restarting the application.