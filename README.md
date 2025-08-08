# Fish Image Classification

Aim: To input the image of a fish and classify it into one of the 11 categories.

The dataset which is provided has three main folders which are test, train and val serving different purposes.

#### Data preprocessing and augmentation

In this stage, the paths for each of the folders are specified with batch size and their dimensions. The train data is rescaled, rotated and flipped to make the model learn better, whereas the test and val data are only resized (224 x 224) using the image generator function from Keras. After this, generators are created which will be used to train and test the model. A small block of code is used to create a JSON file consisting of class labels that are used later.

#### Model Training

For model training, a total of six models were used which are

* Custom CNN Model: A CNN model built from scratch using Sequential function consisting of four layers and 20 epochs.
* VGG16: A pre-trained which is fine-tuned and modified by replacing output layers and retraining last 4 layers with 5 epochs size.
* ResNet50: A pre-trained which is fine-tuned and modified by replacing output layers and retraining last 20 layers with 10 epochs size.
* MobileNet: A pre-trained which is fine-tuned and modified by replacing output layers and retraining last 30 layers with 10 epochs size.
* InceptionV3: A pre-trained which is fine-tuned and modified by replacing output layers and retraining last 50 layers with 10 epochs size.
* EfficientNetB0: A pre-trained which is fine-tuned and modified by replacing output layers and retraining last 30 layers with 10 epochs size.

Each model, after being trained, is exported as .h5 and .keras format to be used later.

#### Model Evaluation and Selection

For model evaluation metrics such as Accuracy, Precision, F1 Score were used. In addition to this, confusion matrices for each model were plotted and analyzed. The Accuracy for each model are as follows

Custom CNN Model: 0.9448
VGG16: 0.9649
ResNet50: 0.7330
MobileNet: 0.9940
InceptionV3: 0.9940
EfficientNetB0: 0.1632

Note: A point to be noted that all the models performed poorly recognizing animal fish bass due to insufficient images causing underfitting

After looking at all the factors considering model complexity, execution times, accuracy and confidence scores on some images, MobileNet seemed like an appropriate model.

#### StreamLit Application

Finally, a streamlit application with the appropriate UI is built that allows the users to upload an image and the fine-tuned MobileNet model predicts top 3 fish's names with the confidence scores.
