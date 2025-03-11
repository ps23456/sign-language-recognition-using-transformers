from PIL import Image
#importing Libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# Reading the ISL Dataset
def load_isl_dataset(isl_path):
    images_data = []
    labels = []
    for label in os.listdir(isl_path):
        label_folder = os.path.join(isl_path, label)
        if os.path.isdir(label_folder):
            for img_file in os.listdir(label_folder):
                img_path = os.path.join(label_folder, img_file)
                
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    with Image.open(img_path) as img:
                        img = img.convert('L')  # Convert to grayscale
                        img = img.resize((28, 28))  # Resize to 28x28 pixels
                        img_array = np.array(img).flatten()
                        images_data.append(img_array)
                        labels.append(label)
    return np.array(images_data), np.array(labels)

# Load ISL dataset
isl_images, isl_labels = load_isl_dataset('Real-Time-Sign-Language-Recognition-Using-Machine-Learning-main/ISL_Dataset')

# Loading MNIST Dataset
mnist_train = pd.read_csv(r"Real-Time-Sign-Language-Recognition-Using-Machine-Learning-main/sign_mnist_train.csv")
mnist_test = pd.read_csv(r"Real-Time-Sign-Language-Recognition-Using-Machine-Learning-main/sign_mnist_test.csv")

# Preprocess MNIST dataset
mnist_labels = mnist_train['label'].values
mnist_train.drop('label', axis=1, inplace=True)
mnist_images = np.array([np.reshape(i, (28, 28)) for i in mnist_train.values])

mnist_images = mnist_images.reshape(mnist_images.shape[0], 28, 28, 1)

# Reshaping ISL images to have the same number of dimensions as MNIST (28x28x1)
isl_images = isl_images.reshape(isl_images.shape[0], 28, 28, 1)

# Combine MNIST and ISL datasets
combined_images = np.concatenate((mnist_images, isl_images), axis=0)
combined_labels = np.concatenate((mnist_labels, isl_labels), axis=0)


# Transform labels into binary representation
label_binrizer = LabelBinarizer()

combined_labels = label_binrizer.fit_transform(combined_labels)

# Split combined data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(combined_images, combined_labels, test_size=0.3, random_state=101)

# Scaling and reshaping the data
x_train = x_train / 255
x_test = x_test / 255
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Data augmentation to prevent overfitting
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(x_train)

# Building CNN model
model = Sequential()

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(len(label_binrizer.classes_), activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model with augmented data
history = model.fit(datagen.flow(x_train, y_train, batch_size=128), validation_data=(x_test, y_test),
                    epochs=15, callbacks=[early_stopping])

# Saving the model
model.save("combined_sign_language_model.h5")
print("Model Saved")

# Plotting accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Accuracy")
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train', 'test'])
plt.show()

# Evaluate on test data
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)
print("Accuracy:", accuracy_score(y_true, y_pred_classes))
print(confusion_matrix(y_true, y_pred_classes))
print(classification_report(y_true, y_pred_classes))

# Real-Time Gesture Recognition
def getLetter(result):
    classLabels = {i: chr(65+i) for i in range(len(label_binrizer.classes_))}
    try:
        res = int(result)
        return classLabels[res]
    except:
        return "Error"

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    roi = frame[100:400, 320:620]
    cv2.imshow('roi', roi)
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    cv2.imshow('roi scaled and gray', roi)
    copy = frame.copy()
    cv2.rectangle(copy, (320, 100), (620, 400), (255, 0, 0), 5)
    roi = roi.reshape(1, 28, 28, 1)
    result = str(np.argmax(model.predict(roi, 1, verbose=0), axis=1)[0])
    cv2.putText(copy, getLetter(result), (300, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
    cv2.imshow('frame', copy)
    if cv2.waitKey(1) == 13:  # 13 is the Enter Key
        break
cap.release()
cv2.destroyAllWindows()
