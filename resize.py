import pandas as pd
import numpy as np
import cv2

# Load the MNIST dataset
mnist_train_path = r"Real-Time-Sign-Language-Recognition-Using-Machine-Learning-main/sign_mnist_train.csv"
mnist_train = pd.read_csv(mnist_train_path)

# Print the first few rows and the shape of the DataFrame
print(mnist_train.head())
print(f"Shape of the MNIST train dataset: {mnist_train.shape}")

# Preprocess MNIST dataset
mnist_labels = mnist_train['label'].values
mnist_train.drop('label', axis=1, inplace=True)

# Convert the MNIST data into a list of 28x28 grayscale images
mnist_images = np.array([np.reshape(i, (28, 28)) for i in mnist_train.values])

# Print shapes and sample values to diagnose
for idx, img in enumerate(mnist_images):
    print(f"Image at index {idx}: shape {img.shape} values: {img.flatten()[:5]}")  # First 5 pixel values

# Ensure the images are valid for resizing
valid_mnist_images = []
for idx, img in enumerate(mnist_images):
    if img is not None:
        try:
            resized_img = cv2.resize(img, (64, 64))  # Resize to 64x64
            valid_mnist_images.append(resized_img)
        except Exception as e:
            print(f"Error resizing image at index {idx}: {e}")
    else:
        print(f"Invalid image encountered at index {idx}, skipping.")

# Convert the list of resized images into a numpy array
if valid_mnist_images:
    mnist_images_resized = np.array(valid_mnist_images).reshape(len(valid_mnist_images), 64, 64, 1)
    print(f"Successfully resized {len(valid_mnist_images)} MNIST images to 64x64.")
else:
    print("No valid images found to resize.")
