import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Define constants
IMG_HEIGHT, IMG_WIDTH = 128, 128
BATCH_SIZE = 32
EPOCHS = 10  # Increased number of epochs

# Define paths to your dataset folders
train_dir = '/kaggle/input/pixelated-image-detection-and-correction/Image_Processing/'
train_pixelated_dir = os.path.join(train_dir, 'Pixelated')
train_original_dir = os.path.join(train_dir, 'Original')

# Function to split data into train and validation directories
def split_data(original_dir, pixelated_dir, train_dir, test_dir, split_ratio):
    os.makedirs(os.path.join(train_dir, 'Original'), exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'Pixelated'), exist_ok=True)
    os.makedirs(os.path.join(test_dir, 'Original'), exist_ok=True)
    os.makedirs(os.path.join(test_dir, 'Pixelated'), exist_ok=True)
    
    original_files = os.listdir(original_dir)
    pixelated_files = os.listdir(pixelated_dir)
    paired_files = [(file, file) for file in original_files if file in pixelated_files]
    np.random.shuffle(paired_files)
    split_index = int(len(paired_files) * split_ratio)
    train_files = paired_files[:split_index]
    test_files = paired_files[split_index:]
    
    for original_file, pixelated_file in train_files:
        shutil.copy(os.path.join(original_dir, original_file), os.path.join(train_dir, 'Original', original_file))
        shutil.copy(os.path.join(pixelated_dir, pixelated_file), os.path.join(train_dir, 'Pixelated', pixelated_file))
        
    for original_file, pixelated_file in test_files:
        shutil.copy(os.path.join(original_dir, original_file), os.path.join(test_dir, 'Original', original_file))
        shutil.copy(os.path.join(pixelated_dir, pixelated_file), os.path.join(test_dir, 'Pixelated', pixelated_file))

# Split the data
base_dir = '/kaggle/working/'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
split_ratio = 0.8
split_data(train_original_dir, train_pixelated_dir, train_dir, test_dir, split_ratio)

# ImageDataGenerator for augmentation and scaling
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,  # Reduced range
    width_shift_range=0.1,  # Reduced range
    height_shift_range=0.1,  # Reduced range
    shear_range=0.1,  # Reduced range
    zoom_range=0.1,  # Reduced range
    horizontal_flip=True,
    fill_mode='nearest'
)

# Generate batches of augmented data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

validation_generator = train_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# Define the model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D((2, 2)),
    # BatchNormalization(),  # Removed for simplicity

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    # BatchNormalization(),  # Removed for simplicity

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    # BatchNormalization(),  # Removed for simplicity

    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    # BatchNormalization(),  # Removed for simplicity

    GlobalAveragePooling2D(),

    Dense(128, activation='relu'),
    Dropout(0.5),

    Dense(64, activation='relu'),
    Dropout(0.5),

    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss=BinaryCrossentropy(),
              metrics=[BinaryAccuracy(), Precision(), Recall()])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)  # Increased patience
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    callbacks=[early_stopping, reduce_lr, model_checkpoint]
)

# Save the model
model.save('/kaggle/working/pixelated_detection_model.keras')

# Calculate and print model size
model_size = os.path.getsize('/kaggle/working/pixelated_detection_model.keras') / (1024 * 1024)
print(f'Model Size: {model_size:.2f} MB')

# Evaluate the model
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

evaluation_results = model.evaluate(test_generator)
loss, accuracy, precision, recall = evaluation_results[:4]
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Example of using the trained model for prediction
sample_image_path = '/kaggle/input/pixelated-image-detection-and-correction/Image_Processing/Original/113.png'
sample_image = load_img(sample_image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
sample_image = img_to_array(sample_image)
sample_image = np.expand_dims(sample_image, axis=0) / 255.0
prediction = model.predict(sample_image)
if prediction[0] > 0.5:
    print('Prediction: Pixelated')
else:
    print('Prediction: Not Pixelated')
