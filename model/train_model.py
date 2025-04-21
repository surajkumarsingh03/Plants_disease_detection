import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import load_model

# Automatically build the correct path relative to this script
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, '../data/PlantVillage')  # Adjust this if your dataset is somewhere else

# Check if data path exists
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Dataset path does not exist: {data_dir}")

img_size = 128
batch_size = 32

# Data augmentation
datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255)

# Training generator
train_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    subset='training',
    shuffle=True
)

# Validation generator
val_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    subset='validation',
    shuffle=False
)

print(f"Found {train_gen.samples} training images belonging to {train_gen.num_classes} classes.")
print(f"Found {val_gen.samples} validation images.")

# Build model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_gen.num_classes, activation='softmax')
])

# Compile and train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_gen, validation_data=val_gen, epochs=10)

# Save model
model.save(os.path.join(base_dir, 'plant_disease_model.h5'))
