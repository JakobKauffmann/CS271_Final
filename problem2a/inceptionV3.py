import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt

train_dir = '/Users/mithipandey/Documents/ExamCS271/final_data_combined/digraph'
test_dir = '/Users/mithipandey/Documents/ExamCS271/final_data_combined/digraphTest'

# Image dimensions (InceptionV3 default input size)
img_width, img_height = 299, 299
batch_size = 32
num_classes = 5  #A, B, C, D, E

# -------------------------------
# 1. Load and Preprocess the Data
# -------------------------------
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,     # Normalize pixel values
    rotation_range=30,     # Data augmentation
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)  # No augmentation for test data because we need it be as raw as possible
                                                      # for best test results 

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'  # For multi-class classification
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

# --------------------------------------
# 2. Loading the Pre-trained InceptionV3
# --------------------------------------
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Freezing the base model layers
base_model.trainable = False

# --------------------------
# 3. Building the Model
# --------------------------
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')  # Final layer for 6 classes
])

# Compiling the model
model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ------------------------
# 4. Training the Model
# ------------------------
epochs = 10  # 10 for now is okay so see how the model improves

history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=test_generator
)

# ------------------------------------------------------
# 5. Evaluating the Model with Accuracy and Loss Graphs
# ------------------------------------------------------

def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    # Plotting accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    # Plotting loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

plot_training_history(history)

# ---------------------------
# 6. Fine-Tuning the Model 
# ---------------------------

# Unfreeze some layers for fine-tuning
base_model.trainable = True

# Fine-tune from the last 20 layers
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Recompiling the model with a lower learning rate
model.compile(optimizer=optimizers.Adam(learning_rate=0.00001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Continue training
fine_tune_epochs = 5
total_epochs = epochs + fine_tune_epochs

history_fine = model.fit(
    train_generator,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1],
    validation_data=test_generator
)

# Plotting the final fine-tuning results
plot_training_history(history_fine)
