# Imports
import tensorflow
import numpy as np
import matplotlib.pyplot as plt    
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator as IDG
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Defining global seed for reproducability
SEED = 42
np.random.seed(SEED)
keras.utils.set_random_seed(SEED)

#%%----------------------------------------------------------------------------------------------------------------------------------------------
# Step 1: Data Processing

# Defining Input Image Shape (500,500,3), batch size and epochs
img_h = 500 
img_w = 500
img_channel = 3 # RGB Channels
img_shape = (img_h,img_w,img_channel) # Input image shape
batch = 32 # Defined batch size
epoch = 15 # Defined epochs

# Defining relative directory paths
# Since this is a simpler relative referencing, important to ensure that the Python script is in the same directory as the Project 2 Data
train_directory = "Project 2 Data/Data/train"
valid_directory = "Project 2 Data/Data/valid"


# Data Augmentation on training image set using ImageDataGenerator
train_aug = IDG(
    rescale = 1./255, #normalize RGB pixels from [0,255] into [0,1]
    shear_range = 0.15, #shear angle in degrees
    zoom_range = 0.15, #zoom in/out by +-15%
    rotation_range = 10, #random rotation in the range +-10 degrees
    brightness_range = [0.85,1.15], # +-15% variation in brightness
    fill_mode = "nearest", #fill empty pixels with nearest values
    horizontal_flip = True #flip images horizontally
)

# Validation images only rescaled since those would be used to validate our training effectiveness
valid_aug = IDG(rescale = 1./255)

# Data Generator using ImageDataGenerator
# Training DataSet generation
train_gen = train_aug.flow_from_directory(
    train_directory,
    target_size=(img_h,img_w),
    color_mode = 'rgb',
    classes = ['crack','missing-head','paint-off'],
    class_mode = 'categorical',
    batch_size = batch,
    shuffle = True, # Shuffles training sample for better generalization learning
    seed = SEED
)

# Validation DataSet generation
valid_gen = valid_aug.flow_from_directory(
    valid_directory,
    target_size=(img_h,img_w),
    color_mode = 'rgb',
    classes = ['crack','missing-head','paint-off'],
    class_mode = 'categorical',
    batch_size = batch,
    shuffle = False, # Only used for evaluation so need not shuffle, no weight updates
    seed = SEED
)

# Ensuring all image samples are being processed and classes inferred are correct
print("\nTrain dataset:", train_gen.samples, "samples")
print("Validation dataset:", valid_gen.samples, "samples")
print("Train Class indices:", train_gen.class_indices) # Not for validation --> Data Leak
#%%------------------------------------------------------------------------------------------------------------------------------------
# Step 2-3: Neural Network Architecture and Hyperparameter Tuning

# Defining 3 different callbacks
#Stops training automatically when the validation accuracy stops improving for 3 consecutive epochs.
early_stop = EarlyStopping(
    monitor="val_accuracy",
    patience=5, 
    restore_best_weights=True
)
#Saves the model whenever validation accuracy improves -- always have the best-performing model on disk.
checkpoint_filepath_base = "Best_Models/base_model.keras"
ckpt_base = ModelCheckpoint(
    filepath=checkpoint_filepath_base,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True
)
checkpoint_filepath_imp = "Best_Models/overall_best_model.keras"
ckpt_best = ModelCheckpoint(
    filepath=checkpoint_filepath_imp,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True
)
#Automatically reduces the learning rate when validation loss stops improving, allowing finer convergence.
rlrop = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=2,
    min_lr=0.0001
)

# Baseline CNN Architecture with four sets of Convolution and MaxPooling layers for feature extraction
mdl1= Sequential([ 
    Input(shape=img_shape),
    Conv2D(32,3,activation="relu"),
    MaxPooling2D(2,2), 
    Conv2D(64,3),
    layers.Leaky_ReLU(0.1),
    MaxPooling2D(2,2),
    Conv2D(128,3,activation="relu"), 
    MaxPooling2D(2,2), 
    Conv2D(256,3,activation="relu"), 
    MaxPooling2D(2,2),
    Flatten(), # Flattens the 2D feature maps into a 1D feature vector to feed into the fully connected (dense) layers
    Dense(64, activation="relu"), # Fully connected Dense layer with 64 neurons
    Dropout(0.2),                 # Dropout layer -- dropping 20% activations from previous layer 
    Dense(3, activation="softmax") # Output Dense layer with 3 neurons -- one for each class
])

# Optimizer
mdl1.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=['accuracy']
)
# Baseling Model Summary
mdl1.summary()

# Training the Baseline Model
hist1 = mdl1.fit(
    train_gen,
    validation_data = valid_gen,
    epochs= epoch,
    callbacks=[early_stop,ckpt_base,rlrop],
    verbose=1
)

# Tuned CNN Architecture with Dense layer activation elu
mdl2= Sequential([ 
    Input(shape=img_shape),
    Conv2D(16,3,activation="relu"),
    Conv2D(16,3,activation="relu"),
    MaxPooling2D(2,2), 
    Conv2D(32,3,activation="relu"),
    Conv2D(32,3,activation="relu"), 
    MaxPooling2D(2,2),
    Conv2D(64,3,activation="relu"), 
    MaxPooling2D(2,2), 
    Flatten(), 
    Dense(64, activation="elu"),
    Dropout(0.2),                 
    Dense(3, activation="softmax")
])

# Optimizer
mdl2.compile(
    optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Tuned Model Summary
mdl2.summary()

# Training the Tuned Model
hist2 = mdl2.fit(
    train_gen,
    validation_data = valid_gen,
    epochs= epoch,
    callbacks=[early_stop,rlrop,ckpt_best],
    verbose=1
)
#%%------------------------------------------------------------------------------------------------------------------------------------
# Step 4: Model Evaluation

# Finally Evaluate the Model on the Validation Data Gen using the final weights obtained after training
valid_loss1, valid_accuracy1 = mdl1.evaluate(valid_gen)
print(f"\nFinal Validation accuracy (baseline): {valid_accuracy1:.4f}")
print(f"Final Validation loss (baseline): {valid_loss1:.4f}\n")
# Training loss and accuracy obtained from the last Epoch
train_loss = hist1.history['loss'][-1]
train_accuracy = hist1.history['accuracy'][-1]
print(f"\nFinal Training accuracy (baseline): {train_accuracy:.4f}")
print(f"Final Training loss (baseline): {train_loss:.4f}\n")

# Plotting Training & Validation accuracy
plt.figure(figsize=(10, 4))
plt.plot(hist1.history['accuracy'], label="Training Accuracy")
plt.plot(hist1.history['val_accuracy'], label="Validation Accuracy")
plt.title('Baseline Model Training and Validation Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Plotting Training and Validation loss
plt.figure(figsize=(10, 4))
plt.plot(hist1.history['loss'], label="Training Loss")
plt.plot(hist1.history['val_loss'], label="Validation Loss")
plt.title('Baseline Model Training and Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# Finally Evaluate the Model on the Validation Data Gen using the final weights obtained after training
valid_loss2, valid_accuracy2 = mdl2.evaluate(valid_gen)
print(f"\nFinal Validation accuracy (tuned): {valid_accuracy2:.4f}")
print(f"Final Validation loss (tuned): {valid_loss2:.4f}\n")
# Training loss and accuracy obtained from the last Epoch
train_loss2 = hist2.history['loss'][-1]
train_accuracy2 = hist2.history['accuracy'][-1]
print(f"\nFinal Training accuracy (tuned): {train_accuracy2:.4f}")
print(f"Final Training loss (tuned): {train_loss2:.4f}\n")

# Plotting Training & Validation accuracy
plt.figure(figsize=(10, 4))
plt.plot(hist2.history['accuracy'], label="Training Accuracy")
plt.plot(hist2.history['val_accuracy'], label="Validation Accuracy")
plt.title('Tuned Model Training and Validation Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Plotting Training and Validation loss
plt.figure(figsize=(10, 4))
plt.plot(hist2.history['loss'], label="Training Loss")
plt.plot(hist2.history['val_loss'], label="Validation Loss")
plt.title('Tuned Model Training and Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()