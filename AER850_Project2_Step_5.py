import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator as IDG
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd

# Relative paths for test images and model
test_images = [
    ("crack","Project 2 Data/Data/test/crack/test_crack.jpg"),
    ("missing-head","Project 2 Data/Data/test/missing-head/test_missinghead.jpg"),
    ("paint-off","Project 2 Data/Data/test/paint-off/test_paintoff.jpg")]
model_path = "Best_Models/overall_best_model.keras"
print("Loaded Best Model successfully")

best_model = load_model(model_path) # Load the best model
class_labels = ['crack','missing-head','paint-off']

for true_labels,img_path in test_images:
    img = image.load_img(img_path, target_size=(500, 500), color_mode = 'rgb') # Load the image
    img_array = image.img_to_array(img) # Convert Image Tensor to NumPy Array
    img_array = img_array / 255.0 # Normalize the pixel values
    img_array = np.expand_dims(img_array, axis=0) # Adding a Batch Dimension
    prediction = best_model.predict(img_array,verbose=0)[0] # Predict the labels for test dataset
    predicted_class = np.argmax(prediction)
    predicted_label = class_labels[predicted_class]
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"True Crack Classification Label: {true_labels}\n" f"Predicted: {predicted_label}")
    prob_text = "\n".join([f"{c.title().replace('-', ' ')}: {p*100:.1f}%" for c, p in zip(class_labels, prediction)])
    plt.text(10, 470, prob_text, fontsize=12, color="red", weight="bold", va="bottom", ha="left", backgroundcolor=(1, 1, 1, 0.4)) # semi-transparent white box for readability
    plt.show()

test_datagen = IDG(rescale=1./255)
test_gen = test_datagen.flow_from_directory(
    directory="Project 2 Data/Data/test",   # Test directory
    target_size=(500, 500),
    color_mode='rgb',
    classes=['crack', 'missing-head', 'paint-off'],  # same order as training
    class_mode='categorical',
    batch_size=32,          
    shuffle=False           # keep order deterministic
)

# Evaluate Test Accuracy and Test Loss for the entire test dataset
test_loss, test_accuracy = best_model.evaluate(test_gen, verbose=1)

print(f"\nFinal Test Accuracy : {test_accuracy:.4f}")
print(f"Final Test Loss     : {test_loss:.4f}\n")

# Confusion matrix for test results to observe edge cases
y_true = test_gen.classes #True Labels
y_prob = best_model.predict(test_gen, verbose=0)
y_pred = y_prob.argmax(1) #Predicated Labels

cm = cm = confusion_matrix(y_true, y_pred)
cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)

plt.figure(figsize=(6,5))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False,
            annot_kws={"size": 13, "weight": "bold"})
plt.title('Confusion Matrix for Tuned Model', fontsize=14, weight='bold')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()
