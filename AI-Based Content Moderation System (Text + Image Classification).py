import pandas as pd
import numpy as np
import re
import nltk
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


nltk.download('stopwords')
from nltk.corpus import stopwords


data = pd.read_csv('content_moderation_dataset.csv')
df = pd.DataFrame(data)
print("\n Sample Input Data:")
print(df.head())


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

df['cleaned'] = df['text'].apply(clean_text)


vectorizer = TfidfVectorizer()
X_text = vectorizer.fit_transform(df['cleaned'])
y_text = df['label']

try:
    X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
        X_text, y_text, test_size=0.4, stratify=y_text, random_state=42
    )
except ValueError:
    print(" Not enough data to stratify. Proceeding without stratify.")
    X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
        X_text, y_text, test_size=0.4, random_state=42
    )

text_model = LogisticRegression(max_iter=1000, class_weight='balanced')
text_model.fit(X_train_t, y_train_t)


y_pred_t = text_model.predict(X_test_t)
print("\n --- Text Classification Report ---")
print(classification_report(y_test_t, y_pred_t))


sample_input = [
    "You are so dumb!",
    "Great work on the presentation!",
    "Free money now!!!"
]
sample_cleaned = [clean_text(txt) for txt in sample_input]
sample_vectorized = vectorizer.transform(sample_cleaned)
sample_prediction = text_model.predict(sample_vectorized)

print("\n Text Predictions:")
for i, txt in enumerate(sample_input):
    print(f"Input: '{txt}' --> Predicted Category: {sample_prediction[i]}")


def load_images_from_folder(folder_path, label):
    images = []
    labels = []
    for file in os.listdir(folder_path):
        path = os.path.join(folder_path, file)
        img = cv2.imread(path)
        if img is not None:
            img = cv2.resize(img, (128, 128))
            images.append(img)
            labels.append(label)
    return images, labels

categories = ['safe', 'offensive', 'spam', 'hate']
image_data = []
image_labels = []

for cat in categories:
    folder = f'images_dataset/{cat}'
    if os.path.exists(folder):
        imgs, lbls = load_images_from_folder(folder, cat)
        image_data.extend(imgs)
        image_labels.extend(lbls)

if image_data:
    X_img = np.array(image_data) / 255.0
    label_encoder = LabelEncoder()
    y_img = label_encoder.fit_transform(image_labels)

    
    model_img = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D(),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(len(categories), activation='softmax')
    ])

    model_img.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model_img.fit(X_img, y_img, epochs=10, validation_split=0.2)

    
    test_image_path = 'sample_test_image.jpg'
    if os.path.exists(test_image_path):
        test_img = cv2.imread(test_image_path)
        test_img_resized = cv2.resize(test_img, (128, 128))
        test_input = np.expand_dims(test_img_resized / 255.0, axis=0)
        prediction = model_img.predict(test_input)
        predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])[0]
        print(f"\n Predicted Image Category: {predicted_class}")
    else:
        print("\n No sample image found for prediction. Place a file named 'sample_test_image.jpg' in the directory.")
else:
    print("\n No image data found. Skipping image model training.")
