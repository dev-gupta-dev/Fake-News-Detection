import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Step 1: Load Dataset
df = pd.read_csv("fake_news.csv")  # rename your dataset file
df = df[['title', 'text', 'label']].dropna()

# Combine title and text
df['content'] = df['title'] + " " + df['text']

# Step 2: Tokenization
max_words = 10000
max_len = 200

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(df['content'])
sequences = tokenizer.texts_to_sequences(df['content'])
padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

# Step 3: Prepare Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(padded, df['label'], test_size=0.2, random_state=42)

# Step 4: Build Model
model = Sequential([
    Embedding(max_words, 128, input_length=max_len),
    LSTM(64, dropout=0.3, recurrent_dropout=0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# Step 5: Compile Model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Step 6: Train Model
history = model.fit(X_train, y_train, epochs=3, batch_size=64, validation_split=0.2, verbose=1)

# Step 7: Evaluate Model
loss, acc = model.evaluate(X_test, y_test)
print(f"\n✅ Test Accuracy: {acc:.4f}")

# Step 8: Classification Report
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.legend()
plt.show()

news = ["Government announces new AI policy to boost startups"]
seq = tokenizer.texts_to_sequences(news)
padded_seq = pad_sequences(seq, maxlen=max_len)
pred = model.predict(padded_seq)[0][0]

print("📰 News:", news[0])
print("Predicted:", "FAKE 😞" if pred > 0.5 else "REAL 🟩")
