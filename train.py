from sklearn.preprocessing import LabelEncoder
import pickle
from Utils import load_chatbot_data, generate_vectors, TextClassifier
from keras.layers import Dense, Embedding, GlobalAveragePooling1D, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping
import numpy as np
# Load training data from JSON
training_sentences, training_labels, labels, responses = load_chatbot_data("data.json")

print(f"Training sentences:{training_sentences[:5]}")
print(f"Training labels:{training_labels[:5]}")
print(f"labels:{labels}")

num_classes = len(labels)
# Encode labels to integers
lbl_encoder = LabelEncoder()
lbl_encoder.fit(training_labels)
training_labels_encoded = lbl_encoder.transform(training_labels)

print(f"Training_lables_encoded: {training_labels_encoded[:5]}")
#save our label encoder for later use
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(lbl_encoder, f)
print("Successfully saved label_encoder")


#convert words to vectors
training_vectors = generate_vectors(training_sentences)

print(f"The first 5 training vectors: {training_vectors[:5]}")

print(f"training_vectors_shape: {len(training_vectors)}")

# Instantiate
# Instantiate
vocab_size = 1000
embedding_dim =16
max_len = 20

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.3))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(16, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

print(model.summary())
epochs = 500
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
history = model.fit(training_vectors,
                    np.array(training_labels_encoded),validation_split=0.1,callbacks=[early_stop], epochs=epochs)


# to save the trained model
model.save("chat_model")

