from sklearn.preprocessing import LabelEncoder
import pickle
from Utils import load_chatbot_data, generate_vectors

# Load training data from JSON
training_sentences, training_labels, labels, responses = load_chatbot_data("data.json")

# Encode labels to integers
lbl_encoder = LabelEncoder()
lbl_encoder.fit(training_labels)
training_labels_encoded = lbl_encoder.transform(training_labels)

#save our label encoder for later use
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(lbl_encoder, f)
print("Successfully saved label_encoder")


