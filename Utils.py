import json
import numpy
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.optimizers import Adam



def load_chatbot_data(json_path):
    '''
    This function loads the data that is in json format and extract the entitie that we
    need
    '''
    training_sentences = []
    training_labels = []
    labels = []
    responses = []

    with open(json_path, 'r') as file:
        data = json.load(file)

    for intent in data['intents']:
        tag = intent['tag']
        if tag not in labels:
            labels.append(tag)

        for pattern in intent['patterns']:
            training_sentences.append(pattern)
            training_labels.append(tag)

        responses.append({tag: intent['responses']})

    return training_sentences, training_labels, labels, responses


def generate_vectors(sentences -> array):
    vocab_size = 1000
    #embedding_dim = 32
    max_len = 20
    oov_token = "<OOV>"

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)

    tokenizer.fit_on_texts(sentences)
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(sentences)
    Training_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)
    # to save the fitted tokenizer
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return Training_sequences

class TextClassifier:
    def __init__(self, vocab_size, embedding_dim, max_len, num_classes, learning_rate=0.001):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_len))
        model.add(GlobalAveragePooling1D())
        model.add(Dense(16, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(self.num_classes, activation='softmax'))

        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=Adam(learning_rate=self.learning_rate),
            metrics=['accuracy']
        )

        return model

    def summary(self):
        return self.model.summary()

    def train(self, padded_sequences, labels, epochs=10, batch_size=32, validation_data=None):
        return self.model.fit(
            padded_sequences,
            np.array(labels),
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data
        )

    def predict(self, new_data):
        return self.model.predict(new_data)

    def save(self, path):
        self.model.save(path)

    def load_weights(self, path):
        self.model.load_weights(path)

