import json
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences



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
    embedding_dim = 32
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

