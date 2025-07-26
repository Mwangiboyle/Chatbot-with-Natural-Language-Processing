# 🇰🇪 Kenyan Food Chatbot 🍲

A simple chatbot built with **Python**, **TensorFlow**, **scikit-learn**, and **FastAPI** to handle user interactions like greetings, food orders, and complaints for a Kenyan food restaurant. The chatbot uses machine learning to classify user intents and respond accordingly.

---

## 🚀 Features

- Intent classification using a neural network (TensorFlow)
- Natural language preprocessing (nltk + scikit-learn)
- Custom intent dataset (no need for large external datasets)
- FastAPI-based RESTful API for interaction
- Designed for food service scenarios (e.g. ordering *ugali*, *nyama choma*, *pilau*)

---

## 🧠 Project Structure

- `data.json`: Custom training data for chatbot intents
- `train_chatbot.py`: Script to preprocess data and train the ML model
- `model.h5`: Trained model saved for inference
- `chatbot_api.py`: FastAPI app to interact with the model via API
- `requirements.txt`: All necessary Python packages

---

## 📦 Requirements

Install dependencies using pip:

```bash
pip install -r requirements.txt

