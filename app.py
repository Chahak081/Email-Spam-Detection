import streamlit as st
import joblib

def load_model():
    # Load the trained model and TfidfVectorizer
    model = joblib.load('spam_classifier_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

def classify_spam_ham(model, vectorizer, message):
    # Preprocess the input message and make predictions
    input_data_features = vectorizer.transform([message])
    prediction = model.predict(input_data_features)[0]
    return "ham" if prediction == 1 else "spam"

def main():
    st.title("SMS Spam Classifier")

    # Load the model and vectorizer
    model, vectorizer = load_model()

    # Get user input
    user_input = st.text_area("Type your message here:", "")

    if st.button("Classify"):
        if user_input.strip() != "":
            result = classify_spam_ham(model, vectorizer, user_input)
            st.write(f"Prediction: {result}")

if __name__ == "__main__":
    main()
