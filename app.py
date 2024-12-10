import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Ensure required NLTK data is downloaded


ps = PorterStemmer()

# Text preprocessing function
def transform_text(text):
    text = text.lower()  # Convert to lowercase
    text = nltk.word_tokenize(text)  # Tokenize text
    y = []
    for i in text:
        if i.isalnum():  # Remove non-alphanumeric characters
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:  # Remove stopwords and punctuation
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))  # Perform stemming

    return " ".join(y)

# Load the saved TF-IDF vectorizer and model
tfidf = pickle.load(open(r"C:\Users\rohan\OneDrive\Desktop\Spam-Email-detection\vectorizer.pkl", 'rb'))
model = pickle.load(open(r"C:\Users\rohan\OneDrive\Desktop\Spam-Email-detection\model.pkl", 'rb'))

# Streamlit app interface
st.set_page_config(
    page_title="Spam Classifier",
    page_icon="📧",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Page Title
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>📧 Email/SMS Spam Classifier</h1>", unsafe_allow_html=True)

# Add a horizontal divider
st.markdown("---")

# Sidebar Information
with st.sidebar:
    st.header("About This App")
    st.write("""
        This app uses **Natural Language Processing (NLP)** techniques and a **Machine Learning model** to classify whether a given text is **Spam** or **Not Spam**.
    """)
    st.write("Built with ❤️ by **Rohan Sabale**.")
    st.markdown("[GitHub Repository](https://github.com/)", unsafe_allow_html=True)  # Update the link with your repo
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Developer")
    st.write("👤 **Rohan Sabale**")

# Input Section
st.markdown("<h3 style='text-align: center;'>📝 Enter Your Message Below</h3>", unsafe_allow_html=True)
input_sms = st.text_area("Enter the message you want to classify:", height=150)

# Process Input
if st.button("Classify"):
    if input_sms.strip():  # Ensure input is not empty
        with st.spinner("Classifying..."):
            # 1. Preprocess the text
            transformed_sms = transform_text(input_sms)

            # 2. Vectorize the text
            vector_input = tfidf.transform([transformed_sms])

            # 3. Predict using the model
            result = model.predict(vector_input)[0]

        # 4. Display the result
        st.markdown("<hr>", unsafe_allow_html=True)
        if result == 1:
            st.markdown("<h2 style='text-align: center; color: red;'>🚨 This is Spam!</h2>", unsafe_allow_html=True)
        else:
            st.markdown("<h2 style='text-align: center; color: green;'>✅ This is Not Spam!</h2>", unsafe_allow_html=True)
    else:
        st.warning("Please enter a message to classify.")

# Footer Section
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; font-size: 14px;'>© 2024 Rohan Sabale. All rights reserved.</p>",
    unsafe_allow_html=True,
)
