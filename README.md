# Spam Email Detection

This project is a spam classifier for emails and SMS messages. It uses machine learning algorithms to predict whether a message is spam or not. It includes both a web app interface for real-time classification and a Jupyter notebook for model training and analysis.

## Features
- Classifies emails and SMS as spam or not spam.
- Web interface built using Streamlit.
- Uses a machine learning model trained on labeled spam data.

## Technologies Used
- Python
- Streamlit
- scikit-learn
- NLTK
- pandas
- numpy

## Setup Instructions

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/Spam-Email-Detection.git
    ```

2. Navigate to the project folder:
    ```bash
    cd Spam-Email-Detection
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

5. Access the web interface at `http://localhost:8501` (or the URL provided after running Streamlit).

## Data Files
- `spam.csv`: Contains labeled data for training the spam classification model.
- `model.pkl`: The trained machine learning model.
- `vectorizer.pkl`: The trained TF-IDF vectorizer used for text transformation.

## How to Use
- Open the Streamlit web app.
- Enter an email or SMS message into the input field.
- Click the "Classify" button to check if the message is spam or not.

