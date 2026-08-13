# FAQ Chatbot

A simple FAQ chatbot I built in Python. It cleans up the question using NLTK,
converts the FAQs into TF-IDF vectors, and picks the closest one with cosine
similarity. The Streamlit UI shows the answer plus a confidence score.

## Run it

cd chatbot
pip install -r requirements.txt
python -m nltk.downloader punkt punkt_tab stopwords
streamlit run app.py

## Files
- `faq_data.py` – the list of FAQs
- `preprocess.py` – text cleaning / tokenizing
- `chatbot.py` – the matching logic
- `app.py` – the Streamlit interface
