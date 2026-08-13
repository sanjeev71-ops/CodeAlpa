CodeAlpa

A collection of projects developed as part of my **CodeAlpha Internship**.

Projects

1. FAQ Chatbot

A simple FAQ chatbot built in Python using **NLTK, TF-IDF, cosine similarity, and Streamlit**.

The chatbot preprocesses user questions, converts the FAQ dataset into TF-IDF vectors, and identifies the most similar question using cosine similarity. The Streamlit interface displays the predicted answer along with a confidence score.

**Technologies:**
- Python
- NLTK
- Scikit-learn
- TF-IDF
- Cosine Similarity
- Streamlit

**Project:** [`chatbot/`](chatbot/)

See [`chatbot/README.md`](chatbot/README.md) for setup instructions and details.

---

2. AI Music Generation

An AI-based music generation project that learns patterns from MIDI melodies and generates new musical sequences.

The project uses the **Tegridy MIDI Dataset**, extracts representative musical events, encodes them into a vocabulary, and trains an LSTM-based neural network to predict the next musical event.

The trained model generates new sequences of notes and durations, which can then be converted back into MIDI music.

**Technologies:**
- Python
- PyTorch
- TensorFlow
- music21
- NumPy
- Pandas
- Scikit-learn
- LSTM
- MIDI

**Project:** [`ai-music-generation/`](ai-music-generation/)

See [`ai-music-generation/README.md`](ai-music-generation/README.md) for details about the dataset, preprocessing, model architecture, training, and generation process.

---

## Repository Structure

```text
CodeAlpa/
│
├── chatbot/
│   ├── app.py
│   ├── chatbot.py
│   ├── faq_data.py
│   ├── preprocess.py
│   ├── requirements.txt
│   └── README.md
│
├── ai-music-generation/
│   ├── AI_Music_Generation.ipynb
│   ├── README.md
│   ├── .gitignore
│   └── processed/
│       ├── melody_vocabulary.txt
│       ├── midi_file_statistics.csv
│       └── training_files.txt
│
└── README.md
