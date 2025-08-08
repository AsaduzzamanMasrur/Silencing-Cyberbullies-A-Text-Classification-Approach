# Silencing-Cyberbullies-A-Text-Classification-Approach

# 🧠 Text Classification with Deep Learning and ML

## Project Summary
In the landscape of sentiment analysis, this project explores the effectiveness
of two distinct models, Support Vector Machine (SVM) and Bidirectional Long
Short-Term Memory (BiLSTM), in discerning sentiments within textual data.
The primary objective is to evaluate and compare the performance of these
models, ultimately selecting the most robust one for deployment.

The rationale behind this exploration lies in the growing importance of sentiment analysis in understanding user opinions, emotions, and trends across
diverse domains. Businesses, social platforms, and decision-makers increasingly
rely on sentiment analysis to glean valuable insights from the vast pool of textual data generated daily. Consequently, choosing an adept model becomes
imperative for accurate and nuanced sentiment classification.

The project employs a meticulous methodology encompassing model training, evaluation, and deployment. Both SVM and BiLSTM undergo rigorous
training processes, with hyperparameter tuning and iterative adjustments to
optimize their performance. The evaluation phase employs comprehensive metrics to compare accuracy, precision, recall, and F1 score, providing a holistic
view of each model’s proficiency.

In the deployment phase, the chosen model, BiLSTM, is integrated into a
user-friendly Flask-based web interface, ensuring accessibility for end-users. The
executive summary encapsulates the project’s essence, highlighting the significance of sentiment analysis and articulating the project’s methodology. The
selected approach aligns with the overarching goal of providing valuable insights
into sentiment patterns, contributing to the broader field of natural language
processing and enhancing the decision-making processes dependent on textual
sentiment interpretation.

## 📌 Project Objectives

- Build and train models to classify textual descriptions into labels.
- Compare performance between a BiLSTM deep learning model and an SVM classifier.
- Deploy the final model as a web app using Flask.

## 🛠️ Tech Stack & Tools

- **Languages & Frameworks**: Python, Flask
- **Libraries**: 
  - Data Handling: Pandas, NumPy
  - Visualisation: Matplotlib, Seaborn
  - ML/NLP: Scikit-learn, Keras, TensorFlow, Gensim
  - Deployment: Flask, Ngrok
 **Models**: 
  - Support Vector Machine (SVM)
  - Bidirectional LSTM (BiLSTM)
 
 ## Pipeline Steps
Data Loading – Importing and exploring the dataset.

Preprocessing – Cleaning text, label encoding.

Feature Engineering – TF-IDF and Word2Vec embeddings.

Model Training – Using both:

Traditional ML: SVM

Deep Learning: BiLSTM (with Word2Vec & Keras)

Evaluation – Accuracy, precision, recall, F1-score, classification report.

Deployment – Flask app interface to input text and get predictions.

Screenshots:
1.


Research Paper:

