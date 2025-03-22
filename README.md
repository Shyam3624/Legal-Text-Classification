# Legal-Text-Classification
# Overview  
The dataset provided for the Legal Citation Text Classification task consists of 25,000 legal 
cases represented as text documents. Each case is annotated with metadata, including catchphrases, 
citation sentences, citation catchphrases, and citation classes. The citation classes indicate the type 
of treatment given to the cases cited by the present case. This dataset is designed to facilitate the 
development of models for classifying legal texts based on their content and citation context. 
Dataset Structure 
The dataset is provided in CSV format and contains the following columns: 
# Dataset Link :  
https://www.kaggle.com/datasets/amohankumar/legal-text-classification-dataset/data  
# Description 
Case ID : A unique identifier for each legal case. 
Case Outcome :Indicates the outcome or result of the legal case (e.g., affirmed, reversed, etc.).
Case Title :The title or heading of the legal case. 
Case Text :The full text of the legal case, including details, arguments, and citations.
# Key Features of the Dataset 
• Catchphrases: Summarized phrases or key points extracted from the legal case. 

• Citation Sentences: Sentences within the case text reference other legal cases. 

• Citation Catchphrases: Summarized phrases or key points related to the cited cases. 

• Citation Classes: Indicates the treatment given to the cited cases (e.g., positive, negative, 
neutral, or other specific legal treatments). 

# Dataset Use Cases 
This dataset can be used for various natural language processing (NLP) tasks in the legal domain, 
including but not limited to: 
 •  Legal Text Classification: 
Classifying cases based on their outcomes, citation treatments, or 
legal domains. 
 • Citation Analysis:
Understanding how legal cases reference and treat prior cases. 
• Catchphrase Extraction:
Automatically identifying key points or summaries from legal texts. 
• Legal Research:
Assisting legal professionals in retrieving relevant cases based on citation 
# context or outcomes. 
# Preprocessing Steps 
To prepare the dataset for modelling, the following preprocessing steps are recommended: 
1. Text Cleaning:
Remove special characters, stop words, and irrelevant sections (e.g., headers, 
footers). 
2. Tokenization: 
Split the text into individual words or tokens. 
3. Normalization: 
Convert text to lowercase and handle abbreviations or legal-specific terms. 
4. Annotation Handling: 
Extract and utilize catchphrases, citation sentences, and citation 
classes as features. 
5. Class Balancing: 
Address class imbalance through techniques such as oversampling, under 
sampling, or data augmentation. 
6. Word Embedding Techniques: 
o Use pre-trained embeddings like Word2Vec, GloVe, or FastText to represent words in 
a dense vector space. 
o Fine-tune embeddings using domain-specific legal corpora for better representation 
of legal terminology. 
o Utilize transformer-based embeddings like BERT, RoBERTa, or Legal-BERT for 
contextualized word representations. (If possible)  
# Machine Learning Models 
1. Linear Regression 
2. Logistic Regression 
3. Support Vector Machines (SVM) 
4. Random Forest (RF) 
5. Decision Tree 
6. K-Nearest Neighbors (KNN) 
7. Boosting Algorithms: 
o AdaBoost (Adaptive Boosting) 
o LightGBM 
o XGBoost (Extreme Gradient Boosting) 
8. Multi-Layer Perceptron (MLP) 
# Deep Learning Models 
1. Deep Neural Networks (DNN) 
2. Recurrent Neural Networks (RNNs) 
3. Long Short-Term Memory (LSTM) 
4. Transformer-Based Models: 
o BERT (Bidirectional Encoder Representations from Transformers) 
o RoBERTa (Robustly Optimized BERT Pretraining Approach) 
o Legal-BERT
