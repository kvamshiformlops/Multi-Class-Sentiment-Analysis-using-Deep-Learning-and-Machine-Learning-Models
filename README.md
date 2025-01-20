# Multi-Class Sentiment Analysis using Deep Learning and Machine Learning Models

This repository contains a comprehensive analysis of multi-class sentiment analysis using a variety of deep learning and traditional machine learning algorithms. The primary goal is to evaluate and compare the performance of these models on a real-world dataset, ultimately determining the most effective approach for sentiment classification.

### Dataset
- **Source**: The HappyDB dataset, a large-scale collection of human-generated happy moments sourced from social media.

### 🔍 Preprocessing
The raw text data undergoes extensive preprocessing to ensure optimal performance:
- Removing HTML tags.
- Converting text to lowercase.
- Eliminating non-alphabetic characters.
- Tokenizing text into individual words.
- Removing common stop words using NLTK.
- Lemmatizing words to their base forms with WordNet lemmatizer.

### 🔧 Feature Engineering
- **TF-IDF Vectorization**: Converts textual data into numerical format. Parameters:
  - Maximum Features: 5000
  - N-grams: (1,2)
  - Minimum Document Frequency: 2
  - Maximum Document Frequency: 90%
- **Word Embeddings**: 
  - Tokenized and padded sequences.
  - Embedding Dimension: 128
  - Maximum Sequence Length: 100

### Model Selection
####  🧠 Deep Learning Models
1. **Vanilla LSTM (Long Short-Term Memory)**:
   - Recurrent neural network architecture designed for sequential data processing.
   - Contains multiple LSTM layers for robust context capture.
2. **CNN (Convolutional Neural Network)**:
   - Uses Conv1D layers to extract n-gram features.
   - Employs GlobalMaxPooling for dimensionality reduction.
3. **RNN (Recurrent Neural Network)**:
   - A simple recurrent network prone to vanishing gradient issues.
4. **GRU (Gated Recurrent Unit)**:
   - A lightweight version of LSTM with comparable results.
5. **Bi-LSTM (Bidirectional LSTM)**:
   - Processes sequences in both forward and backward directions.
6. **Bi-RNN (Bidirectional RNN)**:
   - Extends RNNs with bidirectional information flow.
7. **Bi-GRU (Bidirectional GRU)**:
   - Bidirectional implementation of GRU for better context awareness.
8. **Self-Attention Network**:
   - Utilizes self-attention mechanisms to weigh word importance dynamically.
9. **RCNN (Recurrent Convolutional Neural Network)**:
   - Combines the strengths of RNNs and CNNs for enriched feature extraction.

#### Machine Learning Models
1. **Cu-SVM (CUDA Support Vector Machine)**:
   - GPU-accelerated SVM for faster computations.
2. **Logistic Regression**:
   - A linear model for classification.
3. **SVM (Support Vector Machine)**:
   - Kernel: Radial Basis Function (RBF).
   - Regularization Parameter: 1.0.
4. **Random Forest Classifier**:
   - Ensemble of decision trees for robust predictions.
5. **Gradient Boosting Classifiers**:
   - Includes LightGBM, XGBoost, and CatBoost.
6. **SGD Classifiers**:
   - Loss Functions: Hinge (for SVM), Log Loss (for Logistic Regression).
7. **LinearSVC (Linear Support Vector Classifier)**:
   - Optimized for high-dimensional datasets.
8. **MLP (Multilayer Perceptron)**:
   - Feedforward neural network with adaptive learning rate.
9. **Decision Tree Classifier**:
   - Simple yet interpretable tree-based model.
10. **Naive Bayes Classifier**:
   - Probabilistic model based on Bayes' theorem.
11. **AdaBoost Classifier**:
   - Sequential boosting of weak learners.
12. **K-NN (K-Nearest Neighbors)**:
   - Non-parametric method for classification.

### 🎯 Evaluation Metrics
Models are assessed using:
- **Accuracy**: Ratio of correct predictions to total predictions.
- **Precision**: True Positives / (True Positives + False Positives).
- **Recall**: True Positives / (True Positives + False Negatives).
- **F1 Score**: Harmonic mean of precision and recall.

## 📊 Results

### Deep Learning Models

| Model                  | Accuracy | F1 Score | Precision | Recall |
|------------------------|----------|----------|-----------|--------|
| Vanilla LSTM           | 88.78%   | 88.71%   | 88.74%    | 88.78% |
| CNN                    | 88.34%   | 88.41%   | 88.58%    | 88.34% |
| RNN                    | 33.98%   | 17.24%   | 70.14%    | 33.98% |
| GRU                    | 89.11%   | 89.07%   | 89.11%    | 89.11% |
| Bi-LSTM                | 89.10%   | 89.04%   | 89.10%    | 89.10% |
| Bi-RNN                 | 85.20%   | 85.56%   | 86.81%    | 85.20% |
| Bi-GRU                 | 88.38%   | 88.11%   | 88.28%    | 88.38% |
| Self-Attention Network | 89.07%   | 88.96%   | 88.98%    | 89.07% |
| RCNN                   | 89.13%   | 89.12%   | 89.13%    | 89.13% |

### Machine Learning Models

| Model                               | Accuracy | F1 Score | Precision | Recall |
|-------------------------------------|----------|----------|-----------|--------|
| Cu-SVM                              | 88.85%   | 88.76%   | 88.78%    | 88.85% |
| Logistic Regression                 | 89.46%   | 89.38%   | 89.40%    | 89.46% |
| SVM                                 | 88.78%   | 88.68%   | 88.71%    | 88.78% |
| SVM (with C=100)                    | 89.03%   | 88.98%   | 88.97%    | 89.03% |
| Random Forest Classifier            | 84.80%   | 84.35%   | 84.50%    | 84.80% |
| SGD Classifier (Hinge Loss)         | 86.64%   | 86.07%   | 86.64%    | 86.64% |
| SGD Classifier (Log Loss)           | 83.91%   | 82.93%   | 84.49%    | 83.91% |
| LinearSVC                           | 89.09%   | 88.92%   | 88.97%    | 89.09% |
| MLP Classifier                      | 87.94%   | 87.96%   | 88.00%    | 87.94% |
| Decision Tree Classifier            | 62.17%   | 56.06%   | 79.93%    | 62.17% |
| Naive Bayes Classifier              | 78.32%   | 77.51%   | 78.70%    | 78.32% |
| Gradient Boosting Classifier        | 84.80%   | 84.42%   | 85.04%    | 84.80% |
| LightGBM Classifier                 | 88.40%   | 88.27%   | 88.27%    | 88.40% |
| ExtraTrees Classifier               | 43.53%   | 35.62%   | 53.27%    | 43.53% |
| AdaBoost Classifier                 | 52.00%   | 44.08%   | 72.63%    | 52.00% |
| HistGradientBoostingClassifier      | 87.79%   | 87.62%   | 87.65%    | 87.79% |
| K-NN (K-Nearest Neighbors)          | 53.46%   | 50.20%   | 67.67%    | 53.46% |
| XGBoost Classifier                  | 83.50%   | 82.96%   | 83.98%    | 83.50% |
| CatBoost Classifier                 | 77.53%   | 75.35%   | 80.75%    | 77.53% |
| NuSVC                               | 85.04%   | 84.93%   | 84.89%    | 85.04% |

## 🏆 Key Findings
- **Deep Learning Models** generally outperform traditional machine learning models in terms of accuracy, precision, and recall.
- **Bi-LSTM and GRU** achieve the best results among deep learning models, with accuracy around 89%.
- **Logistic Regression and LinearSVC** are the most effective traditional machine learning models, achieving accuracies close to 89%.
- Certain models like RNN and ExtraTrees Classifier exhibit significantly lower performance due to their limitations with complex datasets.

## 🔧 Dependencies
To run this project, install the following dependencies:
- Python 3.8+
- TensorFlow 2.x
- Keras
- scikit-learn
- NLTK
- pandas
- numpy
- LightGBM
- XGBoost
- CatBoost


##  📈 Future Improvements
- **Dataset Expansion**: Include additional datasets to enhance model robustness.
- **Hyperparameter Tuning**: Perform grid search or Bayesian optimization for hyperparameter tuning.
- **Transfer Learning**: Leverage pre-trained language models like BERT or GPT for improved feature extraction.
- **Real-Time Deployment**: Implement the best-performing model as a real-time sentiment analysis tool.
- **Model Interpretability**: Develop tools to explain model predictions for better transparency and trust.

## 📜 License

This project is licensed under the MIT License - see the LICENSE.md file for details.
