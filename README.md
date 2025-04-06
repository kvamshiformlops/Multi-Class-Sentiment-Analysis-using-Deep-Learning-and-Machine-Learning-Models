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

Got it! Here's the refined list, keeping only the models for which you've provided metrics:

---

#### Machine Learning Models

1. **SVM (Support Vector Machine)**:
   - Kernel: Radial Basis Function (RBF).
   - Regularization Parameter: 100.

2. **Logistic Regression**:
   - A simple yet effective linear model widely used for binary and multi-class classification.

3. **Random Forest Classifier**:
   - An ensemble of decision trees for robust predictions.

4. **Gradient Boosting Classifiers**:
   - Includes:
     - **LightGBM**: Speed-efficient boosting model for large datasets.
     - **CatBoost**: Optimized for categorical features.
     - **XGBoost**: High-performance boosting with parameter control.

5. **SGD Classifiers**:
   - Loss Functions:
     - **Hinge Loss**: Equivalent to SVM optimization.
     - **Log Loss**: Designed for probabilistic outputs in logistic regression.

6. **LinearSVC (Linear Support Vector Classifier)**:
   - Optimized for high-dimensional datasets.

7. **MultinomialNB (Naive Bayes)**:
   - Probabilistic model based on Bayes’ theorem, suitable for text classification.

8. **HistGradientBoosting Classifier**:
   - Gradient-boosted model offering interpretability and efficiency.

9. **MLP (Multilayer Perceptron)**:
   - A feedforward neural network with adaptive learning rates.

### 🎯 Evaluation Metrics
Models are assessed using:
- **Accuracy**: Ratio of correct predictions to total predictions.
- **Precision**: True Positives / (True Positives + False Positives).
- **Recall**: True Positives / (True Positives + False Negatives).
- **F1 Score**: Harmonic mean of precision and recall.

## 📊 Results

### 🔢 Deep Learning Performance Summary

| Model     | Accuracy (%) | F1 Score (%) | Precision (%) | Recall (%) |
|-----------|--------------|--------------|----------------|-------------|
| LSTM      | 88.78        | 88.71        | 88.74          | 88.78       |
| GRU       | 89.11        | 89.07        | 89.11          | 89.11       |
| Bi-LSTM   | 89.10        | 89.04        | 89.10          | 89.10       |
| Bi-GRU    | 88.38        | 88.11        | 88.47          | 88.38       |
| RCNN      | 89.05        | 88.92        | 89.01          | 89.05       |
| CNN       | 88.34        | 88.41        | 88.58          | 88.34       |
| RNN       | 33.98        | 17.24        | 70.14          | 33.98       |

### 🧮 Transformer Model Performance

| Model         | Accuracy (%) | F1 Score (%) | Precision (%) | Recall (%) |
|---------------|--------------|--------------|----------------|-------------|
| DistilBERT    | 87.80        | 87.94        | 88.39          | 87.80       |
| RoBERTa       | 87.54        | 87.60        | 87.77          | 87.54       |


### Machine Learning Models

| Model                          | Accuracy | F1 Score | Precision | Recall |
|--------------------------------|----------|----------|----------|----------|
| SVM (C=100)                    | 88.87    | 88.80    | 88.80    | 89.87    |
| Logistic Regression            | 89.28    | 89.18    | 89.22    | 89.28    |
| Random Forest Classifier       | 84.66    | 84.21    | 84.37    | 84.66    |
| SGD Classifier (Hinge Loss)    | 86.56    | 85.99    | 86.56    | 86.56    |
| SGD Classifier (Log Loss)      | 83.82    | 82.82    | 84.37    | 83.82    |
| Linear SVC                     | 88.91    | 88.73    | 88.79    | 88.91    |
| Nu-SVC                         | 84.86    | 84.71    | 84.65    | 84.86    |
| MultinomialNB                  | 78.39    | 77.56    | 78.86    | 78.39    |
| GradientBoosting Classifier    | 84.65    | 84.25    | 84.87    | 84.65    |
| LightGBM                       | 88.46    | 88.34    | 88.34    | 88.46    |
| HistGradientBoosting Classifier| 87.53    | 87.36    | 87.40    | 87.53    |
| XGBoost Classifier             | 83.42    | 82.89    | 83.93    | 83.42    |
| CatBoost Classifier            | 77.50    | 75.24    | 81.10    | 77.50    |
| MLP Classifier                 | 87.65    | 87.65    | 87.67    | 87.65    |

### 🏆Key Findings

- **Deep Learning Models** continue to demonstrate superior performance over traditional machine learning models, especially in terms of accuracy and recall. Among deep learning models:
  - **GRU** slightly edges out other architectures with the highest accuracy at 89.11%.
  - **Bi-LSTM** follows closely, making it another strong contender for robust performance.
  - **RNN**, however, exhibits significantly lower results, likely due to its inability to effectively capture complex dependencies in datasets.

- **Transformer Models**:
  - Perform consistently well, with **DistilBERT** achieving a higher F1 score compared to **RoBERTa**.
  - Their ability to handle textual datasets efficiently highlights their growing importance in deep learning applications.

- **Traditional Machine Learning Models**:
  - **Logistic Regression** emerges as the top-performing traditional model, matching the performance of **LinearSVC**, both nearing deep learning metrics.
  - **SVM (C=100)** also performs well, achieving commendable scores across all metrics.
  - Models like **CatBoost Classifier** and **MultinomialNB** show relatively lower performance, suggesting they might not be well-suited for this specific dataset.

- **Overall Observation**: Deep learning models remain the leaders for complex tasks, while well-tuned traditional models like Logistic Regression and SVM prove to be effective alternatives in simpler scenarios..

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
- **Real-Time Deployment**: Implement the best-performing model as a real-time sentiment analysis tool.
- **Model Interpretability**: Develop tools to explain model predictions for better transparency and trust.

## 📜 License

This project is licensed under the MIT License - see the LICENSE.md file for details.
