# Sentiment Analysis of Tweets

## Overview

This project focuses on sentiment analysis of tweets to classify sentiments into three categories: Positive, Negative, and Neutral. The project involves data preprocessing, feature extraction using TF-IDF vectorization, and classification using Naive Bayes.

## Project Structure

1. **Data Loading and Preprocessing:**
   - `loading_tweets(file_name)`: Loads tweet data from a CSV file and filters out irrelevant sentiments.
   - `preprocessing_tweets(tweets)`: Preprocesses tweets by removing special characters, converting text to lowercase, removing stop words, and applying stemming.

2. **Feature Extraction:**
   - **TF-IDF Vectorization:** Converts preprocessed text data into numerical features.

3. **Classification:**
   - **Naive Bayes Model:** Trains a Multinomial Naive Bayes classifier on the TF-IDF features and sentiment labels.

4. **Evaluation:**
   - Computes and prints training and testing accuracies of the Naive Bayes model.
## Results

**Without Neutral Sentiment**:
- Training Accuracy: 91.607%.
- Testing Accuracy: 94.107%.

**With Neutral Sentiment**:
- Training Accuracy: 81.944%.
- Testing Accuracy: 83.092%.


