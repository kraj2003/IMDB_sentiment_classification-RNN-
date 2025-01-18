<h2>Sentiment Analysis Using RNN</h2>

<h3>Table of Contents</h3>

- [About the Project](#abouttheproject)

- [Features](#Features)

- [Technologies Used](#technologies used)

- [Dataset](#dataset)

- [Model Architecture](#modelartichecture)

- [Installation](#installation)

8. Usage

9. Contributing

License

Contact

## <h3>About the Project</h3>

This project is a sentiment analysis model built using Recurrent Neural Networks (RNN) to analyze user feedback. It determines whether feedback expresses positive or negative sentiment. The IMDB dataset from the Keras library is used for training and testing the model. The model leverages word embeddings to represent textual data as vectors and is designed to process sequences effectively using an RNN.

## Features

- Sentiment classification of user feedback (positive/negative).

- Utilizes the IMDB dataset with preprocessed reviews.

- Processes text sequences with padding (maximum sequence length = 500).

- Embedding layer to convert words into dense vector representations.

- Implements RNN for sequential data modeling.

- Achieves reliable accuracy on the IMDB dataset.

## <h3>Technologies Used</h3>

+ Python 3.8+

+ TensorFlow 2.x

+ Keras

+ NumPy

+ Pandas

<h3>Dataset</h3>

+ Dataset Name: IMDB Movie Reviews Dataset

+ Source: Downloaded from the Keras library.

+ Description: A dataset containing 50,000 highly polar movie reviews (25,000 for training and 25,000 for testing) labeled as positive or negative.

<h3>Preprocessing:</h3>

+ Padded sequences to a maximum length of 500 tokens.

+ Tokenized text data converted to numerical form.

<h3>Model Architecture</h3>

1. Embedding Layer:

  + Converts words into dense vector representations of fixed size.

  + Input dimension: Vocabulary size (number of words in the dataset).

  + Output dimension: Embedding size.

2. Recurrent Neural Network (RNN):

  + Processes sequential data to capture temporal dependencies.

  + Output from the RNN is passed to a dense layer for final predictions.

  + Dense Layer with Sigmoid Activation:

Outputs the probability of positive sentiment.

Summary of Model Layers:

Input Layer: Padded sequences.

Embedding Layer: Word-to-vector conversion.

RNN Layer: Sequential data modeling.

Dense Layer: Classification.

Installation

Clone the repository:

git clone https://github.com/your-username/sentiment-analysis-rnn.git

Navigate to the project directory:

cd sentiment-analysis-rnn

Install the required dependencies:

pip install -r requirements.txt

Usage

Train the model:

python train.py

This script loads the IMDB dataset, preprocesses the data, trains the RNN model, and saves the trained model.

Evaluate the model:

python evaluate.py

This script evaluates the model's performance on the test dataset.

Analyze feedback:

python predict.py --feedback "Your feedback text here"

Replace "Your feedback text here" with the actual feedback you want to analyze.

Contributing

Contributions are welcome! Follow these steps:

Fork the project.

Create a feature branch:

git checkout -b feature/your-feature-name

Commit your changes:

git commit -m "Add your feature description"

Push to the branch:

git push origin feature/your-feature-name

Open a pull request.

License

This project is licensed under the MIT License. See the LICENSE file for details.

Contact

Name: Your Name

Email: your.email@example.com

GitHub: your-username
