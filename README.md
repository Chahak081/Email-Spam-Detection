#  Spam-Detection
The purpose of this work is to demonstrate a simple email spam classification model using logistic regression and TF-IDF (Term Frequency-Inverse Document Frequency) vectorization. It reads a CSV file ('mail_data.csv') containing email messages labeled as 'spam' or 'ham' (non-spam) and trains a logistic regression model to classify emails as spam or ham. It also evaluates the model's accuracy on training and testing data. Additionally, it allows users to input their own email messages for prediction using a Streamlit web application.

# Code Structure:
* Importing Libraries: Import the necessary libraries - pandas, numpy, scikit-learn's train_test_split, TfidfVectorizer, LogisticRegression, and accuracy_score.

* Reading and Exploring Data: Read the CSV file ('mail_data.csv') into a pandas DataFrame called 'df'. Display the DataFrame and a sample of 5 rows.

* Data Preprocessing: Modify the DataFrame 'df' by replacing any missing values with empty strings. Display information about the DataFrame.

* Data Transformation: Convert the 'Category' column to numerical labels, where 'spam' is encoded as 0 and 'ham' is encoded as 1.

* Splitting Data: Split the data into training and testing sets using the train_test_split function from scikit-learn.

* Feature Extraction: Transform the text data into numerical features using the TF-IDF vectorizer.

* Model Training: Create and train a logistic regression model on the training data.

* Model Evaluation: Evaluate the accuracy of the model on both the training and testing data.

* Making Predictions on Test Data: Provide a sample email message specified in dataset as 'input_your_mail' and predict whether it is spam or ham.

* Streamlit Deployment: Create a Streamlit web application to allow users to input email messages (sample examples from test data ['mail_data.csv']) and display the prediction.

# Usage of NLP (Natural Language Processing):
* The 'Message' column containing text data is transformed into numerical features using TF-IDF vectorization.
* TF-IDF stands for Term Frequency-Inverse Document Frequency, which is a numerical representation of a document (email) based on the importance of each word in the document relative to a collection of documents (corpus).
* It is a common technique in Natural Language Processing (NLP) to convert textual data into numerical form, making it suitable for machine learning algorithms.
* By using TF-IDF, the model learns to identify important keywords or features in emails that distinguish spam from non-spam


# Code Funcgtionality:
* Follow the .ipynb file for code
* For deployment run given command in terminal to land onto the local host-
```
streamlit run app.py
```


## Contributing :↓

Contributions are welcome! If you find a bug or have a feature request, please open an issue. Pull requests are also welcome.
