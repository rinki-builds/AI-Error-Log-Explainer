# AI-Powered Error Log Explainer

An NLP-based application that helps developers understand common programming errors and find possible solutions.

## Features

- Classifies common programming errors.
- Uses TF-IDF and a Linear Support Vector Machine (SVM).
- Finds similar errors from a reference dataset.
- Explains possible causes and solutions.
- Extracts useful details from error messages.
- Maintains analysis history and supports CSV export.

## Technologies Used

- Python
- Natural Language Processing (NLP)
- Pandas
- Scikit-learn
- Streamlit

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/rinki-builds/AI-Error-Log-Explainer.git
   ```

2. Open the project folder:

   ```bash
   cd AI-Error-Log-Explainer
   ```

3. Install the dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Run the application:

   ```bash
   python -m streamlit run app.py
   ```

5. Open the local URL displayed in the terminal.

## How It Works

The application preprocesses the submitted error text, classifies it using an NLP model, and searches for similar examples in the reference dataset to suggest possible causes and solutions.

**Note:** Results depend on the dataset and similarity matching. Suggested explanations should be verified by the developer.

## Future Improvements

- Expand the error dataset.
- Support more programming languages.
- Improve error classification and solution recommendations.
- Add more detailed debugging guidance.

## Author

Rinki

## GitHub Repository

[AI-Powered Error Log Explainer](https://github.com/rinki-builds/AI-Error-Log-Explainer)
