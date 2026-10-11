
import pandas as pd
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "error_dataset.csv"

df = pd.read_csv(DATASET_PATH)

required_columns = [
    "error_log", "error_type", "language", "cause", "solution"
]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Missing dataset column: {column}")

df = df.dropna(subset=required_columns).reset_index(drop=True)


def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s.:']", " ", text)
    return text.strip()


# Train the error classification model.
X = df["error_log"].apply(preprocess_text)
y = df["error_type"]

classifier = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("model", LinearSVC(class_weight="balanced"))
])
classifier.fit(X, y)

# Build the similarity index.
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
error_vectors = vectorizer.fit_transform(X)

# Recognize common programming exception names.
EXCEPTION_PATTERN = re.compile(
    r"\b[A-Za-z_]\w*(?:Error|Exception)\b"
    r"|NullPointerException"
    r"|ClassNotFoundException"
    r"|NoClassDefFoundError"
    r"|StackOverflowError"
    r"|OutOfMemoryError"
    r"|Traceback \(most recent call last\)"
    r"|SyntaxError"
    r"|IndentationError",
    re.IGNORECASE
)


def extract_error_details(error_log):
    exceptions = list(dict.fromkeys(
        re.findall(
            r"\b[A-Za-z_]\w*(?:Error|Exception)\b",
            error_log
        )
    ))

    line_numbers = re.findall(
        r"\bline\s+(\d+)\b",
        error_log,
        flags=re.IGNORECASE
    )

    filenames = re.findall(
        r'File ["\']([^"\']+)["\']',
        error_log
    )

    return {
        "exceptions_found": exceptions,
        "line_numbers": line_numbers,
        "filenames": filenames
    }


def analyze_error(error_log):
    if not isinstance(error_log, str) or not error_log.strip():
        return {"error": "Please enter a programming error log."}

    cleaned_log = preprocess_text(error_log)
    details = extract_error_details(error_log)

    # Reject ordinary text without recognizable error evidence.
    has_exception = bool(EXCEPTION_PATTERN.search(error_log))
    has_traceback = "traceback" in error_log.lower()
    has_stack_trace = " at " in error_log and (
        ".java:" in error_log or ".py:" in error_log
    )
    has_error_keyword = bool(re.search(
        r"\b(error|exception|traceback|failed|failure)\b",
        error_log,
        re.IGNORECASE
    ))

    if not (has_exception or has_traceback or has_stack_trace or has_error_keyword):
        return {
            "error": (
                "This input does not appear to be a programming error log. "
                "Please paste an actual Python or Java error message."
            ),
            **details
        }

    input_vector = vectorizer.transform([cleaned_log])
    similarities = cosine_similarity(
        input_vector, error_vectors
    )[0]

    best_index = int(similarities.argmax())
    similarity_score = float(similarities[best_index])

    # Reject error-like text that is too dissimilar to known examples.
    if similarity_score < 0.10:
        return {
            "error": (
                "This error is not recognized in the current dataset. "
                "Try adding more examples or checking the error message."
            ),
            **details
        }

    predicted_type = classifier.predict([cleaned_log])[0]
    match = df.iloc[best_index]

    return {
        "input": error_log,
        "predicted_type": predicted_type,
        "matched_error": match["error_log"],
        "language": match["language"],
        "cause": match["cause"],
        "solution": match["solution"],
        "similarity": round(similarity_score * 100, 2),
        **details
    }


if __name__ == "__main__":
    test_logs = [
        "ZeroDivisionError: division by zero",
        "NameError: name 'total' is not defined",
        "Exception in thread main java.lang.NullPointerException",
        "Today is a beautiful day and the weather is sunny."
    ]

    for log in test_logs:
        print("\nInput:", log)
        result = analyze_error(log)

        if "error" in result:
            print("Result:", result["error"])
        else:
            print("Predicted error:", result["predicted_type"])
            print("Language:", result["language"])
            print("Cause:", result["cause"])
            print("Solution:", result["solution"])
