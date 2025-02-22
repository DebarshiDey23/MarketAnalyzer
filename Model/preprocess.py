import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data files (run this once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    """
    Cleans and preprocesses financial text data.
    1. Removes special characters, links, and numbers.
    2. Tokenizes text.
    3. Removes stop words.
    4. Converts words to their root form using lemmatization.
    """
    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

    # Remove special characters, punctuation, and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Join tokens back into a string
    cleaned_text = " ".join(tokens)
    
    return cleaned_text

# Example usage
sample_text = "Breaking: AAPL stock soars 5% after earnings report! 🚀 #Investing https://finance.yahoo.com"
cleaned_text = preprocess_text(sample_text)
print(cleaned_text)  # Output: "breaking aapl stock soar earning report investing"
