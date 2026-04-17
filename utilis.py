from PyPDF2 import PdfReader
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

def extract_text_from_pdf(file):
    text = ""
    pdf_reader = PdfReader(file)
    
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    return text

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    
    stop_words = set(stopwords.words('english'))
    words = text.split()
    
    filtered_words = [w for w in words if w not in stop_words]
    
    return " ".join(filtered_words)