from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(documents)
    
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
    
    scores = similarity_scores.flatten()
    
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    
    return ranked