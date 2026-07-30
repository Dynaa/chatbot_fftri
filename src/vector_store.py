from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class VectorStore:
    """
    Indexation et recherche vectorielle / TF-IDF des articles de la réglementation FFTRI.
    Permet de récupérer les n-chunks les plus pertinents pour une question ou situation.
    """
    def __init__(self, chunks: List[Dict[str, Any]]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            max_df=0.85,
            min_df=1,
            lowercase=True
        )
        self.corpus = [c["text"] for c in chunks]
        if self.corpus:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)
        else:
            self.tfidf_matrix = None

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Effectue une recherche par similarité cosinus sur la base d'articles.
        """
        if self.tfidf_matrix is None or not query.strip():
            return []

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Récupération des top_k meilleurs indices
        top_indices = scores.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0.01: # Seuil minimum de pertinence
                chunk = self.chunks[idx].copy()
                chunk["score"] = round(score, 4)
                results.append(chunk)

        return results
