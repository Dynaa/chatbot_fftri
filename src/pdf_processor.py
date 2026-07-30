import os
import re
import json
from typing import List, Dict, Any
from pypdf import PdfReader

CACHE_DIR = ".cache"
CACHE_FILE = os.path.join(CACHE_DIR, "parsed_chunks.json")

class PDFProcessor:
    """
    Lit et extrait les textes de la Réglementation Sportive FFTRI.
    Découpe le document en chunks intelligents en conservant la référence des articles.
    """
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path

    def process_and_chunk(self, chunk_size_words: int = 350, overlap_words: int = 50) -> List[Dict[str, Any]]:
        """
        Extrait et découpe le PDF en blocs de texte enrichis de métadonnées d'articles.
        Utilise un cache local pour des chargements ultra-rapides.
        """
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, "r", encoding="utf-8") as f:
                    cached_chunks = json.load(f)
                    if cached_chunks:
                        return cached_chunks
            except Exception:
                pass

        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"Le fichier PDF {self.pdf_path} est introuvable.")

        reader = PdfReader(self.pdf_path)
        raw_pages = []

        current_article = "Réglementation Générale FFTRI"
        
        for idx, page in enumerate(reader.pages):
            page_num = idx + 1
            text = page.extract_text() or ""
            if not text.strip():
                continue

            lines = text.split("\n")
            cleaned_lines = []
            for line in lines:
                line_str = line.strip()
                if not line_str:
                    continue
                # Détection de titre d'article / chapitre (ex: ARTICLE 5, CHAPITRE II, 3.1.2, REGLES GENERALES)
                article_match = re.search(
                    r'^(ARTICLE\s+\d+[\.\d]*|CHAPITRE\s+[I|V|X|\d]+|SECTION\s+\d+|[A-Z0-9\.\s\-]{4,50}\:\s*|[0-9]\.[0-9]\s+[A-Z\s]{4,})', 
                    line_str, 
                    re.IGNORECASE
                )
                if article_match:
                    current_article = line_str

                cleaned_lines.append(line_str)

            page_text = "\n".join(cleaned_lines)
            raw_pages.append({
                "page": page_num,
                "article_context": current_article,
                "text": page_text
            })

        # Découpage en chunks par mots avec chevauchement
        chunks = []
        chunk_id = 0

        for p in raw_pages:
            words = p["text"].split()
            if not words:
                continue

            i = 0
            while i < len(words):
                chunk_words = words[i:i + chunk_size_words]
                chunk_text = " ".join(chunk_words)
                
                # Essai de détection fine d'article spécifique dans ce chunk
                article_header_match = re.search(
                    r'(ARTICLE\s+\d+[\.\d]*|CHAPITRE\s+[I|V|X|\d]+|Art\.\s*\d+)',
                    chunk_text,
                    re.IGNORECASE
                )
                article_ref = article_header_match.group(0) if article_header_match else p["article_context"]

                chunks.append({
                    "id": chunk_id,
                    "page": p["page"],
                    "article_ref": article_ref,
                    "text": chunk_text
                })
                chunk_id += 1
                i += (chunk_size_words - overlap_words)

        # Sauvegarde dans le cache
        os.makedirs(CACHE_DIR, exist_ok=True)
        try:
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        return chunks
