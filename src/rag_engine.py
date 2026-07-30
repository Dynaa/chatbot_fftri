import os
from typing import List, Dict, Any, Optional
from PIL import Image
from google import genai
from google.genai import types

from src.prompt_templates import (
    SYSTEM_PROMPT, 
    USER_QUERY_TEMPLATE, 
    IMAGE_ANALYSIS_TEMPLATE
)

class RAGEngine:
    """
    Moteur RAG multimodal combinant la recherche documentaire FFTRI et l'IA Gemini.
    Sélectionne dynamiquement le modèle fonctionnel adapté à la clé API fournie.
    """
    def __init__(self, vector_store, api_key: Optional[str] = None):
        self.vector_store = vector_store
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self._cached_model = None
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = None

    def _get_candidate_models(self) -> List[str]:
        """
        Détermine dynamiquement la liste des modèles Gemini supportés par la clé API.
        """
        candidates = []
        if self.client:
            try:
                models = list(self.client.models.list())
                for m in models:
                    m_name = getattr(m, "name", "")
                    methods = getattr(m, "supported_generation_methods", []) or []
                    if "generateContent" in methods:
                        candidates.append(m_name)
            except Exception:
                pass

        # Noms de modèles standards par défaut si list_models ne répond pas ou est restreint
        default_fallbacks = [
            "gemini-2.0-flash",
            "gemini-1.5-flash",
            "gemini-1.5-flash-latest",
            "gemini-1.5-pro",
            "gemini-2.5-flash"
        ]

        for fb in default_fallbacks:
            if fb not in candidates:
                candidates.append(fb)

        return candidates

    def query_text(self, question: str, top_k: int = 4) -> Dict[str, Any]:
        """
        Traite une question textuelle d'arbitrage.
        """
        retrieved_chunks = self.vector_store.search(question, top_k=top_k)
        
        if not retrieved_chunks:
            return {
                "answer": "Aucun article correspondant n'a été trouvé dans la réglementation sportive fournie pour cette recherche.",
                "sources": []
            }

        context_str = self._format_context(retrieved_chunks)
        prompt_user = USER_QUERY_TEMPLATE.format(
            retrieved_context=context_str,
            user_query=question
        )

        if not self.client:
            return {
                "answer": (
                    "⚠️ **Clé API Gemini non configurée.**\n\n"
                    "Voici les articles du règlement identifiés pour votre question :\n\n"
                    + context_str +
                    "\n\n*Veuillez configurer la variable GEMINI_API_KEY dans votre fichier `.env` ou sur Railway pour générer la synthèse IA.*"
                ),
                "sources": retrieved_chunks
            }

        candidates = self._get_candidate_models()
        last_error = None

        for model_name in candidates:
            try:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=prompt_user,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.1
                    )
                )
                return {
                    "answer": response.text,
                    "sources": retrieved_chunks
                }
            except Exception as e:
                last_error = e

        return {
            "answer": f"Erreur lors de la génération IA : {str(last_error)}",
            "sources": retrieved_chunks
        }

    def query_multimodal(self, image: Image.Image, question: str = "Analyser la conformité de cette situation de course", top_k: int = 4) -> Dict[str, Any]:
        """
        Traite une situation de course visuelle (Photo + Texte optionnel).
        """
        search_query = question if len(question) > 5 else "dossard velo casque transition tenue materiel sanction"
        retrieved_chunks = self.vector_store.search(search_query, top_k=top_k)
        
        context_str = self._format_context(retrieved_chunks)
        prompt_user = IMAGE_ANALYSIS_TEMPLATE.format(
            retrieved_context=context_str,
            user_query=question
        )

        if not self.client:
            return {
                "answer": (
                    "⚠️ **Clé API Gemini non configurée.**\n\n"
                    "Articles du règlement extraits :\n\n"
                    + context_str +
                    "\n\n*Veuillez configurer GEMINI_API_KEY pour l'analyse visuelle par IA.*"
                ),
                "sources": retrieved_chunks
            }

        candidates = self._get_candidate_models()
        last_error = None

        for model_name in candidates:
            try:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=[prompt_user, image],
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.1
                    )
                )
                return {
                    "answer": response.text,
                    "sources": retrieved_chunks
                }
            except Exception as e:
                last_error = e

        return {
            "answer": f"Erreur lors de l'analyse d'image par l'IA : {str(last_error)}",
            "sources": retrieved_chunks
        }

    def _format_context(self, chunks: List[Dict[str, Any]]) -> str:
        formatted = []
        for c in chunks:
            formatted.append(
                f"--- [Réf: {c.get('article_ref', 'N/A')} | Page PDF: {c.get('page', '?')}] ---\n"
                f"{c.get('text', '')}\n"
            )
        return "\n".join(formatted)
