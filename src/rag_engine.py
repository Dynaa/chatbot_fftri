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
    """
    def __init__(self, vector_store, api_key: Optional[str] = None):
        self.vector_store = vector_store
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = None

    def query_text(self, question: str, top_k: int = 4) -> Dict[str, Any]:
        """
        Traite une question textuelle d'arbitrage.
        """
        # 1. Retrieval des articles pertinents
        retrieved_chunks = self.vector_store.search(question, top_k=top_k)
        
        if not retrieved_chunks:
            return {
                "answer": "Aucun article correspondant n'a été trouvé dans la réglementation sportive fournie pour cette recherche.",
                "sources": []
            }

        # Formater le contexte
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

        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
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
            # Fallback vers gemini-1.5-flash si gemini-2.5-flash rencontre un souci de quota
            try:
                response = self.client.models.generate_content(
                    model="gemini-1.5-flash",
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
            except Exception as ex:
                return {
                    "answer": f"Erreur lors de la génération IA : {str(ex)}",
                    "sources": retrieved_chunks
                }

    def query_multimodal(self, image: Image.Image, question: str = "Analyser la conformité de cette situation de course", top_k: int = 4) -> Dict[str, Any]:
        """
        Traite une situation de course visuelle (Photo + Texte optionnel).
        """
        # Recherche par mots clés par défaut ou tirés du champ question
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

        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
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
            try:
                response = self.client.models.generate_content(
                    model="gemini-1.5-flash",
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
            except Exception as ex:
                return {
                    "answer": f"Erreur lors de l'analyse d'image par l'IA : {str(ex)}",
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
