import os
from typing import List, Dict, Any, Optional
from PIL import Image

# Import SDK principal (google-genai)
try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

# Import SDK legacy fallback (google-generativeai)
try:
    import google.generativeai as genai_legacy
    HAS_GENAI_LEGACY = True
except ImportError:
    HAS_GENAI_LEGACY = False

from src.prompt_templates import (
    SYSTEM_PROMPT, 
    USER_QUERY_TEMPLATE, 
    IMAGE_ANALYSIS_TEMPLATE
)

class RAGEngine:
    """
    Moteur RAG multimodal combinant la recherche documentaire FFTRI et l'IA Gemini.
    Prend en charge les SDK google-genai et google-generativeai avec messages d'aide clairs.
    """
    def __init__(self, vector_store, api_key: Optional[str] = None):
        self.vector_store = vector_store
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if self.api_key:
            self.api_key = self.api_key.strip()
            
        if self.api_key and HAS_GENAI:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = None

        if self.api_key and HAS_GENAI_LEGACY:
            genai_legacy.configure(api_key=self.api_key)

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

        if not self.api_key:
            return {
                "answer": (
                    "⚠️ **Clé API Gemini non configurée.**\n\n"
                    "Voici les articles du règlement identifiés pour votre question :\n\n"
                    + context_str +
                    "\n\n*Veuillez configurer la variable GEMINI_API_KEY dans votre fichier `.env` ou sur Railway pour générer la synthèse IA.*"
                ),
                "sources": retrieved_chunks
            }

        # 1. Essai avec SDK google-genai
        models_to_try = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.5-flash"]
        last_error = None

        if self.client:
            for m_name in models_to_try:
                try:
                    response = self.client.models.generate_content(
                        model=m_name,
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

        # 2. Essai de secours avec SDK google-generativeai
        if HAS_GENAI_LEGACY:
            for m_name in ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]:
                try:
                    g_model = genai_legacy.GenerativeModel(
                        model_name=m_name,
                        system_instruction=SYSTEM_PROMPT
                    )
                    res = g_model.generate_content(prompt_user)
                    return {
                        "answer": res.text,
                        "sources": retrieved_chunks
                    }
                except Exception as e:
                    last_error = e

        # Gestion explicite des erreurs d'authentification / 404
        err_msg = str(last_error)
        if "404" in err_msg or "NOT_FOUND" in err_msg:
            help_text = (
                "⚠️ **Erreur d'accès à l'API Gemini (404 NOT_FOUND)**\n\n"
                "La clé API saisie semble ne pas disposer des autorisations sur les modèles Gemini standard.\n\n"
                "**Comment obtenir une clé API fonctionnelle en 1 minute ?**\n"
                "1. Allez sur **[Google AI Studio](https://aistudio.google.com/)**\n"
                "2. Cliquez sur le bouton bleu **'Create API Key'**\n"
                "3. Copiez la clé et collez-la dans la barre latérale à gauche."
            )
            return {"answer": help_text, "sources": retrieved_chunks}

        return {
            "answer": f"Erreur lors de la génération IA : {err_msg}",
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

        if not self.api_key:
            return {
                "answer": (
                    "⚠️ **Clé API Gemini non configurée.**\n\n"
                    "Articles du règlement extraits :\n\n"
                    + context_str +
                    "\n\n*Veuillez configurer GEMINI_API_KEY pour l'analyse visuelle par IA.*"
                ),
                "sources": retrieved_chunks
            }

        models_to_try = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.5-flash"]
        last_error = None

        if self.client:
            for m_name in models_to_try:
                try:
                    response = self.client.models.generate_content(
                        model=m_name,
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

        if HAS_GENAI_LEGACY:
            for m_name in ["gemini-1.5-flash", "gemini-1.5-pro"]:
                try:
                    g_model = genai_legacy.GenerativeModel(
                        model_name=m_name,
                        system_instruction=SYSTEM_PROMPT
                    )
                    res = g_model.generate_content([prompt_user, image])
                    return {
                        "answer": res.text,
                        "sources": retrieved_chunks
                    }
                except Exception as e:
                    last_error = e

        err_msg = str(last_error)
        if "404" in err_msg or "NOT_FOUND" in err_msg:
            help_text = (
                "⚠️ **Erreur d'accès à l'API Gemini (404 NOT_FOUND)**\n\n"
                "La clé API n'a pas accès aux modèles d'analyse d'image. Obtenez une clé gratuite sur **[Google AI Studio](https://aistudio.google.com/)**."
            )
            return {"answer": help_text, "sources": retrieved_chunks}

        return {
            "answer": f"Erreur lors de l'analyse d'image par l'IA : {err_msg}",
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
