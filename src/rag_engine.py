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
    """
    def __init__(self, vector_store, api_key: Optional[str] = None):
        self.vector_store = vector_store
        raw_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.api_key = raw_key.strip().strip("'").strip('"').strip() if raw_key else ""
            
        if self.api_key and HAS_GENAI:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = None

        if self.api_key and HAS_GENAI_LEGACY:
            try:
                genai_legacy.configure(api_key=self.api_key)
            except Exception:
                pass

    def _get_model_candidates(self) -> List[str]:
        candidates = [
            "gemini-2.0-flash",
            "gemini-1.5-flash",
            "models/gemini-2.0-flash",
            "models/gemini-1.5-flash",
            "gemini-1.5-flash-latest",
            "gemini-1.5-pro",
            "gemini-2.0-flash-exp"
        ]
        
        # Découverte dynamique via l'API list() si possible
        if self.client:
            try:
                listed = list(self.client.models.list())
                for m in listed:
                    name = getattr(m, "name", "")
                    if name:
                        if name not in candidates:
                            candidates.insert(0, name)
                        short = name.replace("models/", "")
                        if short not in candidates:
                            candidates.insert(0, short)
            except Exception:
                pass

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

        if not self.api_key:
            return {
                "answer": (
                    "⚠️ **Clé API Gemini non configurée.**\n\n"
                    "Voici les articles du règlement identifiés pour votre question :\n\n"
                    + context_str +
                    "\n\n*Veuillez saisir votre clé API Gemini dans la barre latérale.*"
                ),
                "sources": retrieved_chunks
            }

        candidates = self._get_model_candidates()
        last_error = None

        # 1. Essai avec SDK google-genai
        if self.client:
            for m_name in candidates:
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
            for m_name in candidates:
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

        err_msg = str(last_error)
        if "404" in err_msg or "NOT_FOUND" in err_msg:
            help_text = (
                "⚠️ **Erreur de propagation Google API (404 NOT_FOUND)**\n\n"
                "Votre clé API AI Studio (`chatbotfftri`) est bien enregistrée ! Cependant, Google met généralement **1 à 3 minutes** à propager les autorisations des nouvelles clés API sur l'ensemble de ses serveurs.\n\n"
                "**Que faire ?**\n"
                "1. Patientez 1 minute puis réessayez de cliquer sur **'Interroger le règlement'**.\n"
                "2. Vérifiez que la clé copiée dans la barre latérale ne contient aucun espace supplémentaire.\n"
                "3. Si l'erreur persiste, vous pouvez créer une seconde clé sur AI Studio et la coller dans la barre latérale."
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

        candidates = self._get_model_candidates()
        last_error = None

        if self.client:
            for m_name in candidates:
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
            for m_name in candidates:
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
                "⚠️ **Erreur de propagation Google API (404 NOT_FOUND)**\n\n"
                "La clé API est récente et en cours d'activation sur les serveurs Google (délai de 1 à 3 min). Réessayez dans un instant."
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
