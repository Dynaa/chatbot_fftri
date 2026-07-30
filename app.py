import os
import streamlit as st
from PIL import Image
from dotenv import load_dotenv

from src.pdf_processor import PDFProcessor
from src.vector_store import VectorStore
from src.rag_engine import RAGEngine

# Chargement des variables d'environnement (.env)
load_dotenv()

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Arbitrage FFTRI - Assistant IA",
    page_icon="🏊‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé FFTRI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        color: #0d3b66;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555555;
        margin-bottom: 1.5rem;
    }
    .stAlert {
        border-radius: 8px;
    }
    .source-box {
        background-color: #f8f9fa;
        border-left: 4px solid #0d3b66;
        padding: 10px 15px;
        margin-top: 10px;
        border-radius: 4px;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Chargement et indexation de la réglementation FFTRI...")
def load_rag_system(pdf_path: str):
    """Initialise le processeur PDF, l'index vectoriel et le moteur RAG."""
    processor = PDFProcessor(pdf_path)
    chunks = processor.process_and_chunk()
    vector_store = VectorStore(chunks)
    return vector_store, len(chunks)


def main():
    # Sidebar
    st.sidebar.image("https://www.fftri.com/wp-content/uploads/2019/04/logo-FFTRI.png", width=180)
    st.sidebar.title("Configuration & Statut")
    
    # Clé API
    env_api_key = os.environ.get("GEMINI_API_KEY", "")
    user_api_key = st.sidebar.text_input(
        "Clé API Google Gemini", 
        value=env_api_key, 
        type="password",
        help="Nécessaire pour la génération IA. Peut aussi être définie dans la variable d'environnement GEMINI_API_KEY."
    )
    
    if user_api_key:
        os.environ["GEMINI_API_KEY"] = user_api_key
        st.sidebar.success("✅ Clé API configurée")
    else:
        st.sidebar.warning("⚠️ Clé API absente. (Génération IA désactivée)")

    pdf_path = os.path.join("docs", "20251215 - Réglementation sportive.pdf")
    
    if not os.path.exists(pdf_path):
        st.error(f"❌ Le fichier de réglementation est introuvable à l'emplacement : `{pdf_path}`")
        return

    vector_store, total_chunks = load_rag_system(pdf_path)
    rag_engine = RAGEngine(vector_store, api_key=user_api_key or env_api_key)

    st.sidebar.info(f"📚 **Document source** :\n`20251215 - Réglementation sportive.pdf`\n\n🧩 **Chunks indexés** : {total_chunks}")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("💡 Exemples de cas pratiques")
    
    sample_queries = [
        "Un athlète retire son casque avant de poser son vélo dans le parc, quelle est la règle et la sanction ?",
        "Où doit être positionné le dossard pendant l'épreuve cycliste et en course à pied ?",
        "Quelle est la distance de drafting autorisée à vélo et la sanction en cas de non-respect ?",
        "Un concurrent peut-il courir nu-pieds ou torse nu ?"
    ]
    
    selected_sample = None
    for q in sample_queries:
        if st.sidebar.button(q, key=f"btn_{hash(q)}"):
            selected_sample = q

    # En-tête principal
    st.markdown('<div class="main-header">🏊‍♂️🚴‍♀️🏃‍♂️ Assistant IA d\'Arbitrage FFTRI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Prototype d\'aide à la décision pour les arbitres de triathlon — Conforme au règlement sportif officiel</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["💬 Question sur une situation", "📸 Analyse de Photo de situation"])

    # ONGLET 1 : Questions Textuelles
    with tab1:
        st.subheader("Poser une question réglementaire")
        
        default_val = selected_sample if selected_sample else ""
        query_input = st.text_area(
            "Décrivez la situation de course ou la question d'arbitrage :",
            value=default_val,
            height=100,
            placeholder="Ex: Quelle est la sanction pour un athlète qui monte sur son vélo avant la ligne de monte ?"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            submit_text = st.button("🔍 Interroger le règlement", type="primary", use_container_width=True)

        if submit_text and query_input.strip():
            with st.spinner("Recherche dans les articles de la réglementation FFTRI et génération de la synthèse..."):
                result = rag_engine.query_text(query_input)
                
                st.markdown("### 📋 Synthèse d'Arbitrage & Réponse")
                st.markdown(result["answer"])
                
                if result.get("sources"):
                    with st.expander("📖 Voir les extraits officiels et articles cités"):
                        for idx, src in enumerate(result["sources"], 1):
                            st.markdown(f"**Extrait #{idx} — Réf: `{src['article_ref']}` (Page PDF: {src['page']})**")
                            st.text(src["text"][:600] + ("..." if len(src["text"]) > 600 else ""))
                            st.markdown("---")

    # ONGLET 2 : Analyse de Photo
    with tab2:
        st.subheader("Soumettre une photo de situation de course")
        st.caption("Téléversez une photo prise sur le terrain (ex: tenue, vélo, dossard, parc à vélos) pour vérifier sa conformité.")
        
        uploaded_file = st.file_uploader("Choisissez une image (JPG, PNG, WEBP)...", type=["jpg", "jpeg", "png", "webp"])
        
        image_note = st.text_input(
            "Précisions ou question sur la photo (optionnel) :", 
            placeholder="Ex: Le dossard est-il bien positionné pour la partie course à pied ?"
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            col_img, col_res = st.columns([1, 1])
            with col_img:
                st.image(image, caption="Photo soumise", use_container_width=True)

            with col_res:
                if st.button("🔍 Analyser la conformité de la situation", type="primary", use_container_width=True):
                    with st.spinner("Analyse visuelle et confrontation à la réglementation FFTRI..."):
                        q_text = image_note if image_note.strip() else "Analyser la conformité de cette situation visuelle au règlement sportif FFTRI."
                        result = rag_engine.query_multimodal(image, question=q_text)
                        
                        st.markdown("### 🔍 Diagnostic de Conformité")
                        st.markdown(result["answer"])
                        
                        if result.get("sources"):
                            with st.expander("📖 Articles de référence consultés"):
                                for idx, src in enumerate(result["sources"], 1):
                                    st.markdown(f"**Article `{src['article_ref']}` (Page {src['page']})**")
                                    st.caption(src["text"][:300] + "...")

if __name__ == "__main__":
    main()
