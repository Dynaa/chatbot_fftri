# 🏊‍♂️🚴‍♀️🏃‍♂️ Prototype Chatbot d'Aide à l'Arbitrage FFTRI

Ce prototype est un assistant logiciel basé sur l'IA (RAG multimodal avec Google Gemini 2.5 Flash) conçu pour aider les arbitres de la **Fédération Française de Triathlon (FFTRI)** à vérifier les cas pratiques de course et à analyser des situations en photo.

---

## 🎯 Fonctionnalités Clés

1. **Strict Grounding (Zéro Hallucination)** : S'appuie exclusivement sur le document officiel `20251215 - Réglementation sportive.pdf`. Si une situation n'est pas traitée dans le règlement, l'assistant l'indique clairement.
2. **Citations Systématiques des Articles** : Chaque réponse inclut une synthèse terrain ET pointe directement le numéro d'article et la page PDF de référence.
3. **Analyse Multimodale (Photos de situations)** : Téléversement d'une photo prise en course pour analyser sa conformité (ex: tenue, position du dossard, matériel, comportement dans le parc à vélos).
4. **Interface Intuitive Streamlit** : Interface responsive adaptée aux démonstrations devant le groupe de travail informatique d'arbitrage FFTRI.

---

## 🚀 Lancement Local

### 1. Prérequis
- Python 3.10+
- Une clé API Google Gemini (gratuite sur [Google AI Studio](https://aistudio.google.com/))

### 2. Installation
```bash
# Activation de l'environnement virtuel
python -m venv .venv
# Sur Windows :
.venv\Scripts\activate
# Sur Linux/macOS :
source .venv/bin/activate

# Installation des dépendances
pip install -r requirements.txt
```

### 3. Configuration de la Clé API
Créez un fichier `.env` à la racine :
```env
GEMINI_API_KEY=votre_cle_api_gemini_ici
```

### 4. Démarrage de l'Application
```bash
streamlit run app.py
```
L'interface sera accessible sur `http://localhost:8501`.

---

## ☁️ Déploiement en Ligne sur Railway

1. **Créer un nouveau projet sur Railway** :
   - Connectez votre compte Railway à GitHub.
   - Cliquez sur **New Project** > **Deploy from GitHub repo** et sélectionnez ce dépôt `chatbot_fftri`.

2. **Ajouter la variable d'environnement** :
   - Dans le dashboard Railway de votre projet, allez dans **Variables**.
   - Ajoutez `GEMINI_API_KEY` avec votre clé API Google Studio.

3. **Génération de l'URL publique** :
   - Dans l'onglet **Settings** > **Networking**, cliquez sur **Generate Domain** pour obtenir l'URL publique de votre démo (ex: `chatbot-fftri-production.up.railway.app`).

Railway détectera automatiquement le `Dockerfile` et déploiera votre prototype en moins de 2 minutes !

---

## 📁 Arborescence du Code

```
chatbot_fftri/
├── docs/
│   └── 20251215 - Réglementation sportive.pdf   # Règlement officiel de référence
├── src/
│   ├── pdf_processor.py                       # Extraction & structuration des articles du PDF
│   ├── vector_store.py                        # Moteur d'indexation vectorielle (RAG)
│   ├── rag_engine.py                          # Intégration Gemini Multimodal (Texte & Vision)
│   └── prompt_templates.py                    # Directives anti-hallucination & formatage de réponse
├── app.py                                     # Application Web Streamlit
├── Dockerfile                                 # Container pour déploiement Railway
├── requirements.txt                           # Dépendances Python
└── README.md                                  # Documentation
```
