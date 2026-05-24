# RAPPORT HANDOFF IA v2

## Confirmation écrasement
- Réinitialisation effectuée.
- Les éléments conservés côté projet sont `venv/`, `.env` et `.gitignore`.

## Fichiers créés
- `app/main.py`
- `app/models/schemas.py`
- `app/routes/chat.py`
- `app/routes/exercice.py`
- `app/routes/rag.py`
- `app/services/model_factory.py`
- `app/services/openai_service.py`
- `app/services/colab_service.py`
- `app/services/rag_service.py`
- `dataset/process_documents.py`
- `dataset/generate_dataset_v2.py`
- `dataset/validate_dataset.py`
- `dataset/input/.gitkeep`
- `dataset/processed/.gitkeep`
- `prompts/base_system_prompt.txt`
- `prompts/exercice_prompt.txt`
- `colab/EduBenin_GLM47_Finetune.ipynb`
- `.env.example`
- `requirements.txt`

## Architecture
- Mode développement: `openai` avec GPT-5.5 via API OpenAI.
- Mode production: `colab` avec GLM-4.7 fine-tuné servi via FastAPI sur Colab et exposé par ngrok.

## Variables `.env`
- `MODEL_MODE=openai`
- `OPENAI_API_KEY=ta_cle_ici`
- `OPENAI_MODEL=gpt-5.5`
- `COLAB_NGROK_URL=`

## Lancement
- `python -m uvicorn app.main:app --reload --port 8000`

## Test attendu
- `GET /` retourne `{"status":"EduBénin IA opérationnel","version":"2.0.0","mode":"openai"}` en mode développement.
- Documentation disponible sur `/docs`.
- Routes présentes: `POST /ia/chat`, `POST /ia/exercice`, `POST /ia/rag/integrer`.

## Colab
- Notebook prêt pour importation et fine-tuning GLM-4.7.
- Utiliser `COLAB_NGROK_URL` dans le backend pour pointer vers l’instance Colab publiée.
