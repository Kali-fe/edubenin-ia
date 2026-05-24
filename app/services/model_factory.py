import os


def get_model_service():
    """
    MODEL_MODE=openai -> GPT-5.5 (développement)
    MODEL_MODE=colab -> GLM-4.7 fine-tuné sur Colab
    """
    mode = os.getenv("MODEL_MODE", "openai")

    if mode == "openai":
        from app.services import openai_service
        return openai_service
    if mode == "colab":
        from app.services import colab_service
        return colab_service

    from app.services import openai_service
    return openai_service
