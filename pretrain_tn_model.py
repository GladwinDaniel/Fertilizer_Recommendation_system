from __future__ import annotations

from tn_model_utils import MODEL_ARTIFACT_PATH, train_and_save_model


if __name__ == "__main__":
    artifacts = train_and_save_model(MODEL_ARTIFACT_PATH)
    print("Saved pre-trained model artifact:", MODEL_ARTIFACT_PATH)
    print("Validation accuracy:", round(artifacts["accuracy"] * 100, 2), "%")
