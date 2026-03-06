from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "CAMERA AI - Transfer Detection"
    database_url: str = "postgresql+psycopg2://camera_ai:camera_ai@localhost:5432/camera_ai"
    image_dir: str = "data/images"
    ocr_use_angle_cls: bool = True
    ocr_confidence_threshold: float = 0.3
    log_level: str = "INFO"

    # AI extension settings
    ner_model_path: str = "data/models/ner_model"
    dataset_train_path: str = "data/dataset/train.json"
    layout_model_name: str = "microsoft/layoutlmv3-base"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
