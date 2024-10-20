from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Qdrant collections
    JINA_INDEX: str
    SIGLIP_INDEX: str
    SKETCH_INDEX: str

    VIDEO_DIR: str
    KEYFRAME_DIR: str
    SKETCH_IMG_DIR: str

    OPENAI_API_KEY: str

    TRANSCRIPTION_DIR: str
    class Config:
        env_file = ".env"


settings = Settings()
