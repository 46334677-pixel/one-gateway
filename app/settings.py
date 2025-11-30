from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()


class Settings(BaseModel):
    GATEWAY_API_KEY: str = os.getenv("GATEWAY_API_KEY", "")
    FEATURE_TRIP: int = int(os.getenv("FEATURE_TRIP", "1"))
    FEATURE_COMPARE: int = int(os.getenv("FEATURE_COMPARE", "0"))
    FEATURE_BIZ: int = int(os.getenv("FEATURE_BIZ", "0"))
    AMAP_KEY: str | None = os.getenv("AMAP_KEY")
    WEATHER_KEY: str | None = os.getenv("WEATHER_KEY")
    JD_APPKEY: str | None = os.getenv("JD_APPKEY")
    PDD_CLIENT_ID: str | None = os.getenv("PDD_CLIENT_ID")
    PDD_CLIENT_SECRET: str | None = os.getenv("PDD_CLIENT_SECRET")


settings = Settings()

