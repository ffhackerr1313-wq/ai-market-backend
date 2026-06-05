import os
from dotenv import load_dotenv

load_dotenv()

NEWS_API_KEY        = os.getenv("NEWS_API_KEY", "")
TELEGRAM_BOT_TOKEN  = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID    = os.getenv("TELEGRAM_CHAT_ID", "")
ALERT_THRESHOLD     = int(os.getenv("ALERT_THRESHOLD", "60"))
FRONTEND_URL        = os.getenv("FRONTEND_URL", "")
