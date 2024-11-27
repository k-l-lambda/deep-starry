
import os
from dotenv import load_dotenv



PROJECT_DIR = os.path.join(os.path.dirname(__file__), '../..')

load_dotenv()
load_dotenv(dotenv_path=os.path.join(PROJECT_DIR, '.env.local'), override=True)
