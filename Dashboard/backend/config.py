from dotenv import load_dotenv
load_dotenv()
import os

HOST = os.getenv("HOST")
USER = os.getenv("USER")
PASS = os.getenv("PASS")
DB = os.getenv("DB")
TABLE1 = os.getenv("TABLE1")
TABLE2 = os.getenv("TABLE2")
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")