from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from passlib.hash import bcrypt
import mysql.connector
from config import HOST, USER, PASS, DB, TABLE2
router = APIRouter()

class UserRegister(BaseModel):
    username: str
    password: str

def get_db():
    return mysql.connector.connect(
        host=HOST,
        user=USER,
        password=PASS,
        database=DB
    )

@router.post("/register")
def register_user(user: UserRegister):
    db = get_db()
    cursor = db.cursor()

    cursor.execute(f"SELECT * FROM {TABLE2} WHERE username = %s", (user.username,))
    if cursor.fetchone():
        raise HTTPException(status_code=400, detail="Username already exists")

    hashed_pw = bcrypt.hash(user.password)
    cursor.execute(f"INSERT INTO {TABLE2} (username, password) VALUES (%s, %s)", (user.username, hashed_pw))
    db.commit()

    cursor.close()
    db.close()
    return {"message": "User registered successfully"}