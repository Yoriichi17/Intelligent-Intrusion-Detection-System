from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from passlib.hash import bcrypt
from jose import JWTError, jwt
import mysql.connector
import datetime
from config import HOST, USER, PASS, DB, TABLE2, SECRET_KEY, ALGORITHM
router = APIRouter()

class UserLogin(BaseModel):
    username: str
    password: str

def get_db():
    return mysql.connector.connect(
        host=HOST,
        user=USER,
        password=PASS,
        database=DB
    )

@router.post("/login")
def login(user: UserLogin):
    db = get_db()
    cursor = db.cursor(dictionary=True)

    cursor.execute(f"SELECT * FROM {TABLE2} WHERE username = %s", (user.username,))
    result = cursor.fetchone()

    cursor.close()
    db.close()

    if not result or not bcrypt.verify(user.password, result["password"]):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    expiration = datetime.datetime.utcnow() + datetime.timedelta(hours=2)
    token = jwt.encode({"sub": user.username, "exp": expiration}, SECRET_KEY, algorithm=ALGORITHM)


    return {"access_token": token, "token_type": "bearer"}
