from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.security import OAuth2PasswordRequestForm, HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from config.configrations import account_collection
from models.Account import LoginRequest, Account
from bson import ObjectId

SECRET_KEY = "your_secret_key"  # Change this to a secure value!
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
router = APIRouter()
bearer_scheme = HTTPBearer()

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    token = credentials.credentials
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        account_id: str = payload.get("account_id")
        jabatan: str = payload.get("jabatan")
        if account_id is None or jabatan is None:
            raise credentials_exception
        return {"account_id": account_id, "jabatan": jabatan}
    except JWTError:
        raise credentials_exception

@router.post("/login")
async def login(login_req: LoginRequest):
    account = account_collection.find_one({"akun_upi": login_req.akun_upi})
    hashed_password = account.get("password") if account else None

    if not account or not hashed_password or not isinstance(hashed_password, str):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid akun_upi or password"
        )

    # Verifikasi password (bcrypt atau plaintext, passlib handle otomatis)
    try:
        if not verify_password(login_req.password, hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid akun_upi or password"
            )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid akun_upi or password"
        )

    access_token = create_access_token(
        data={"account_id": str(account["_id"]), "jabatan": account["jabatan"]}
    )
    # Prepare user_info to match Account model and ensure all ObjectId are string
    jadwal_mengajar = account.get("jadwal_mengajar", [])
    # Convert embedded class_id in jadwal_mengajar to string if present
    for jm in jadwal_mengajar:
        if "class_id" in jm and isinstance(jm["class_id"], ObjectId):
            jm["class_id"] = str(jm["class_id"])
    user_info = {
        "id": str(account["_id"]),
        "nama": account["nama"],
        "akun_upi": account["akun_upi"],
        "jabatan": account["jabatan"],
        "jadwal_mengajar": jadwal_mengajar
    }
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": user_info
    }

@router.get("/me")
async def read_me(current_user: dict = Depends(get_current_user)):
    return {"message": "You are authenticated!", "user": current_user}