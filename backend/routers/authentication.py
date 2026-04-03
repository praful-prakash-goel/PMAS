from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from backend.core import auth_token
from backend import database, schemas, models
from backend.core.hashing import Hash

router = APIRouter(
    tags=['authentication']
)
get_db = database.get_db

@router.post('/login')
def login(
    request: schemas.UserLogin,
    db: Session = Depends(get_db)
):
    user = db.query(models.User).filter(models.User.email == request.email).first()
    
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invalid Credentials")
    
    # print(f"DEBUG: Email: {request.email}")
    # print(f"DEBUG: Plain password length: {len(request.password)}")
    # print(f"DEBUG: Hashed password from DB starts with: {user.password[:10]}")

    if not Hash.verify(request.password, user.password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Credentials")

    # return {"access_token": "test", "token_type": "bearer", "user": {"id": 1, "email": "test@test.com", "role": "admin"}}

    access_token = auth_token.create_access_token(
        data={
            "sub":user.email,
            "user_id":user.user_id,
            "role":user.role.value
        }
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user.user_id,
            "email": user.email,
            "name": user.name,
            "role": user.role.value,
            "org_name": user.org_name,
        }
    }