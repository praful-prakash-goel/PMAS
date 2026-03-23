from backend.database import SessionLocal
from backend import models
from backend.core.hashing import Hash

def seed_admin():
    db = SessionLocal()
    try:
        admin_email = "almighty@gmail.com"
        exists = db.query(models.User).filter(models.User.email == admin_email).first()

        if not exists:
            print(f"Creating admin: {admin_email}...")
            new_admin = models.User(
                name="Almighty",
                email=admin_email,
                password=Hash.bcrypt("Almighty@123"), # Hashing is the key here!
                org_name="Nexus",
                role="ADMIN"
            )
            db.add(new_admin)
            db.commit()
            print("Admin created successfully!")
        else:
            print("Admin already exists in the database. No changes made.")
            
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    seed_admin()