from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os
from urllib.parse import quote_plus

# Database URL - update with your PostgreSQL credentials
# URL-encode the password to handle special characters like @
password = quote_plus("devika@2005")
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    f"postgresql://postgres:{password}@localhost:5432/rural_healthcare"
)

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create Base class
Base = declarative_base()

# Database Models
class User(Base):
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    mobile_number = Column(String, unique=True, nullable=False, index=True)
    password = Column(String, nullable=False)
    village = Column(String)
    age = Column(Integer)
    preferred_language = Column(String, default="hi")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    chat_messages = relationship("ChatMessage", back_populates="user")
    medical_records = relationship("MedicalRecord", back_populates="user")
    medication_reminders = relationship("MedicationReminder", backref="user")

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    
    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    text = Column(Text, nullable=False)
    sender = Column(String, nullable=False)  # 'user' or 'bot'
    timestamp = Column(DateTime, default=datetime.utcnow)
    language = Column(String, default="hi")
    intent = Column(String)  # 'qna', 'symptom', 'simplify'
    
    # Relationships
    user = relationship("User", back_populates="chat_messages")

class MedicalRecord(Base):
    __tablename__ = "medical_records"
    
    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    condition = Column(String, nullable=False)
    symptoms = Column(Text, nullable=False)
    diagnosis = Column(Text)
    treatment = Column(Text)
    date = Column(DateTime, nullable=False)
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    report_file = Column(String, nullable=True)  # Path to uploaded report file
    
    # Relationships
    user = relationship("User", back_populates="medical_records")

class MedicationReminder(Base):
    __tablename__ = "medication_reminders"

    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    medication_name = Column(String, nullable=False)
    frequency = Column(String, nullable=False)  # daily, weekly, monthly
    time = Column(String, nullable=False)       # "09:00"
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Create all tables
def create_tables():
    Base.metadata.create_all(bind=engine) 