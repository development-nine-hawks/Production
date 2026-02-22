from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime
import config

engine = create_engine(config.DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Pattern(Base):
    __tablename__ = "patterns"
    id = Column(Integer, primary_key=True, index=True)
    seed = Column(Integer, nullable=False)
    serial_number = Column(String(50), default="")
    label = Column(String(100), default="")
    filename = Column(String(255), nullable=False)
    pattern_size = Column(Integer, default=512)
    notes = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)
    verifications = relationship("Verification", back_populates="pattern",
                                 cascade="all, delete-orphan")


class Verification(Base):
    __tablename__ = "verifications"
    id = Column(Integer, primary_key=True, index=True)
    pattern_id = Column(Integer, ForeignKey("patterns.id"), nullable=False)
    captured_filename = Column(String(255), default="")
    aligned_filename = Column(String(255), default="")
    verdict = Column(String(20), default="")
    confidence = Column(Float, default=0.0)
    score_moire = Column(Float, default=0.0)
    score_color = Column(Float, default=0.0)
    score_correlation = Column(Float, default=0.0)
    score_gradient = Column(Float, default=0.0)
    markers_found = Column(Integer, default=0)
    alignment_method = Column(String(20), default="")
    print_size_mm = Column(Integer, nullable=True)
    notes = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)
    pattern = relationship("Pattern", back_populates="verifications")


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
