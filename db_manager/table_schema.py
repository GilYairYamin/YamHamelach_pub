from sqlalchemy import (
    Column,
    String,
    Float,
    Integer,
    Boolean,
    ForeignKey,
    CheckConstraint,
    ARRAY,
)
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Image(Base):
    __tablename__ = "images"

    image_id = Column(String(100), primary_key=True)
    image_filename = Column(String(100))


class Patch(Base):
    __tablename__ = "patches"

    patch_id = Column(String(100), primary_key=True)
    image_id = Column(String(100), ForeignKey("images.image_id"))
    patch_filename = Column(String(100))
    coord_top = Column(Integer, nullable=False)
    coord_right = Column(Integer, nullable=False)
    coord_bottom = Column(Integer, nullable=False)
    coord_left = Column(Integer, nullable=False)

    # Relationship to the Images table
    image = relationship("Image", back_populates="patches")


Image.patches = relationship("Patch", back_populates="image")


class Match(Base):
    __tablename__ = "matches"

    patch_id_1 = Column(String(100), ForeignKey("patches.patch_id"), primary_key=True)
    patch_id_2 = Column(String(100), ForeignKey("patches.patch_id"), primary_key=True)
    matches = Column(ARRAY(Float))
    match_score = Column(Integer)
    distance = Column(Float)
    sum_homo_err = Column(Float)
    len_homo_err = Column(Float)
    mean_homo_err = Column(Float)
    std_homo_err = Column(Float)
    is_valid = Column(Boolean)

    # Relationships
    patch1 = relationship("Patch", foreign_keys=[patch_id_1])
    patch2 = relationship("Patch", foreign_keys=[patch_id_2])

    # Add the CheckConstraint
    __table_args__ = (
        CheckConstraint(
            "matches.patch_id_1 < matches.patch_id_2", name="check_patch_id_order"
        ),
    )
