from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from table_schema import Base, Image, Match, Patch


# Define your database URL
DATABASE_URL = "postgresql://yamhamelach:Aa123456@localhost:5432/matches_db"

# Create the database engine
engine = create_engine(DATABASE_URL)

# Create a session maker
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_package():
    Base.metadata.create_all(bind=engine)
    print("Tables created successfully!")


init_package()


# Database Query Functions
def add_image_by_name(image_name: str):
    """Adds a new image to the database."""
    session = SessionLocal()
    try:
        image_id = f"{image_name}.jpg"
        img = Image(image_id=image_id, filename=image_name)
        session.add(img)
        session.commit()
        print(f"Image {image_name} added successfully!")
    except Exception as e:
        session.rollback()
        print(f"Error adding image: {e}")
    finally:
        session.close()


def get_all_images():
    """Fetches all images from the database."""
    session = SessionLocal()
    try:
        images = session.query(Image).all()
        return images
    finally:
        session.close()


def add_patch(
    patch_id: str,
    image_id: str,
    filename: str,
    coord_top: int,
    coord_right: int,
    coord_bottom: int,
    coord_left: int,
):
    """Adds a new patch to the database."""
    session = SessionLocal()
    try:
        patch = Patch(
            patch_id=patch_id,
            image_id=image_id,
            filename=filename,
            coord_top=coord_top,
            coord_right=coord_right,
            coord_bottom=coord_bottom,
            coord_left=coord_left,
        )
        session.add(patch)
        session.commit()
        print(f"Patch {patch_id} added successfully!")
    except Exception as e:
        session.rollback()
        print(f"Error adding patch: {e}")
    finally:
        session.close()


def get_patches_by_image(image_id: str):
    """Fetches all patches associated with a specific image."""
    session = SessionLocal()
    try:
        patches = session.query(Patch).filter(Patch.image_id == image_id).all()
        return patches
    finally:
        session.close()


def add_match(
    patch_id_1: str,
    patch_id_2: str,
    matches=None,
    match_score=None,
    distance=None,
    sum_homo_err=None,
    len_homo_err=None,
    mean_homo_err=None,
    std_homo_err=None,
    is_valid=None,
):
    """Adds a match between two patches with optional parameters."""
    session = SessionLocal()
    try:
        # Initialize the Match object with required parameters
        match = Match(patch_id_1=patch_id_1, patch_id_2=patch_id_2)

        # Set optional parameters if provided
        if matches is not None:
            match.matches = matches
        if match_score is not None:
            match.match_score = match_score
        if distance is not None:
            match.distance = distance
        if sum_homo_err is not None:
            match.sum_homo_err = sum_homo_err
        if len_homo_err is not None:
            match.len_homo_err = len_homo_err
        if mean_homo_err is not None:
            match.mean_homo_err = mean_homo_err
        if std_homo_err is not None:
            match.std_homo_err = std_homo_err
        if is_valid is not None:
            match.is_valid = is_valid

        # Add the Match object to the session and commit
        session.add(match)
        session.commit()

    except Exception as e:
        session.rollback()
        print(f"Error adding match: {e}")

    finally:
        session.close()


def get_matches_between_images(image_id_1: str, image_id_2: str):
    """Fetches all matches between patches from two images."""
    session = SessionLocal()
    if image_id_2 < image_id_1:
        image_id_1, image_id_2 = image_id_2, image_id_1

    try:
        # Fetch all patch ids for the first image
        patch_ids_1 = (
            session.query(Patch.patch_id)
            .filter(Patch.image_id == image_id_1)
            .subquery()
        )

        # Fetch all patch ids for the second image
        patch_ids_2 = (
            session.query(Patch.patch_id)
            .filter(Patch.image_id == image_id_2)
            .subquery()
        )

        # Get matches where patch_id_1 belongs to image 1 and patch_id_2 belongs to image 2
        matches = (
            session.query(Match)
            .filter(
                Match.patch_id_1.in_(patch_ids_1), Match.patch_id_2.in_(patch_ids_2)
            )
            .all()
        )

        return matches

    except Exception as e:
        print(f"query failed: {e}")
        return None

    finally:
        session.close()
