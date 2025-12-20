from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import insert
from .table_schema import Base, Image, Match, Patch

import pandas as pd


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


def add_entire_df_to_db(matches_df: pd.DataFrame, chunk_size: int = 10000):
    """Add or update matches in the database using bulk inserts."""
    session = SessionLocal()
    try:
        for i in range(0, len(matches_df), chunk_size):
            chunk = matches_df.iloc[i : i + chunk_size]

            # Prepare a list of dictionaries for the bulk upsert
            match_data = []
            for _, row in chunk.iterrows():
                patch_id_1 = row["file1"].split(".")[0]
                patch_id_2 = row["file2"].split(".")[0]
                if patch_id_2 < patch_id_1:
                    patch_id_1, patch_id_2 = patch_id_2, patch_id_1

                match_data.append(
                    {
                        "patch_id_1": patch_id_1,
                        "patch_id_2": patch_id_2,
                        "distance": row["distance"],
                        "sum_homo_err": row["sum_homo_err"],
                        "len_homo_err": row["len_homo_err"],
                        "mean_homo_err": row["mean_homo_err"],
                        "std_homo_err": row["std_homo_err"],
                    }
                )

            # Use the insert construct with on_conflict_do_update
            stmt = insert(Match).values(match_data)
            stmt = stmt.on_conflict_do_update(
                index_elements=["patch_id_1", "patch_id_2"],  # Composite unique key
                set_={
                    "distance": stmt.excluded.distance,
                    "sum_homo_err": stmt.excluded.sum_homo_err,
                    "len_homo_err": stmt.excluded.len_homo_err,
                    "mean_homo_err": stmt.excluded.mean_homo_err,
                    "std_homo_err": stmt.excluded.std_homo_err,
                },
            )
            session.execute(stmt)
            session.commit()

            print(
                f"Inserted/Updated {len(match_data)} matches (Chunk {i // chunk_size + 1})"
            )

    except Exception as e:
        session.rollback()
        print(f"Error inserting/updating matches: {e}")
    finally:
        session.close()


# Database Query Functions
def add_image_by_id(image_id: str):
    """Adds a new image to the database."""
    session = SessionLocal()
    try:
        image_filename = f"{image_id}.jpg"
        stmt = insert(Image).values(image_id=image_id, image_filename=image_filename)
        stmt = stmt.on_conflict_do_update(
            index_elements=["image_id"],  # Conflict check on the primary key
            set_={"image_filename": stmt.excluded.image_filename},  # Update on conflict
        )
        session.execute(stmt)
        session.commit()

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
    patch_filename: str,
    coord_top: int,
    coord_right: int,
    coord_bottom: int,
    coord_left: int,
):
    """Adds a new patch to the database."""
    session = SessionLocal()
    try:
        stmt = insert(Patch).values(
            patch_id=patch_id,
            image_id=image_id,
            patch_filename=patch_filename,
            coord_left=coord_left,
            coord_top=coord_top,
            coord_right=coord_right,
            coord_bottom=coord_bottom,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["patch_id"],  # Conflict check on the primary key
            set_={
                "image_id": stmt.excluded.image_id,
                "patch_filename": stmt.excluded.patch_filename,
                "coord_left": stmt.excluded.coord_left,
                "coord_top": stmt.excluded.coord_top,
                "coord_right": stmt.excluded.coord_right,
                "coord_bottom": stmt.excluded.coord_bottom,
            },
        )
        session.execute(stmt)
        session.commit()

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
    distance: float,
    sum_homo_err: float,
    len_homo_err: float,
    mean_homo_err: float,
    std_homo_err: float,
):
    """Adds a match between two patches with optional parameters."""
    session = SessionLocal()
    try:
        match_data = {
            "patch_id_1": patch_id_1,
            "patch_id_2": patch_id_2,
            "distance": distance,
            "sum_homo_err": sum_homo_err,
            "len_homo_err": len_homo_err,
            "mean_homo_err": mean_homo_err,
            "std_homo_err": std_homo_err,
        }

        stmt = insert(Match).values(match_data)
        stmt = stmt.on_conflict_do_update(
            index_elements=[
                "patch_id_1",
                "patch_id_2",
            ],
            set_={
                "distance": stmt.excluded.distance,
                "sum_homo_err": stmt.excluded.sum_homo_err,
                "len_homo_err": stmt.excluded.len_homo_err,
                "mean_homo_err": stmt.excluded.mean_homo_err,
                "std_homo_err": stmt.excluded.std_homo_err,
            },
        )
        session.execute(stmt)
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
