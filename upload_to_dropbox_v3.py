import os
import time
import dropbox
from pathlib import Path
from dotenv import load_dotenv
from dropbox.exceptions import AuthError
from typing import Optional

load_dotenv()

# Dropbox API setup
DROPBOX_ACCESS_TOKEN = os.getenv("DROPBOX_ACCESS_TOKEN")
APP_KEY = os.getenv("DROPBOX_APP_KEY", None)
APP_SECRET = os.getenv("DROPBOX_APP_SECRET", None)


class DropboxUploader:
    def __init__(self, access_token, app_key=None, app_secret=None):
        """Initialize the Dropbox client with access token and refresh capabilities."""
        self.access_token = access_token
        self.app_key = app_key
        self.app_secret = app_secret
        self.dbx = None
        self.initialize_client()

    def initialize_client(self):
        """Initialize the Dropbox client with current access token"""
        self.dbx = dropbox.Dropbox(self.access_token)

    def check_and_refresh_token(self):
        """Check if connection is valid, get new tokens if expired"""
        try:
            # Test the connection
            self.dbx.users_get_current_account()
        except AuthError as e:
            if "expired_access_token" in str(e):
                if self.app_key and self.app_secret:
                    print("Access token expired. Getting new tokens...")
                    self._get_new_tokens()
                else:
                    raise Exception(
                        "Access token expired and app credentials not available. "
                        "Please provide app_key and app_secret."
                    )

    def _get_new_tokens(self):
        """Get new tokens through OAuth flow"""
        try:
            tokens = self.get_initial_tokens(self.app_key, self.app_secret)
            self.access_token = tokens["access_token"]
            self.initialize_client()
            print("Successfully obtained new access token")
        except Exception as e:
            raise Exception(f"Failed to get new tokens: {str(e)}")

    @staticmethod
    def get_initial_tokens(app_key, app_secret):
        """Get initial access and refresh tokens through OAuth flow"""
        auth_flow = dropbox.oauth.DropboxOAuth2FlowNoRedirect(
            app_key, app_secret, token_access_type="offline"
        )

        # 1. Get authorization URL
        auth_url = auth_flow.start()
        print("\n1. Go to this URL:", auth_url)
        print("2. Click 'Allow' (you might have to log in first)")
        print("3. Copy the authorization code")

        # 2. Get authorization code from user
        auth_code = input("\nEnter the authorization code here: ").strip()

        try:
            # 3. Exchange auth code for tokens
            oauth_result = auth_flow.finish(auth_code)
            return {
                "access_token": oauth_result.access_token,
                "refresh_token": oauth_result.refresh_token,
            }
        except Exception as e:
            raise Exception(f"Failed to get tokens: {str(e)}")

    def execute_with_retry(self, operation, max_retries=3):
        """Execute a Dropbox operation with automatic token refresh and retry logic"""
        for attempt in range(max_retries):
            try:
                self.check_and_refresh_token()
                return operation()
            except AuthError as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(1 * (attempt + 1))
                continue
            except Exception as e:
                raise

    def upload_file(self, local_path, dropbox_path):
        """Upload a single file to Dropbox, with path validation and retry logic."""

        def _upload():
            if not self.is_valid_dropbox_path(
                dropbox_path, check_existence=False
            ):
                print(
                    f"Invalid Dropbox path or insufficient permissions: {dropbox_path}"
                )
                return False

            if self.file_exists_in_dropbox(dropbox_path):
                print(f"File already exists, skipping: {dropbox_path}")
                return True

            with open(local_path, "rb") as f:
                file_size = os.path.getsize(local_path)
                CHUNK_SIZE = 4 * 1024 * 1024

                if file_size <= CHUNK_SIZE:
                    self.dbx.files_upload(f.read(), dropbox_path)
                else:
                    upload_session_start_result = (
                        self.dbx.files_upload_session_start(f.read(CHUNK_SIZE))
                    )
                    cursor = dropbox.files.UploadSessionCursor(
                        session_id=upload_session_start_result.session_id,
                        offset=f.tell(),
                    )
                    commit = dropbox.files.CommitInfo(path=dropbox_path)

                    while f.tell() < file_size:
                        if (file_size - f.tell()) <= CHUNK_SIZE:
                            self.dbx.files_upload_session_finish(
                                f.read(CHUNK_SIZE), cursor, commit
                            )
                        else:
                            self.dbx.files_upload_session_append_v2(
                                f.read(CHUNK_SIZE), cursor
                            )
                            cursor.offset = f.tell()

            print(f"Uploaded {local_path} to {dropbox_path}")
            return True

        try:
            return self.execute_with_retry(_upload)
        except Exception as e:
            print(f"Failed to upload {local_path}: {str(e)}")
            return False

    def upload_directory(self, local_dir, dropbox_base_path):
        """Upload an entire directory to Dropbox with retry logic."""

        def _upload_dir():
            local_dir_path = Path(local_dir)
            successful_uploads = 0
            failed_uploads = 0

            for local_path in local_dir_path.rglob("*"):
                if local_path.is_file():
                    if (
                        local_path.name.startswith(".")
                        or local_path.name == ".DS_Store"
                    ):
                        continue

                    relative_path = local_path.relative_to(local_dir_path)
                    dropbox_path = (
                        f"{dropbox_base_path}/{relative_path}".replace(
                            "\\", "/"
                        )
                    )

                    if self.upload_file(str(local_path), dropbox_path):
                        successful_uploads += 1
                    else:
                        failed_uploads += 1

            return successful_uploads, failed_uploads

        return self.execute_with_retry(_upload_dir)

    def ensure_folder_exists(self, path):
        """Ensure folder exists with retry logic."""

        def _ensure_folder():
            try:
                self.dbx.files_get_metadata(path)
            except dropbox.exceptions.ApiError:
                try:
                    self.dbx.files_create_folder_v2(path)
                    print(f"Created folder {path}")
                except dropbox.exceptions.ApiError as e:
                    print(f"Failed to create folder {path}: {str(e)}")

        return self.execute_with_retry(_ensure_folder)

    def copy_file_within_dropbox(self, from_path, to_path):
        """Copy a file within Dropbox with retry logic."""

        def _copy_file():
            if not self.is_valid_dropbox_path(from_path):
                print(
                    f"Invalid Dropbox source path or insufficient permissions: {from_path}"
                )
                return

            if not self.is_valid_dropbox_path(to_path, check_existence=False):
                print(
                    f"Invalid Dropbox destination path or insufficient permissions: {os.path.dirname(to_path)}"
                )
                return

            if self.file_exists_in_dropbox(to_path):
                print(
                    f"File already exists at destination, skipping: {to_path}"
                )
                return

            self.dbx.files_copy_v2(from_path, to_path)
            print(f"Copied {from_path} to {to_path}")

        return self.execute_with_retry(_copy_file)

    def copy_directory_within_dropbox(
        self, from_folder, to_folder, only_sub_folder=None
    ):
        """Copy directory within Dropbox with retry logic."""

        def _copy_directory():
            result = self.dbx.files_list_folder(from_folder, recursive=True)

            def process_entries(entries):
                for entry in entries:
                    relative_path = entry.path_display[
                        len(from_folder) :
                    ].lstrip("/")
                    if (only_sub_folder is not None) and (
                        only_sub_folder not in relative_path
                    ):
                        continue

                    if len(relative_path) == 0:
                        continue
                    dest_path = f"{to_folder}/{relative_path}"

                    if isinstance(entry, dropbox.files.FolderMetadata):
                        self.ensure_folder_exists(dest_path)
                    elif isinstance(entry, dropbox.files.FileMetadata):
                        if not self.file_exists_in_dropbox(dest_path):
                            self.copy_file_within_dropbox(
                                entry.path_display, dest_path
                            )
                        else:
                            print(
                                f"File already exists, skipping: {dest_path}"
                            )

            process_entries(result.entries)

            while result.has_more:
                result = self.dbx.files_list_folder_continue(result.cursor)
                process_entries(result.entries)

            print(
                f"Successfully copied directory from {from_folder} to {to_folder}"
            )

        return self.execute_with_retry(_copy_directory)

    def is_valid_dropbox_path(self, dropbox_path, check_existence=True):
        """Check path validity with retry logic."""

        def _check_path():
            try:
                if check_existence:
                    self.dbx.files_get_metadata(dropbox_path)
                else:
                    parent_path = os.path.dirname(dropbox_path)
                    if parent_path:
                        self.dbx.files_get_metadata(parent_path)
                return True
            except dropbox.exceptions.ApiError as e:
                print(
                    f"Error validating path: {dropbox_path}, error: {str(e)}"
                )
                return False

        return self.execute_with_retry(_check_path)

    def file_exists_in_dropbox(self, dropbox_path):
        """Check file existence with retry logic."""

        def _check_existence():
            try:
                self.dbx.files_get_metadata(dropbox_path)
                return True
            except dropbox.exceptions.ApiError:
                return False

        return self.execute_with_retry(_check_existence)


def main():
    # Initialize uploader with access token and app credentials
    uploader = DropboxUploader(
        access_token=DROPBOX_ACCESS_TOKEN,
        app_key=APP_KEY,
        app_secret=APP_SECRET,
    )

    # Example usage: Copy a directory within Dropbox
    from_folder = "/YamHamelach_data_n_model/"
    to_folder = "/Apps/YamHamelach"
    only_sub_folder = "patches_key_dec_cache"
    uploader.copy_directory_within_dropbox(
        from_folder, to_folder, only_sub_folder
    )


if __name__ == "__main__":
    main()
