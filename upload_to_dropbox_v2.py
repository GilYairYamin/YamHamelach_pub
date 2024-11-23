import os
import dropbox
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Dropbox API setup
DROPBOX_ACCESS_TOKEN = os.getenv('DROPBOX_ACCESS_TOKEN')


class DropboxUploader:
    def __init__(self, access_token):
        """Initialize the Dropbox client with access token."""
        self.dbx = dropbox.Dropbox(access_token)

    def upload_file(self, local_path, dropbox_path):
        """Upload a single file to Dropbox, with path validation."""
        if not self.is_valid_dropbox_path(dropbox_path):
            print(f"Invalid Dropbox path or insufficient permissions: {dropbox_path}")
            return False

        try:
            with open(local_path, 'rb') as f:
                # Use chunks for larger files
                file_size = os.path.getsize(local_path)
                CHUNK_SIZE = 4 * 1024 * 1024  # 4MB chunks

                if file_size <= CHUNK_SIZE:
                    # Small file, upload directly
                    self.dbx.files_upload(f.read(), dropbox_path)
                else:
                    # Large file, upload in chunks
                    upload_session_start_result = self.dbx.files_upload_session_start(f.read(CHUNK_SIZE))
                    cursor = dropbox.files.UploadSessionCursor(
                        session_id=upload_session_start_result.session_id,
                        offset=f.tell()
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
        except Exception as e:
            print(f"Failed to upload {local_path}: {str(e)}")
            return False

    def upload_directory(self, local_dir, dropbox_base_path):
        """Upload an entire directory to Dropbox."""
        local_dir = Path(local_dir)
        successful_uploads = 0
        failed_uploads = 0

        # Walk through the directory
        for local_path in local_dir.rglob('*'):
            if local_path.is_file():
                # Skip system files like .DS_Store
                if local_path.name.startswith('.') or local_path.name == ".DS_Store":
                    continue

                # Calculate relative path for Dropbox
                relative_path = local_path.relative_to(local_dir)
                dropbox_path = f"{dropbox_base_path}/{relative_path}".replace('\\', '/')

                # Upload the file
                if self.upload_file(str(local_path), dropbox_path):
                    successful_uploads += 1
                else:
                    failed_uploads += 1

        return successful_uploads, failed_uploads

    def ensure_folder_exists(self, path):
        """Ensure that a folder exists at the given path in Dropbox."""
        try:
            self.dbx.files_get_metadata(path)
        except dropbox.exceptions.ApiError:
            try:
                self.dbx.files_create_folder_v2(path)
                print(f"Created folder {path}")
            except dropbox.exceptions.ApiError as e:
                print(f"Failed to create folder {path}: {str(e)}")

    def copy_file_within_dropbox(self, from_path, to_path):
        """Copy a file within Dropbox from one location to another, with path validation."""
        if not self.is_valid_dropbox_path(from_path):
            print(f"Invalid Dropbox source path or insufficient permissions: {from_path}")
            return

        if not self.is_valid_dropbox_path(to_path , check_existence=False):
            print(f"Invalid Dropbox destination path or insufficient permissions: {os.path.dirname(to_path)}")
            return

        try:
            self.dbx.files_copy_v2(from_path, to_path)
            print(f"Copied {from_path} to {to_path}")
        except dropbox.exceptions.ApiError as e:
            print(f"Failed to copy {from_path} to {to_path}: {str(e)}")


    def list_all_files_within_dropbox(self, folder_path):
        """List all files within a Dropbox folder and process them recursively."""
        try:
            result = self.dbx.files_list_folder(folder_path, recursive=True)
            file_list = []

            def process_entries(entries):
                for entry in entries:
                    if isinstance(entry, dropbox.files.FileMetadata):
                        print(f"File: {entry.path_display}")
                        file_list.append(entry.path_display)

            # Process the initial batch of entries
            process_entries(result.entries)

            # Continue processing while there are more entries
            while result.has_more:
                result = self.dbx.files_list_folder_continue(result.cursor)
                process_entries(result.entries)

            print(f"Total files found: {len(file_list)}")
            for file_path in file_list:
                print(file_path)

            return file_list

        except dropbox.exceptions.ApiError as e:
            print(f"Failed to list files in folder {folder_path}: {str(e)}")
            return []
    def copy_directory_within_dropbox(self, from_folder, to_folder):
        """Recursively copy an entire directory structure from one location to another within Dropbox."""
        try:
            # List all entries in the source directory, recursively
            #result = self.dbx.files_list_folder(from_folder, recursive=True)
            result = self.list_all_files_within_dropbox(from_folder)
            for entry in result.entries:
                # Calculate the relative path and the destination path
                relative_path = entry.path_display[len(from_folder):].lstrip("/")
                if (len(relative_path) == 0):
                    continue
                dest_path = f"{to_folder}/{relative_path}"

                if isinstance(entry, dropbox.files.FolderMetadata):
                    # Ensure each folder exists in the destination
                    self.ensure_folder_exists(dest_path)
                elif isinstance(entry, dropbox.files.FileMetadata):
                    # Copy files to the corresponding path in the destination
                    self.copy_file_within_dropbox(entry.path_display, dest_path)

            print(f"Successfully copied directory from {from_folder} to {to_folder}")
        except dropbox.exceptions.ApiError as e:
            print(f"Failed to list or copy folder {from_folder}: {str(e)}")

    def is_valid_dropbox_path(self, dropbox_path, check_existence=True):
        """Check if a Dropbox path is valid and the app has permission to access it."""
        try:
            if check_existence:
                # Check if the path exists by attempting to list it
                self.dbx.files_get_metadata(dropbox_path)
            else:
                # Check parent folder existence without necessarily existing yet
                parent_path = os.path.dirname(dropbox_path)
                if parent_path:
                    self.dbx.files_get_metadata(parent_path)
            return True
        except dropbox.exceptions.ApiError as e:
            # If the error is related to the path not existing or permission issues
            print(f"Error validating path: {dropbox_path}, error: {str(e)}")
            return False


def main():
    # Initialize uploader
    uploader = DropboxUploader(DROPBOX_ACCESS_TOKEN)

    # # Example usage: Copy a specific file within Dropbox
    # from_path = "/YamHamelach_data_n_model/test/Book1.xlsx"  # Example file path
    # to_path =   "/Apps/YamHamelach/Book1.xlsx"
    #
    # # Copy a file from one path to another within Dropbox
    # uploader.copy_file_within_dropbox(from_path, to_path)
    #
    # # Alternatively, copy all files in a directory within Dropbox
    # from_folder = "/YamHamelach_data_n_model/test/"
    # to_folder = "/Apps/YamHamelach/"
    #
    # # List files in the source folder and copy them
    # try:
    #     result = uploader.dbx.files_list_folder(from_folder)
    #     for entry in result.entries:
    #         if isinstance(entry, dropbox.files.FileMetadata):
    #             from_file_path = f"{from_folder}{entry.name}"
    #             to_file_path = f"{to_folder}{entry.name}"
    #             uploader.copy_file_within_dropbox(from_file_path, to_file_path)
    # except dropbox.exceptions.ApiError as e:
    #     print(f"Failed to list folder {from_folder}: {str(e)}")

    # Example usage: Upload a directory from local to Dropbox
    from_folder = "/YamHamelach_data_n_model/"
    dropbox_path = "/Apps/YamHamelach"

    successful, failed = uploader.copy_directory_within_dropbox(from_folder, dropbox_path)

    print(f"\nUpload complete!")
    print(f"Successfully uploaded: {successful} files")
    print(f"Failed uploads: {failed} files")

import dropbox

def test_permissions(access_token):
    try:
        dbx = dropbox.Dropbox(access_token)
        # Test metadata read
        files = dbx.files_list_folder('')
        print("Successfully listed files!")
        # Test file content read
        for entry in files.entries[:1]:
            if isinstance(entry, dropbox.files.FileMetadata):
                dbx.files_download(entry.path_display)
                print("Successfully accessed file content!")
                break
        return True
    except dropbox.exceptions.AuthError as e:
        print(f"Auth Error: {e}")
        return False

def list_all_files(dbx):

    files = dbx.files_list_folder('/YamHamelach_data_n_model', recursive=True)
    for entry in files.entries:
        print(entry.name)

def test_dropbox_access(access_token):
    try:
        dbx = dropbox.Dropbox(access_token)

        # Test basic connectivity
        account = dbx.users_get_current_account()
        print(f"Successfully connected to account: {account.name.display_name}")

        # List all files and folders
        print("\nListing all files and folders:")
        list_all_files(dbx)

    except Exception as e:
        print(f"Error: {e}")


# Test your token
if __name__ == "__main__":
    # test_permissions(DROPBOX_ACCESS_TOKEN)
    test_dropbox_access(DROPBOX_ACCESS_TOKEN)
    main()
