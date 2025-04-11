"""Local file storage provider implementation.

This module provides a concrete implementation of the StorageProvider
for local filesystem storage operations.
"""

import logging
import os
import shutil
import mimetypes
import hashlib
import pathlib
from typing import Dict, List, Optional, BinaryIO, Tuple, Union
from datetime import datetime
from pydantic import field_validator

from ...core.errors import ProviderError, ErrorContext
from ..decorators import provider
from ..constants import ProviderType
from .base import StorageProvider, StorageProviderSettings, FileMetadata

logger = logging.getLogger(__name__)


class LocalStorageProviderSettings(StorageProviderSettings):
    """Settings for local file storage provider.
    
    Attributes:
        base_path: Base directory path for file operations
        create_dirs: Whether to create directories if they don't exist
        use_relative_paths: Whether to interpret object keys as relative paths
        permissions: Default file permissions (Unix style, e.g. 0o644)
    """
    
    # Local file storage settings
    base_path: str
    create_dirs: bool = True
    use_relative_paths: bool = True
    permissions: Optional[int] = None
    
    @field_validator('base_path')
    def validate_base_path(cls, v):
        """Validate that base_path is absolute and normalized."""
        base_path = os.path.abspath(os.path.normpath(v))
        # Ensure base path ends with a trailing slash
        if not base_path.endswith(os.path.sep):
            base_path += os.path.sep
        return base_path


@provider(provider_type=ProviderType.STORAGE, name="local-storage")
class LocalStorageProvider(StorageProvider):
    """Local filesystem implementation of the StorageProvider.
    
    This provider implements storage operations using the local filesystem,
    useful for development, testing, or deployments without cloud storage.
    """
    
    def __init__(self, name: str = "local-storage", settings: Optional[LocalStorageProviderSettings] = None):
        """Initialize local file storage provider.
        
        Args:
            name: Provider instance name
            settings: Optional provider settings
        """
        super().__init__(name=name, settings=settings or LocalStorageProviderSettings())
        
    async def initialize(self):
        """Initialize the local file storage provider."""
        if self._initialized:
            return
            
        try:
            # Create base directory if it doesn't exist
            if self._settings.create_dirs and not os.path.exists(self._settings.base_path):
                os.makedirs(self._settings.base_path, exist_ok=True)
                logger.debug(f"Created base directory: {self._settings.base_path}")
                
            # Check if base directory is writable
            if not os.access(self._settings.base_path, os.W_OK):
                raise ProviderError(
                    message=f"Base directory {self._settings.base_path} is not writable",
                    provider_name=self.name
                )
                
            self._initialized = True
            logger.debug(f"{self.name} provider initialized successfully")
            
        except Exception as e:
            if isinstance(e, ProviderError):
                raise e
                
            logger.error(f"Failed to initialize {self.name} provider: {str(e)}")
            raise ProviderError(
                message=f"Failed to initialize local file storage provider: {str(e)}",
                provider_name=self.name,
                cause=e
            )
            
    async def shutdown(self):
        """Shut down local file storage provider."""
        self._initialized = False
        logger.debug(f"{self.name} provider shut down successfully")
            
    def _get_bucket_path(self, bucket: Optional[str] = None) -> str:
        """Get the absolute path for a bucket.
        
        Args:
            bucket: Bucket name (default from settings if None)
            
        Returns:
            Absolute path to the bucket directory
        """
        bucket = bucket or self._settings.bucket
        return os.path.join(self._settings.base_path, bucket)
        
    def _get_object_path(self, object_key: str, bucket: Optional[str] = None) -> str:
        """Get the absolute path for an object.
        
        Args:
            object_key: Object key/path
            bucket: Bucket name (default from settings if None)
            
        Returns:
            Absolute path to the object file
        """
        bucket_path = self._get_bucket_path(bucket)
        
        # Normalize object key to handle both Windows and Unix paths
        if self._settings.use_relative_paths:
            # Convert to posix path and remove leading slashes to avoid escaping the bucket
            object_key = object_key.lstrip('/').lstrip('\\')
            
        return os.path.join(bucket_path, object_key)
        
    async def create_bucket(self, bucket: Optional[str] = None) -> bool:
        """Create a new bucket (directory).
        
        Args:
            bucket: Bucket name (default from settings if None)
            
        Returns:
            True if bucket was created successfully
            
        Raises:
            ProviderError: If bucket creation fails
        """
        if not self._initialized:
            raise ProviderError(
                message="Provider not initialized",
                provider_name=self.name
            )
            
        bucket = bucket or self._settings.bucket
        bucket_path = self._get_bucket_path(bucket)
        
        try:
            # Check if bucket already exists
            if os.path.exists(bucket_path):
                return True
                
            # Create bucket directory
            os.makedirs(bucket_path, exist_ok=True)
            logger.debug(f"Created bucket directory: {bucket_path}")
            
            return True
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to create bucket directory {bucket}: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(bucket=bucket, path=bucket_path),
                cause=e
            )
        
    async def delete_bucket(self, bucket: Optional[str] = None, force: bool = False) -> bool:
        """Delete a bucket (directory).
        
        Args:
            bucket: Bucket name (default from settings if None)
            force: Whether to delete all objects in the bucket first
            
        Returns:
            True if bucket was deleted successfully
            
        Raises:
            ProviderError: If bucket deletion fails
        """
        if not self._initialized:
            raise ProviderError(
                message="Provider not initialized",
                provider_name=self.name
            )
            
        bucket = bucket or self._settings.bucket
        bucket_path = self._get_bucket_path(bucket)
        
        try:
            # Check if bucket exists
            if not os.path.exists(bucket_path):
                return True
                
            # Check if bucket is empty or force flag is set
            if not force and os.listdir(bucket_path):
                raise ProviderError(
                    message=f"Bucket {bucket} is not empty. Use force=True to delete it anyway.",
                    provider_name=self.name,
                    context=ErrorContext.create(bucket=bucket, path=bucket_path)
                )
                
            # Delete bucket directory
            shutil.rmtree(bucket_path)
            logger.debug(f"Deleted bucket directory: {bucket_path}")
            
            return True
            
        except Exception as e:
            if isinstance(e, ProviderError):
                raise e
                
            raise ProviderError(
                message=f"Failed to delete bucket directory {bucket}: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(bucket=bucket, path=bucket_path),
                cause=e
            )
        
    async def bucket_exists(self, bucket: Optional[str] = None) -> bool:
        """Check if a bucket (directory) exists.
        
        Args:
            bucket: Bucket name (default from settings if None)
            
        Returns:
            True if bucket exists
            
        Raises:
            ProviderError: If check fails
        """
        if not self._initialized:
            raise ProviderError(
                message="Provider not initialized",
                provider_name=self.name
            )
            
        bucket = bucket or self._settings.bucket
        bucket_path = self._get_bucket_path(bucket)
        
        try:
            return os.path.exists(bucket_path) and os.path.isdir(bucket_path)
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to check if bucket {bucket} exists: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(bucket=bucket, path=bucket_path),
                cause=e
            )
        
    async def upload_file(self, file_path: str, object_key: str, bucket: Optional[str] = None,
                        metadata: Optional[Dict[str, str]] = None) -> FileMetadata:
        """Upload a file to local storage.
        
        Args:
            file_path: Path to local file
            object_key: Storage object key/path
            bucket: Bucket name (default from settings if None)
            metadata: Optional object metadata
            
        Returns:
            File metadata
            
        Raises:
            ProviderError: If upload fails
        """
        if not self._initialized:
            raise ProviderError(
                message="Provider not initialized",
                provider_name=self.name
            )
            
        bucket = bucket or self._settings.bucket
        file_path = os.path.abspath(file_path)
        object_path = self._get_object_path(object_key, bucket)
        
        try:
            # Check if source file exists
            if not os.path.exists(file_path):
                raise ProviderError(
                    message=f"Source file {file_path} does not exist",
                    provider_name=self.name,
                    context=ErrorContext.create(file_path=file_path)
                )
                
            # Create directory for the object if it doesn't exist
            os.makedirs(os.path.dirname(object_path), exist_ok=True)
            
            # Copy file
            shutil.copy2(file_path, object_path)
            
            # Set permissions if specified
            if self._settings.permissions is not None:
                os.chmod(object_path, self._settings.permissions)
                
            # Get file metadata
            stat = os.stat(object_path)
            size = stat.st_size
            modified = datetime.fromtimestamp(stat.st_mtime)
            
            # Calculate MD5 hash for etag
            md5_hash = hashlib.md5()
            with open(object_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    md5_hash.update(chunk)
            etag = md5_hash.hexdigest()
            
            # Determine content type
            content_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
            
            # Create metadata object
            file_metadata = FileMetadata(
                key=object_key,
                size=size,
                etag=etag,
                content_type=content_type,
                modified=modified,
                metadata=metadata or {}
            )
            
            logger.debug(f"Uploaded file to {object_key} in bucket {bucket}")
            
            return file_metadata
            
        except Exception as e:
            if isinstance(e, ProviderError):
                raise e
                
            raise ProviderError(
                message=f"Failed to upload file to {object_key}: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    file_path=file_path,
                    object_key=object_key,
                    bucket=bucket
                ),
                cause=e
            )
        
    async def upload_data(self, data: Union[bytes, BinaryIO], object_key: str, bucket: Optional[str] = None,
                        content_type: Optional[str] = None, metadata: Optional[Dict[str, str]] = None) -> FileMetadata:
        """Upload binary data to local storage.
        
        Args:
            data: Binary data or file-like object
            object_key: Storage object key/path
            bucket: Bucket name (default from settings if None)
            content_type: Optional content type
            metadata: Optional object metadata
            
        Returns:
            File metadata
            
        Raises:
            ProviderError: If upload fails
        """
        if not self._initialized:
            raise ProviderError(
                message="Provider not initialized",
                provider_name=self.name
            )
            
        bucket = bucket or self._settings.bucket
        object_path = self._get_object_path(object_key, bucket)
        
        try:
            # Create directory for the object if it doesn't exist
            os.makedirs(os.path.dirname(object_path), exist_ok=True)
            
            # Write data to file
            md5_hash = hashlib.md5()
            
            with open(object_path, 'wb') as f:
                if isinstance(data, bytes):
                    f.write(data)
                    md5_hash.update(data)
                else:
                    # File-like object
                    chunk = data.read(4096)
                    while chunk:
                        f.write(chunk)
                        md5_hash.update(chunk)
                        chunk = data.read(4096)
            
            # Set permissions if specified
            if self._settings.permissions is not None:
                os.chmod(object_path, self._settings.permissions)
                
            # Get file metadata
            stat = os.stat(object_path)
            size = stat.st_size
            modified = datetime.fromtimestamp(stat.st_mtime)
            
            # Calculate etag
            etag = md5_hash.hexdigest()
            
            # Determine content type
            if content_type is None and self._settings.auto_content_type:
                content_type = mimetypes.guess_type(object_key)[0] or "application/octet-stream"
            else:
                content_type = content_type or "application/octet-stream"
                
            # Create metadata object
            file_metadata = FileMetadata(
                key=object_key,
                size=size,
                etag=etag,
                content_type=content_type,
                modified=modified,
                metadata=metadata or {}
            )
            
            logger.debug(f"Uploaded data to {object_key} in bucket {bucket}")
            
            return file_metadata
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to upload data to {object_key}: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    object_key=object_key,
                    bucket=bucket
                ),
                cause=e
            )
        
    async def download_file(self, object_key: str, file_path: str, bucket: Optional[str] = None) -> FileMetadata:
        """Download a file from local storage.
        
        Args:
            object_key: Storage object key/path
            file_path: Path to save file
            bucket: Bucket name (default from settings if None)
            
        Returns:
            File metadata
            
        Raises:
            ProviderError: If download fails
        """
        if not self._initialized:
            raise ProviderError(
                message="Provider not initialized",
                provider_name=self.name
            )
            
        bucket = bucket or self._settings.bucket
        object_path = self._get_object_path(object_key, bucket)
        file_path = os.path.abspath(file_path)
        
        try:
            # Check if source object exists
            if not os.path.exists(object_path):
                raise ProviderError(
                    message=f"Object {object_key} does not exist in bucket {bucket}",
                    provider_name=self.name,
                    context=ErrorContext.create(
                        object_key=object_key,
                        bucket=bucket,
                        path=object_path
                    )
                )
                
            # Create directory for the destination file if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Copy file
            shutil.copy2(object_path, file_path)
            
            # Get file metadata
            return await self.get_metadata(object_key, bucket)
            
        except Exception as e:
            if isinstance(e, ProviderError):
                raise e
                
            raise ProviderError(
                message=f"Failed to download object {object_key}: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    object_key=object_key,
                    bucket=bucket,
                    file_path=file_path
                ),
                cause=e
            )
        
    async def download_data(self, object_key: str, bucket: Optional[str] = None) -> Tuple[bytes, FileMetadata]:
        """Download binary data from local storage.
        
        Args:
            object_key: Storage object key/path
            bucket: Bucket name (default from settings if None)
            
        Returns:
            Tuple of (data bytes, file metadata)
            
        Raises:
            ProviderError: If download fails
        """
        if not self._initialized:
            raise ProviderError(
                message="Provider not initialized",
                provider_name=self.name
            )
            
        bucket = bucket or self._settings.bucket
        object_path = self._get_object_path(object_key, bucket)
        
        try:
            # Check if object exists
            if not os.path.exists(object_path):
                raise ProviderError(
                    message=f"Object {object_key} does not exist in bucket {bucket}",
                    provider_name=self.name,
                    context=ErrorContext.create(
                        object_key=object_key,
                        bucket=bucket,
                        path=object_path
                    )
                )
                
            # Read file data
            with open(object_path, 'rb') as f:
                data = f.read()
                
            # Get file metadata
            metadata = await self.get_metadata(object_key, bucket)
            
            return data, metadata
            
        except Exception as e:
            if isinstance(e, ProviderError):
                raise e
                
            raise ProviderError(
                message=f"Failed to download object data {object_key}: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    object_key=object_key,
                    bucket=bucket
                ),
                cause=e
            )
        
    async def get_metadata(self, object_key: str, bucket: Optional[str] = None) -> FileMetadata:
        """Get metadata for a storage object.
        
        Args:
            object_key: Storage object key/path
            bucket: Bucket name (default from settings if None)
            
        Returns:
            File metadata
            
        Raises:
            ProviderError: If metadata retrieval fails
        """
        if not self._initialized:
            raise ProviderError(
                message="Provider not initialized",
                provider_name=self.name
            )
            
        bucket = bucket or self._settings.bucket
        object_path = self._get_object_path(object_key, bucket)
        
        try:
            # Check if object exists
            if not os.path.exists(object_path):
                raise ProviderError(
                    message=f"Object {object_key} does not exist in bucket {bucket}",
                    provider_name=self.name,
                    context=ErrorContext.create(
                        object_key=object_key,
                        bucket=bucket,
                        path=object_path
                    )
                )
                
            # Get file stats
            stat = os.stat(object_path)
            size = stat.st_size
            modified = datetime.fromtimestamp(stat.st_mtime)
            
            # Calculate MD5 hash for etag
            md5_hash = hashlib.md5()
            with open(object_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    md5_hash.update(chunk)
            etag = md5_hash.hexdigest()
            
            # Determine content type
            content_type = mimetypes.guess_type(object_path)[0] or "application/octet-stream"
            
            # Create metadata object
            file_metadata = FileMetadata(
                key=object_key,
                size=size,
                etag=etag,
                content_type=content_type,
                modified=modified,
                metadata={}  # Local storage doesn't support object metadata
            )
            
            return file_metadata
            
        except Exception as e:
            if isinstance(e, ProviderError):
                raise e
                
            raise ProviderError(
                message=f"Failed to get metadata for {object_key}: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    object_key=object_key,
                    bucket=bucket
                ),
                cause=e
            )
        
    async def delete_object(self, object_key: str, bucket: Optional[str] = None) -> bool:
        """Delete an object from local storage.
        
        Args:
            object_key: Storage object key/path
            bucket: Bucket name (default from settings if None)
            
        Returns:
            True if object was deleted successfully
            
        Raises:
            ProviderError: If deletion fails
        """
        if not self._initialized:
            raise ProviderError(
                message="Provider not initialized",
                provider_name=self.name
            )
            
        bucket = bucket or self._settings.bucket
        object_path = self._get_object_path(object_key, bucket)
        
        try:
            # Check if object exists
            if not os.path.exists(object_path):
                return True
                
            # Delete object
            if os.path.isdir(object_path):
                shutil.rmtree(object_path)
            else:
                os.remove(object_path)
                
            logger.debug(f"Deleted object {object_key} from bucket {bucket}")
            
            return True
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to delete object {object_key}: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    object_key=object_key,
                    bucket=bucket
                ),
                cause=e
            )
        
    async def list_objects(self, prefix: Optional[str] = None, bucket: Optional[str] = None) -> List[FileMetadata]:
        """List objects in local storage.
        
        Args:
            prefix: Optional prefix to filter objects
            bucket: Bucket name (default from settings if None)
            
        Returns:
            List of file metadata
            
        Raises:
            ProviderError: If listing fails
        """
        if not self._initialized:
            raise ProviderError(
                message="Provider not initialized",
                provider_name=self.name
            )
            
        bucket = bucket or self._settings.bucket
        bucket_path = self._get_bucket_path(bucket)
        
        try:
            # Check if bucket exists
            if not os.path.exists(bucket_path):
                return []
                
            results = []
            prefix_path = ''
            
            if prefix:
                # Normalize prefix path
                prefix = prefix.replace('/', os.path.sep)
                prefix_path = os.path.join(bucket_path, prefix)
                
                # If prefix is a directory, list contents
                if os.path.isdir(prefix_path):
                    base_dir = prefix_path
                    prefix = ''
                else:
                    base_dir = bucket_path
            else:
                base_dir = bucket_path
                prefix = ''
                
            # Walk directory tree
            for root, dirs, files in os.walk(base_dir):
                # Skip files not under prefix
                if prefix and not root.startswith(prefix_path):
                    continue
                    
                # Process files
                for file in files:
                    file_path = os.path.join(root, file)
                    # Generate object key relative to bucket path
                    rel_path = os.path.relpath(file_path, bucket_path)
                    object_key = rel_path.replace(os.path.sep, '/')
                    
                    # Get file metadata
                    try:
                        metadata = await self.get_metadata(object_key, bucket)
                        results.append(metadata)
                    except Exception as e:
                        logger.warning(f"Error getting metadata for {object_key}: {str(e)}")
            
            return results
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to list objects in bucket {bucket}: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    bucket=bucket,
                    prefix=prefix
                ),
                cause=e
            )
        
    async def generate_presigned_url(self, object_key: str, expiration: int = 3600, bucket: Optional[str] = None,
                                  operation: str = "get") -> str:
        """Generate a presigned URL for an object.
        
        For local storage, this returns a file:// URL to the object.
        
        Args:
            object_key: Storage object key/path
            expiration: Expiration time in seconds (ignored for local storage)
            bucket: Bucket name (default from settings if None)
            operation: Operation type ('get', 'put', etc.) (ignored for local storage)
            
        Returns:
            File URL
            
        Raises:
            ProviderError: If URL generation fails
        """
        if not self._initialized:
            raise ProviderError(
                message="Provider not initialized",
                provider_name=self.name
            )
            
        bucket = bucket or self._settings.bucket
        object_path = self._get_object_path(object_key, bucket)
        
        try:
            # Check if object exists for GET operation
            if operation.lower() == "get" and not os.path.exists(object_path):
                raise ProviderError(
                    message=f"Object {object_key} does not exist in bucket {bucket}",
                    provider_name=self.name,
                    context=ErrorContext.create(
                        object_key=object_key,
                        bucket=bucket
                    )
                )
                
            # Create file:// URL
            url = pathlib.Path(object_path).as_uri()
            
            return url
            
        except Exception as e:
            if isinstance(e, ProviderError):
                raise e
                
            raise ProviderError(
                message=f"Failed to generate URL for {object_key}: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    object_key=object_key,
                    bucket=bucket,
                    operation=operation
                ),
                cause=e
            )
        
    async def check_connection(self) -> bool:
        """Check if local storage is accessible.
        
        Returns:
            True if local storage is accessible
        """
        if not self._initialized:
            await self.initialize()
            
        try:
            # Check if base directory exists and is writable
            base_path = self._settings.base_path
            return os.path.exists(base_path) and os.access(base_path, os.W_OK)
        except Exception as e:
            logger.error(f"Local storage connection check failed: {str(e)}")
            return False 