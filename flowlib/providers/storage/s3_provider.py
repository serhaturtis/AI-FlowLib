"""S3 storage provider implementation.

This module provides a concrete implementation of the StorageProvider
for AWS S3 and S3-compatible storage services using boto3.
"""

import os
import logging
from typing import Dict, List, Optional, BinaryIO, Tuple, Union

from ...core.errors import ProviderError, ErrorContext
from .base import StorageProvider, StorageProviderSettings, FileMetadata


logger = logging.getLogger(__name__)

try:
    import boto3
    import botocore
    from botocore.exceptions import ClientError
    from aiobotocore.session import get_session
except ImportError:
    logger.warning(
        "boto3 or aiobotocore packages not found. Install with 'pip install boto3 aiobotocore'"
    )


class S3ProviderSettings(StorageProviderSettings):
    """Settings for S3 storage provider.
    
    Attributes:
        endpoint_url: Custom endpoint URL for S3-compatible services
        region_name: AWS region name
        bucket: S3 bucket name
        access_key_id: AWS access key ID
        secret_access_key: AWS secret access key
        session_token: AWS session token (optional)
        profile_name: AWS profile name (optional, for local credentials)
        path_style: Whether to use path-style addressing
        signature_version: AWS signature version
        validate_checksums: Whether to validate checksums on uploads/downloads
        max_pool_connections: Maximum number of connections to keep in the pool
    """
    
    # S3-specific settings
    endpoint_url: Optional[str] = None
    region_name: Optional[str] = None
    bucket: str = "default"
    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None
    session_token: Optional[str] = None
    profile_name: Optional[str] = None
    path_style: bool = False
    signature_version: str = "s3v4"
    validate_checksums: bool = True
    max_pool_connections: int = 10
    
    # Multipart uploads
    multipart_threshold: int = 8 * 1024 * 1024  # 8MB
    multipart_chunksize: int = 8 * 1024 * 1024  # 8MB
    
    # ACL and caching
    acl: Optional[str] = None  # "private", "public-read", etc.
    cache_control: Optional[str] = None


class S3Provider(StorageProvider):
    """S3 implementation of the StorageProvider.
    
    This provider implements storage operations using boto3,
    supporting AWS S3 and S3-compatible storage services.
    """
    
    def __init__(self, name: str = "s3", settings: Optional[S3ProviderSettings] = None):
        """Initialize S3 provider.
        
        Args:
            name: Unique provider name
            settings: Optional provider settings
        """
        # Create settings first to avoid issues with _default_settings() method
        settings = settings or S3ProviderSettings()
        
        # Pass explicit settings to parent class
        super().__init__(name=name, settings=settings)
        
        # Store settings for local use
        self._settings = settings
        self._client = None
        self._resource = None
        self._async_session = None
        self._async_client = None
        
    async def initialize(self):
        """Initialize the S3 client and check or create the bucket."""
        if self._initialized:
            return
            
        try:
            # Check if boto3 is installed
            if "boto3" not in globals():
                raise ProviderError(
                    message="boto3 package not installed. Install with 'pip install boto3 aiobotocore'",
                    provider_name=self.name
                )
                
            # Create session
            session_kwargs = {}
            if self._settings.profile_name:
                session_kwargs["profile_name"] = self._settings.profile_name
                
            session = boto3.Session(**session_kwargs)
            
            # Create client config
            client_kwargs = {
                "region_name": self._settings.region_name,
                "config": botocore.config.Config(
                    signature_version=self._settings.signature_version,
                    s3={"addressing_style": "path" if self._settings.path_style else "auto"},
                    max_pool_connections=self._settings.max_pool_connections
                )
            }
            
            # Add endpoint_url for S3-compatible services
            if self._settings.endpoint_url:
                client_kwargs["endpoint_url"] = self._settings.endpoint_url
                
            # Add credentials if provided
            if self._settings.access_key_id and self._settings.secret_access_key:
                client_kwargs["aws_access_key_id"] = self._settings.access_key_id
                client_kwargs["aws_secret_access_key"] = self._settings.secret_access_key
                
                if self._settings.session_token:
                    client_kwargs["aws_session_token"] = self._settings.session_token
                    
            # Create S3 client
            self._client = session.client("s3", **client_kwargs)
            
            # Create S3 resource for some operations
            self._resource = session.resource("s3", **client_kwargs)
            
            # Create aiobotocore session for async operations
            self._async_session = get_session()
            
            # Check if bucket exists or create it
            if self._settings.create_bucket:
                await self._ensure_bucket_exists()
                
            self._initialized = True
            logger.debug(f"{self.name} provider initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize {self.name} provider: {str(e)}")
            raise ProviderError(
                message=f"Failed to initialize S3 provider: {str(e)}",
                provider_name=self.name,
                cause=e
            )
            
    async def shutdown(self):
        """Close S3 client and release resources."""
        if not self._initialized:
            return
            
        try:
            # Close async client if it exists
            if self._async_client:
                await self._async_client.close()
                
            # No need to close boto3 clients, they handle their own lifecycle
            self._client = None
            self._resource = None
            self._async_session = None
            self._async_client = None
            self._initialized = False
            logger.debug(f"{self.name} provider shut down successfully")
            
        except Exception as e:
            logger.error(f"Error during {self.name} provider shutdown: {str(e)}")
            raise ProviderError(
                message=f"Failed to shut down S3 provider: {str(e)}",
                provider_name=self.name,
                cause=e
            )
            
    async def create_bucket(self, bucket: Optional[str] = None) -> bool:
        """Create a new bucket/container.
        
        Args:
            bucket: Bucket name (default from settings if None)
            
        Returns:
            True if bucket was created successfully
            
        Raises:
            ProviderError: If bucket creation fails
        """
        if not self._initialized or not self._client:
            raise ProviderError(
                message="Provider not initialized",
                provider_name=self.name
            )
            
        bucket = bucket or self._settings.bucket
        
        try:
            # Check if bucket already exists
            if await self.bucket_exists(bucket):
                logger.debug(f"Bucket {bucket} already exists")
                return True
                
            # Create bucket
            create_bucket_kwargs = {"Bucket": bucket}
            
            # Add location constraint if region is specified
            if self._settings.region_name and self._settings.region_name != "us-east-1":
                create_bucket_kwargs["CreateBucketConfiguration"] = {
                    "LocationConstraint": self._settings.region_name
                }
                
            # Create bucket using sync client (boto3)
            self._client.create_bucket(**create_bucket_kwargs)
            logger.debug(f"Created bucket: {bucket}")
            
            return True
            
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            
            # Handle specific error codes
            if error_code == "BucketAlreadyOwnedByYou":
                logger.debug(f"Bucket {bucket} already owned by you")
                return True
            elif error_code == "BucketAlreadyExists":
                logger.debug(f"Bucket {bucket} already exists but is owned by another account")
                return True
                
            # Wrap and re-raise other errors
            raise ProviderError(
                message=f"Failed to create bucket {bucket}: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(bucket=bucket),
                cause=e
            )
        except Exception as e:
            # Wrap and re-raise other errors
            raise ProviderError(
                message=f"Failed to create bucket {bucket}: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(bucket=bucket),
                cause=e
            )
            
    async def delete_bucket(self, bucket: Optional[str] = None, force: bool = False) -> bool:
        """Delete a bucket/container.
        
        Args:
            bucket: Bucket name (default from settings if None)
            force: Whether to delete all objects in the bucket first
            
        Returns:
            True if bucket was deleted successfully
            
        Raises:
            ProviderError: If bucket deletion fails
        """
        if not self._initialized or not self._client:
            raise ProviderError(
                message="Provider not initialized",
                provider_name=self.name
            )
            
        bucket = bucket or self._settings.bucket
        
        try:
            # Check if bucket exists
            if not await self.bucket_exists(bucket):
                logger.debug(f"Bucket {bucket} doesn't exist, nothing to delete")
                return True
                
            # If force is True, delete all objects in the bucket first
            if force:
                # List all objects in the bucket
                paginator = self._client.get_paginator("list_objects_v2")
                for page in paginator.paginate(Bucket=bucket):
                    if "Contents" in page:
                        # Collect keys to delete
                        keys = [{"Key": obj["Key"]} for obj in page["Contents"]]
                        
                        # Delete objects in a batch
                        if keys:
                            self._client.delete_objects(
                                Bucket=bucket,
                                Delete={"Objects": keys}
                            )
                            
            # Delete bucket using sync client (boto3)
            self._client.delete_bucket(Bucket=bucket)
            logger.debug(f"Deleted bucket: {bucket}")
            
            return True
            
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            
            # Handle specific error codes
            if error_code == "NoSuchBucket":
                logger.debug(f"Bucket {bucket} doesn't exist, nothing to delete")
                return True
            elif error_code == "BucketNotEmpty" and not force:
                raise ProviderError(
                    message=f"Cannot delete non-empty bucket {bucket}. Use force=True to delete all objects first.",
                    provider_name=self.name,
                    context=ErrorContext.create(bucket=bucket)
                )
                
            # Wrap and re-raise other errors
            raise ProviderError(
                message=f"Failed to delete bucket {bucket}: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(bucket=bucket, force=force),
                cause=e
            )
        except Exception as e:
            # Wrap and re-raise other errors
            raise ProviderError(
                message=f"Failed to delete bucket {bucket}: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(bucket=bucket, force=force),
                cause=e
            )
            
    async def bucket_exists(self, bucket: Optional[str] = None) -> bool:
        """Check if a bucket/container exists.
        
        Args:
            bucket: Bucket name (default from settings if None)
            
        Returns:
            True if bucket exists
            
        Raises:
            ProviderError: If check fails
        """
        if not self._initialized or not self._client:
            raise ProviderError(
                message="Provider not initialized",
                provider_name=self.name
            )
            
        bucket = bucket or self._settings.bucket
        
        try:
            # Check if bucket exists using head_bucket
            self._client.head_bucket(Bucket=bucket)
            return True
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            
            # Handle specific error codes
            if error_code in ("404", "NoSuchBucket"):
                return False
            elif error_code in ("403", "AccessDenied"):
                # Bucket exists but we don't have permission to access it
                return True
                
            # Wrap and re-raise other errors
            raise ProviderError(
                message=f"Failed to check if bucket {bucket} exists: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(bucket=bucket),
                cause=e
            )
        except Exception as e:
            # Wrap and re-raise other errors
            raise ProviderError(
                message=f"Failed to check if bucket {bucket} exists: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(bucket=bucket),
                cause=e
            )
            
    async def upload_file(self, file_path: str, object_key: str, bucket: Optional[str] = None,
                        metadata: Optional[Dict[str, str]] = None) -> FileMetadata:
        """Upload a file to storage.
        
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
        if not self._initialized or not self._client:
            raise ProviderError(
                message="Provider not initialized",
                provider_name=self.name
            )
            
        bucket = bucket or self._settings.bucket
        
        try:
            # Ensure bucket exists
            if not await self.bucket_exists(bucket):
                if self._settings.create_bucket:
                    await self.create_bucket(bucket)
                else:
                    raise ProviderError(
                        message=f"Bucket {bucket} does not exist",
                        provider_name=self.name,
                        context=ErrorContext.create(bucket=bucket, object_key=object_key)
                    )
                    
            # Check if file exists
            if not os.path.isfile(file_path):
                raise ProviderError(
                    message=f"File {file_path} does not exist",
                    provider_name=self.name,
                    context=ErrorContext.create(file_path=file_path, object_key=object_key)
                )
                
            # Determine content type based on file extension
            content_type = None
            if self._settings.auto_content_type:
                content_type = self.get_content_type(file_path)
                
            # Prepare upload parameters
            upload_args = {
                "Bucket": bucket,
                "Key": object_key,
                "Filename": file_path
            }
            
            # Add extra parameters if provided
            if content_type:
                upload_args["ContentType"] = content_type
                
            if metadata:
                upload_args["Metadata"] = metadata
                
            if self._settings.acl:
                upload_args["ACL"] = self._settings.acl
                
            if self._settings.cache_control:
                upload_args["CacheControl"] = self._settings.cache_control
                
            # Upload file using boto3's upload_file which supports multipart uploads
            self._client.upload_file(
                Filename=file_path,
                Bucket=bucket,
                Key=object_key,
                ExtraArgs={k: v for k, v in upload_args.items() if k not in ["Bucket", "Key", "Filename"]}
            )
            
            logger.debug(f"Uploaded file {file_path} to {bucket}/{object_key}")
            
            # Get metadata for the uploaded file
            return await self.get_metadata(object_key, bucket)
            
        except ClientError as e:
            # Wrap and re-raise boto3 errors
            raise ProviderError(
                message=f"Failed to upload file {file_path} to {bucket}/{object_key}: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(file_path=file_path, bucket=bucket, object_key=object_key),
                cause=e
            )
        except Exception as e:
            # Wrap and re-raise other errors
            raise ProviderError(
                message=f"Failed to upload file {file_path} to {bucket}/{object_key}: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(file_path=file_path, bucket=bucket, object_key=object_key),
                cause=e
            )
            
    async def upload_data(self, data: Union[bytes, BinaryIO], object_key: str, bucket: Optional[str] = None,
                        content_type: Optional[str] = None, metadata: Optional[Dict[str, str]] = None) -> FileMetadata:
        """Upload binary data to storage.
        
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
        if not self._initialized or not self._client:
            raise ProviderError(
                message="Provider not initialized",
                provider_name=self.name
            )
            
        bucket = bucket or self._settings.bucket
        
        try:
            # Ensure bucket exists
            if not await self.bucket_exists(bucket):
                if self._settings.create_bucket:
                    await self.create_bucket(bucket)
                else:
                    raise ProviderError(
                        message=f"Bucket {bucket} does not exist",
                        provider_name=self.name,
                        context=ErrorContext.create(bucket=bucket, object_key=object_key)
                    )
                    
            # Determine content type if not provided
            if not content_type and self._settings.auto_content_type:
                content_type = self.get_content_type(object_key)
                
            # Prepare upload parameters
            upload_args = {
                "Bucket": bucket,
                "Key": object_key
            }
            
            # Check if data is a file-like object or bytes
            if hasattr(data, "read"):
                upload_args["Body"] = data
            else:
                # Convert to bytes if not already
                if not isinstance(data, bytes):
                    data = bytes(data)
                upload_args["Body"] = data
                
            # Add extra parameters if provided
            if content_type:
                upload_args["ContentType"] = content_type
                
            if metadata:
                upload_args["Metadata"] = metadata
                
            if self._settings.acl:
                upload_args["ACL"] = self._settings.acl
                
            if self._settings.cache_control:
                upload_args["CacheControl"] = self._settings.cache_control
                
            # Upload data using boto3's put_object
            async with self._get_async_client() as client:
                await client.put_object(**upload_args)
                
            logger.debug(f"Uploaded data to {bucket}/{object_key}")
            
            # Get metadata for the uploaded file
            return await self.get_metadata(object_key, bucket)
            
        except ClientError as e:
            # Wrap and re-raise boto3 errors
            raise ProviderError(
                message=f"Failed to upload data to {bucket}/{object_key}: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(bucket=bucket, object_key=object_key),
                cause=e
            )
        except Exception as e:
            # Wrap and re-raise other errors
            raise ProviderError(
                message=f"Failed to upload data to {bucket}/{object_key}: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(bucket=bucket, object_key=object_key),
                cause=e
            )
            
    async def download_file(self, object_key: str, file_path: str, bucket: Optional[str] = None) -> FileMetadata:
        """Download a file from storage.
        
        Args:
            object_key: Storage object key/path
            file_path: Path to save file
            bucket: Bucket name (default from settings if None)
            
        Returns:
            File metadata
            
        Raises:
            ProviderError: If download fails
        """
        if not self._initialized or not self._client:
            raise ProviderError(
                message="Provider not initialized",
                provider_name=self.name
            )
            
        bucket = bucket or self._settings.bucket
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            # Download file using boto3's download_file
            self._client.download_file(
                Bucket=bucket,
                Key=object_key,
                Filename=file_path
            )
            
            logger.debug(f"Downloaded {bucket}/{object_key} to {file_path}")
            
            # Get metadata for the downloaded file
            return await self.get_metadata(object_key, bucket)
            
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            
            # Handle specific error codes
            if error_code in ("404", "NoSuchKey"):
                raise ProviderError(
                    message=f"Object {object_key} does not exist in bucket {bucket}",
                    provider_name=self.name,
                    context=ErrorContext.create(bucket=bucket, object_key=object_key),
                    cause=e
                )
                
            # Wrap and re-raise other errors
            raise ProviderError(
                message=f"Failed to download {bucket}/{object_key} to {file_path}: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(bucket=bucket, object_key=object_key, file_path=file_path),
                cause=e
            )
        except Exception as e:
            # Wrap and re-raise other errors
            raise ProviderError(
                message=f"Failed to download {bucket}/{object_key} to {file_path}: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(bucket=bucket, object_key=object_key, file_path=file_path),
                cause=e
            )
            
    async def download_data(self, object_key: str, bucket: Optional[str] = None) -> Tuple[bytes, FileMetadata]:
        """Download binary data from storage.
        
        Args:
            object_key: Storage object key/path
            bucket: Bucket name (default from settings if None)
            
        Returns:
            Tuple of (data bytes, file metadata)
            
        Raises:
            ProviderError: If download fails
        """
        if not self._initialized or not self._client:
            raise ProviderError(
                message="Provider not initialized",
                provider_name=self.name
            )
            
        bucket = bucket or self._settings.bucket
        
        try:
            # Get object using async client
            async with self._get_async_client() as client:
                response = await client.get_object(Bucket=bucket, Key=object_key)
                
                # Read all data from stream
                data = await response["Body"].read()
                await response["Body"].close()
                
            logger.debug(f"Downloaded data from {bucket}/{object_key}")
            
            # Get metadata
            metadata = await self.get_metadata(object_key, bucket)
            
            return data, metadata
            
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            
            # Handle specific error codes
            if error_code in ("404", "NoSuchKey"):
                raise ProviderError(
                    message=f"Object {object_key} does not exist in bucket {bucket}",
                    provider_name=self.name,
                    context=ErrorContext.create(bucket=bucket, object_key=object_key),
                    cause=e
                )
                
            # Wrap and re-raise other errors
            raise ProviderError(
                message=f"Failed to download data from {bucket}/{object_key}: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(bucket=bucket, object_key=object_key),
                cause=e
            )
        except Exception as e:
            # Wrap and re-raise other errors
            raise ProviderError(
                message=f"Failed to download data from {bucket}/{object_key}: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(bucket=bucket, object_key=object_key),
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
        if not self._initialized or not self._client:
            raise ProviderError(
                message="Provider not initialized",
                provider_name=self.name
            )
            
        bucket = bucket or self._settings.bucket
        
        try:
            # Get object metadata using head_object
            response = self._client.head_object(Bucket=bucket, Key=object_key)
            
            # Extract metadata from response
            metadata = FileMetadata(
                key=object_key,
                size=response.get("ContentLength", 0),
                etag=response.get("ETag", "").strip('"'),
                content_type=response.get("ContentType", "application/octet-stream"),
                modified=response.get("LastModified"),
                metadata=response.get("Metadata", {})
            )
            
            return metadata
            
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            
            # Handle specific error codes
            if error_code in ("404", "NoSuchKey"):
                raise ProviderError(
                    message=f"Object {object_key} does not exist in bucket {bucket}",
                    provider_name=self.name,
                    context=ErrorContext.create(bucket=bucket, object_key=object_key),
                    cause=e
                )
                
            # Wrap and re-raise other errors
            raise ProviderError(
                message=f"Failed to get metadata for {bucket}/{object_key}: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(bucket=bucket, object_key=object_key),
                cause=e
            )
        except Exception as e:
            # Wrap and re-raise other errors
            raise ProviderError(
                message=f"Failed to get metadata for {bucket}/{object_key}: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(bucket=bucket, object_key=object_key),
                cause=e
            )
            
    async def delete_object(self, object_key: str, bucket: Optional[str] = None) -> bool:
        """Delete a storage object.
        
        Args:
            object_key: Storage object key/path
            bucket: Bucket name (default from settings if None)
            
        Returns:
            True if object was deleted successfully
            
        Raises:
            ProviderError: If deletion fails
        """
        if not self._initialized or not self._client:
            raise ProviderError(
                message="Provider not initialized",
                provider_name=self.name
            )
            
        bucket = bucket or self._settings.bucket
        
        try:
            # Delete object using async client
            async with self._get_async_client() as client:
                await client.delete_object(Bucket=bucket, Key=object_key)
                
            logger.debug(f"Deleted object {bucket}/{object_key}")
            return True
            
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            
            # Handle specific error codes
            if error_code in ("404", "NoSuchKey"):
                logger.debug(f"Object {object_key} does not exist in bucket {bucket}, nothing to delete")
                return True
                
            # Wrap and re-raise other errors
            raise ProviderError(
                message=f"Failed to delete object {bucket}/{object_key}: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(bucket=bucket, object_key=object_key),
                cause=e
            )
        except Exception as e:
            # Wrap and re-raise other errors
            raise ProviderError(
                message=f"Failed to delete object {bucket}/{object_key}: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(bucket=bucket, object_key=object_key),
                cause=e
            )
            
    async def list_objects(self, prefix: Optional[str] = None, bucket: Optional[str] = None) -> List[FileMetadata]:
        """List objects in a bucket/container.
        
        Args:
            prefix: Optional prefix to filter objects
            bucket: Bucket name (default from settings if None)
            
        Returns:
            List of file metadata
            
        Raises:
            ProviderError: If listing fails
        """
        if not self._initialized or not self._client:
            raise ProviderError(
                message="Provider not initialized",
                provider_name=self.name
            )
            
        bucket = bucket or self._settings.bucket
        
        try:
            # List objects using boto3's list_objects_v2
            result = []
            paginator = self._client.get_paginator("list_objects_v2")
            
            # Prepare list parameters
            list_args = {"Bucket": bucket}
            if prefix:
                list_args["Prefix"] = prefix
                
            # Paginate through results
            for page in paginator.paginate(**list_args):
                if "Contents" in page:
                    for obj in page["Contents"]:
                        # Create file metadata for each object
                        metadata = FileMetadata(
                            key=obj["Key"],
                            size=obj["Size"],
                            etag=obj.get("ETag", "").strip('"'),
                            modified=obj.get("LastModified"),
                            content_type="application/octet-stream"  # Default, we'd need a head_object call to get the real content type
                        )
                        result.append(metadata)
                        
            return result
            
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            
            # Handle specific error codes
            if error_code in ("404", "NoSuchBucket"):
                raise ProviderError(
                    message=f"Bucket {bucket} does not exist",
                    provider_name=self.name,
                    context=ErrorContext.create(bucket=bucket, prefix=prefix),
                    cause=e
                )
                
            # Wrap and re-raise other errors
            raise ProviderError(
                message=f"Failed to list objects in {bucket} with prefix {prefix}: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(bucket=bucket, prefix=prefix),
                cause=e
            )
        except Exception as e:
            # Wrap and re-raise other errors
            raise ProviderError(
                message=f"Failed to list objects in {bucket} with prefix {prefix}: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(bucket=bucket, prefix=prefix),
                cause=e
            )
            
    async def generate_presigned_url(self, object_key: str, expiration: int = 3600, bucket: Optional[str] = None,
                                  operation: str = "get") -> str:
        """Generate a presigned URL for a storage object.
        
        Args:
            object_key: Storage object key/path
            expiration: URL expiration in seconds
            bucket: Bucket name (default from settings if None)
            operation: Operation type ('get', 'put', 'delete')
            
        Returns:
            Presigned URL
            
        Raises:
            ProviderError: If URL generation fails
        """
        if not self._initialized or not self._client:
            raise ProviderError(
                message="Provider not initialized",
                provider_name=self.name
            )
            
        bucket = bucket or self._settings.bucket
        
        # Map operation to S3 client method
        operation_map = {
            "get": "get_object",
            "put": "put_object",
            "delete": "delete_object",
            "head": "head_object"
        }
        
        client_method = operation_map.get(operation.lower())
        if not client_method:
            raise ProviderError(
                message=f"Invalid operation: {operation}. Must be one of: {', '.join(operation_map.keys())}",
                provider_name=self.name,
                context=ErrorContext.create(bucket=bucket, object_key=object_key, operation=operation)
            )
            
        try:
            # Generate presigned URL
            url = self._client.generate_presigned_url(
                ClientMethod=client_method,
                Params={"Bucket": bucket, "Key": object_key},
                ExpiresIn=expiration
            )
            
            return url
            
        except Exception as e:
            # Wrap and re-raise errors
            raise ProviderError(
                message=f"Failed to generate presigned URL for {bucket}/{object_key}: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(bucket=bucket, object_key=object_key, operation=operation),
                cause=e
            )
            
    async def check_connection(self) -> bool:
        """Check if storage connection is active.
        
        Returns:
            True if connection is active, False otherwise
        """
        if not self._initialized or not self._client:
            return False
            
        try:
            # Try to list buckets
            self._client.list_buckets()
            return True
        except Exception:
            return False
            
    async def _ensure_bucket_exists(self):
        """Ensure that the configured bucket exists, creating it if necessary."""
        bucket = self._settings.bucket
        
        # Check if bucket exists
        if not await self.bucket_exists(bucket):
            # Create bucket if it doesn't exist
            if self._settings.create_bucket:
                await self.create_bucket(bucket)
            else:
                raise ProviderError(
                    message=f"Bucket {bucket} does not exist and create_bucket is set to False",
                    provider_name=self.name,
                    context=ErrorContext.create(bucket=bucket)
                )
                
    async def _get_async_client(self):
        """Get or create an async S3 client."""
        if not self._async_session:
            self._async_session = get_session()
            
        # Create client config for the session
        client_kwargs = {}
        
        # Add region if specified
        if self._settings.region_name:
            client_kwargs["region_name"] = self._settings.region_name
            
        # Add endpoint URL for S3-compatible services
        if self._settings.endpoint_url:
            client_kwargs["endpoint_url"] = self._settings.endpoint_url
            
        # Add credentials if provided
        if self._settings.access_key_id and self._settings.secret_access_key:
            client_kwargs["aws_access_key_id"] = self._settings.access_key_id
            client_kwargs["aws_secret_access_key"] = self._settings.secret_access_key
            
            if self._settings.session_token:
                client_kwargs["aws_session_token"] = self._settings.session_token
                
        # Add config
        client_kwargs["config"] = botocore.config.Config(
            signature_version=self._settings.signature_version,
            s3={"addressing_style": "path" if self._settings.path_style else "auto"},
            max_pool_connections=self._settings.max_pool_connections
        )
        
        # Create and return the async client
        return self._async_session.create_client("s3", **client_kwargs) 