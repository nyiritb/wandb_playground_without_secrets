import os
import boto3

# Parse environment variables.
s3_bucket = os.environ['S3_BUCKET']
local_artifact_path = os.environ['LOCAL_ARTIFACT_PATH']
s3_artifact_key = os.environ['S3_ARTIFACT_KEY']

# Upload to S3.
s3 = boto3.client('s3')
print(f"Uploading {local_artifact_path} to {s3_bucket}/{s3_artifact_key} in S3.")
s3.upload_file(local_artifact_path, s3_bucket, s3_artifact_key)