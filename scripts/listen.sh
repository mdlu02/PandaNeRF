#!/bin/zsh

S3_BUCKET="your-bucket-name"
DEST_DIR="path/to/data"

while true; do
    aws s3 sync "s3://${S3_BUCKET}" "${DEST_DIR}"
    echo "Synced at $(date)"
    sleep 1
done
