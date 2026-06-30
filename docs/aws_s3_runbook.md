# AWS S3 Runbook

## Purpose

Amazon S3 is optional in this project. It is used only as a raw-data object
storage and portfolio demonstration, separate from the validated local
PySpark/Delta pipeline and separate from the Databricks Free Edition bundle.

The current bucket is:

- Bucket: `marketplace-lakehouse-demo-9623`
- Region: `ap-south-1`
- Suggested raw prefix: `raw/olist/`

## Local Verification

Store real AWS credentials only in your local `.env` file, shell environment,
AWS CLI profile, or another boto3-supported local credential provider. Do not
commit credentials.

Run the read-only verifier:

```powershell
python aws/verify_s3.py
```

The script requires:

- `AWS_S3_BUCKET`
- `AWS_DEFAULT_REGION`

The script uses boto3 default credential resolution and calls only
`list_objects_v2`. It does not upload, delete, rename, or mutate S3 objects.

## Current Integration Boundary

The AWS S3 bucket and Databricks Free Edition medallion pipeline are validated
separately. They are not live-integrated in this repository.

Databricks Free Edition is used for a separate Unity Catalog managed-table
demonstration. This project does not claim live AWS-to-Databricks connectivity.

## Least-Privilege IAM

The reference policy is stored in `aws/iam_policy.json`. It is scoped only to
`marketplace-lakehouse-demo-9623` and permits:

- `s3:ListBucket`
- `s3:GetBucketLocation`
- `s3:GetObject`
- `s3:PutObject`

It intentionally excludes `s3:DeleteObject` and wildcard access to all buckets.

## Production Architecture

Designed, not provisioned in this free-tier project:

1. S3 raw zone stores immutable source extracts under a controlled prefix.
2. An IAM role grants least-privilege access to that raw zone.
3. Databricks uses a storage credential backed by that IAM role.
4. Unity Catalog defines an external location for the S3 raw zone.
5. Bronze, Silver, and Gold Delta tables are built from governed external
   storage into the lakehouse.

That production connection is designed, not provisioned in this free-tier
project.
