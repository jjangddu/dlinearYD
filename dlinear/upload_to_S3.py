import os
from dotenv import load_dotenv
import boto3
from botocore.exceptions import NoCredentialsError

# .env 파일의 환경 변수 로드
load_dotenv()

aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_DEFAULT_REGION")
bucket_name = os.getenv("AWS_S3_BUCKET")  # 업로드할 S3 버킷 이름

# S3 클라이언트 생성
s3_client = boto3.client(
    's3',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=aws_region
)

def upload_file_to_s3(file_name, bucket, object_name=None):
    if object_name is None:
        object_name = file_name  # S3 상에서 파일 이름을 동일하게 설정
    try:
        s3_client.upload_file(file_name, bucket, object_name)
        print(f"✅ {file_name} 파일이 s3://{bucket}/{object_name} 에 업로드되었습니다.")
    except FileNotFoundError:
        print(f"오류: {file_name} 파일을 찾을 수 없습니다.")
    except NoCredentialsError:
        print("오류: AWS 자격 증명이 설정되어 있지 않습니다.")
    except Exception as e:
        print(f"오류 발생: {str(e)}")

# 업로드할 파일 목록
files_to_upload = [
    # "BTC-USD_data.json"
    "GOOGL_data.json",
    "GOOGL_prediction.json",
    "AAPL_data.json",
    "AAPL_prediction.json",
    "NVDA_data.json",
    "NVDA_prediction.json",
    "AMZN_data.json",
    "AMZN_prediction.json",
    "TSLA_data.json",
    "TSLA_prediction.json",
    "META_data.json",
    "META_prediction.json",

    # "stocks.json"
    "AAPL_investor_results.json",
    "GOOGL_investor_results.json",
    "NVDA_investor_results.json",
    "AMZN_investor_results.json",
    "TSLA_investor_results.json",
    "META_investor_results.json"
]

print(files_to_upload)


# 각 파일을 S3 버킷에 업로드
for file in files_to_upload:
    upload_file_to_s3(file, bucket_name)
