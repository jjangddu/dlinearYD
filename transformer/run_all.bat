@echo off

python main.py
python baseline.py
python upload_to_S3.py

pause
