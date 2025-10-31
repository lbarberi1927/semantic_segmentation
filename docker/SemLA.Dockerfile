FROM nvcr.io/nvidia/pytorch:23.10-py3

COPY ./SemLA /app/SemLA
WORKDIR /app/SemLA
RUN pip install -r requirements.txt

COPY ./app /app/app
