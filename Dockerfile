FROM python:3.11

LABEL authors="sushen"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
COPY /home/sushen/projects/Sourcecode-GNN/data/train_data.json /data/train_data.json

EXPOSE 8000

CMD ["sh", "-c","streamlit run app.py --server.port 8501"]