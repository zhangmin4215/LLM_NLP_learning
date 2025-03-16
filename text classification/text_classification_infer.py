# 安装Hugging Face库: !pip install transformers datasets

from transformers import pipeline

classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
result = classifier("I love using Hugging Face!")
print(result)
# 输出: [{'label': 'POSITIVE', 'score': 0.9997085928916931}]
