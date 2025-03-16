### 安装数据集

```!pip install transformers datasets```

```from datasets import load_dataset```

### 载入Yelp评论数据集

```dataset = load_dataset("yelp_review_full")```

<img width="850" alt="截屏2025-03-16 16 47 25" src="https://github.com/user-attachments/assets/49c1614e-3450-46c5-a72a-7874f0ad0b58" />

### 数据集可视化

```dataset["train"][0]```

<img width="1325" alt="截屏2025-03-16 16 50 50" src="https://github.com/user-attachments/assets/7e392e5c-efc8-4eed-bdc1-3d52bafb8018" />
 
 ### 创建一个分词器来处理文本，并包含填充和截断策略来处理任何可变的序列长度。

 ```from transformers import AutoTokenizer```

### 导入预训练模型

 ```tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")```
 
 <img width="728" alt="截屏2025-03-16 18 02 03" src="https://github.com/user-attachments/assets/a2a5a22c-43d1-4087-8fa8-8ac212d4deb2" />

### 预处理函数

 ```def tokenize_function(examples): return tokenizer(examples["text"], padding="max_length", truncation=True)```

 ### 使用Datasets的map方法将预处理函数应用于整个数据集

 ```tokenized_datasets = dataset.map(tokenize_function, batched=True)```

 <img width="725" alt="截屏2025-03-16 18 02 40" src="https://github.com/user-attachments/assets/a57f8b3e-0f4b-4fc6-aa61-f24b325a8e92" />

### 创建一个较小的完整数据集子集来进行微调，以减少所需时间

```small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))```

```small_test_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))```
