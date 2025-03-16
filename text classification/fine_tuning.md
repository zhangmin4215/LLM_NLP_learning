### 安装数据集

```!pip install transformers datasets```

```from datasets import load_dataset```

### 载入Yelp评论数据集

```dataset = load_dataset("yelp_review_full")```

<img width="850" alt="截屏2025-03-16 16 47 25" src="https://github.com/user-attachments/assets/49c1614e-3450-46c5-a72a-7874f0ad0b58" />

### 数据集可视化

```dataset["train"][0]```

<img width="1325" alt="截屏2025-03-16 16 50 50" src="https://github.com/user-attachments/assets/7e392e5c-efc8-4eed-bdc1-3d52bafb8018" />
