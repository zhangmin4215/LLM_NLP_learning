### 通过生成式预训练提升语言理解能力(GPT)

0.摘要  
* 生成式预训练（如GPT）的核心思想：生成式预训练：在未标注文本上训练语言模型，学习通用的语言表示。判别式微调：在特定任务上微调模型，使其适应具体任务需求。

1.引言  
* 预训练词嵌入(word embeddings)在自然语言处理（NLP）领域的重要性：预训练词嵌入：如Word2Vec、GloVe等，通过大规模未标注数据训练得到。性能提升：这些词嵌入被广泛应用于各种NLP任务，显著提高了模型的表现。
* 从未标注文本(unlabeled text)中学习更高级别信息的难点：优化目标的选择：如何设计有效的目标函数来学习对迁移任务有帮助的文本表示。信息利用的复杂性：词级别以上的信息（如句子、段落）更难捕捉和利用。
* 在本文中，我们探索了一种结合无监督预训练与有监督微调的半监督方法，用于语言理解任务。

3.框架  
我们的训练流程包含两个阶段：第一阶段是在大规模文本语料上训练一个高容量语言模型；第二阶段是微调，利用标注数据使模型适应判别式任务。  

3.1 在没有标号的数据上做预训练  
给定一个无监督的词元语料库 U={u1,...,un}，我们采用标准的语言建模目标，最大化以下似然函数：  
![image](https://github.com/user-attachments/assets/00003004-c819-4209-965b-06a142873467)
语言模型就是要预测第i个词出现的概率。给定k个词、给定模型fai，预测这k个词下一个词的概率

在实验中，我们采用基于多层Transformer解码器的语言模型（该架构是Transformer的一种变体）。该模型首先对输入上下文词元(input context tokens)执行多头自注意力运算，再通过逐位置前馈层(position-wise feedforward layers)处理，最终生成目标词元的输出概率分布。
【transformer有一个编码器和一个解码器，编码器对第i个元素抽特征时能看到整个序列里所有的元素；解码器有掩码的存在，在对第i个元素抽特征时，只能看到当前元素和它之前的元素，它后面的元素通过一个掩码使得在计算注意力机制的时候变为0】
![image](https://github.com/user-attachments/assets/c28b20ae-4d2e-411a-87db-cdb6f7598c1e)
如果要预测u这个词的概率P(u)，把这个词前面的词拿出来U = (u−k ,...,u−1) 。  
UWe: 做词嵌入的投影；+Wp：再加上一个位置信息的编码，得到第一层的输入  
接下来要做n层transformer块，每一层把上一层的输出拿进来然后得出输出。transformer块不会改变输出输出的形状  

* BERT和GPT的区别
![image](https://github.com/user-attachments/assets/ee87edff-366d-477d-93e7-a132b9bc11b6)

3.2监督微调  
假设存在标注数据集C，其中每个样本包含输入词元序列(x₁,...,xₘ)及对应标签y。输入序列经由预训练模型处理后，获取最终Transformer模块的激活输出hₗᵐ，该输出将被馈入新增的线性输出层（参数为W_y）进行标签y的预测。  
![image](https://github.com/user-attachments/assets/fbf4d343-066a-41db-ab46-10b06c1eecad)

这使我们得到以下需要最大化的目标函数：
![image](https://github.com/user-attachments/assets/98fe57ef-d060-4a74-a736-ef5d5a9cc094)

微调时有2个目标函数：  
![image](https://github.com/user-attachments/assets/c98c0064-32db-4064-9d31-63ea46c8a2a2)

第一个目标函数：给你这些序列，然后预测序列的下一个词；第二个目标函数：给你完整的序列，让你预测序列对应的标号

* GPT在不同任务上的微调
![image](https://github.com/user-attachments/assets/358c787f-0855-4dbd-a2a4-096d8f68f2da)

将所有结构化输入均转换为词元序列(token sequences)，经由预训练模型处理后，接入一个线性层+softmax层的组合结构。

4.实验  
* 使用BooksCorpus数据集，7000篇没有被发表的书
* 12层transformer解码器，每一层维度是768


