0.摘要  

* Transformer仅仅依赖于注意力机制（attention mechanisms），进行序列转录，完全不用CNN和RNN
* 应用于机器翻译任务（English-to-German translation task/English-to-French translation task）

1.介绍  

* 循环模型通常沿着输入和输出序列的符号位置进行分解计算。
* 这种固有的顺序特性阻碍了在训练样本内部的并行化处理，这在处理较长的序列时尤为关键，因为内存的限制会制约跨样本的批处理能力。
* Transformer模型能够实现显著更高的并行化程度  

2.背景  

* 减少顺序计算的目标同样构成了扩展神经GPU、ByteNet以及ConvS2S的基石，这些模型均采用卷积神经网络作为基础构建模块，能够在所有输入和输出位置上并行计算隐藏表示。
* 这种情况使得学习远距离位置之间的依赖关系变得更加困难
* 多头注意力机制可以模拟CNN多输出通道的效果
* 端到端记忆网络基于一种循环注意力机制，而非序列对齐的循环结构，并已在简单语言问答和语言建模任务中展现出优异的性能。  

3.模型架构  

* 大多数竞争性的神经序列转换模型都具有编码器-解码器结构。编码器将输入符号表示序列（x1,...,xn）映射为一个连续表示序列 z = (z1,...,zn)。
* 给定 z，解码器随后逐个元素地生成符号的输出序列（y1,...,ym）。在每一步中模型都是自回归的(auto-regressive)，在生成下一个符号时，会将先前生成的符号作为额外输入。
* Transformer 遵循这一整体架构，对编码器和解码器均采用了堆叠的自注意力机制和逐点全连接层(self-attention and point-wise)。
* embedding可以将一个一个词转化为一个一个向量  

3.1编码器和解码器stacks  
（1）编码器6个  
 
* 每个子层的输出是 LayerNorm(x + Sublayer(x))，其中 Sublayer(x) 是子层自身实现的函数。
* 为了支持这些残差连接，模型中的所有子层(sub-layers)以及嵌入层(embedding layers)均生成维度为 dmodel = 512 的输出。  

【输入为二维时，每一行是一个样本，每一列是一个特征。batch norm将每个feature变为均值为0，方差为1:
![image](https://github.com/user-attachments/assets/72af2706-a8f5-4951-bbc1-053ac307599a)

Layer norm将每个样本变为均值为0，方差为1  
![image](https://github.com/user-attachments/assets/33793651-fdcb-45ae-9059-6e9d9c3d769c)

输入为三维时：
![image](https://github.com/user-attachments/assets/035e3e7c-05d7-447a-9928-3338ca1db4f9)
】

（2）解码器6个  

* 修改了解码器堆栈中的自注意力子层，以防止位置关注到后续位置。这种掩码机制(masking)，加上output embeddings向右偏移一个位置的事实，确保了对位置 i 的预测只能依赖于位置小于 i 的已知输出。  

3.2注意力机制  

* 注意力函数可以描述为将查询(query)和一组键值对(key-value pairs)映射到一个输出，其中查询、键、值和输出都是向量。  
* 输出是通过对values进行加权求和来计算的，其中分配给每个value的权重由query与相应key的兼容性函数(compatibility function)计算得出。  

3.2.1缩放点乘/内积注意力(Scaled Dot-Product Attention)
【
![image](https://github.com/user-attachments/assets/f92968b6-9b37-40f6-b54a-d000f9bb2ece)

![image](https://github.com/user-attachments/assets/17f288f8-e19e-4f5d-8c0d-926c7e78d173)

![image](https://github.com/user-attachments/assets/b5a6e6db-9ca4-49e2-894e-9c1c4c83f3ab)
】

![image](https://github.com/user-attachments/assets/343ca1f3-f44b-4c69-b85b-a9b2ec2f68c9)

* 输入由维度为 dk 的query和key，以及维度为 dv 的value组成。

![image](https://github.com/user-attachments/assets/f6300ad0-6394-4a1c-89a6-15075350c3af)

3.2.2 多头注意力机制

* 与其使用 dmodel 维度的key、value和query执行单一的注意力函数，不如将query、key和value分别通过不同的可学习线性投影 h 次到 dk、dk 和 dv 维度，这样做更有益处。
* 在这些投影后的query、key和value的每个版本上，并行执行注意力函数，生成 dv 维度的输出值。
* “多头注意力机制使得模型能够同时关注来自不同位置的不同表示子空间的信息。”
* “由于每个head的维度减少，总计算成本与全维度的单头注意力机制相似。(8个头每个头维度64)

3.2.3 模型中的注意力机制应用

* 在‘编码器-解码器注意力’层中，query来自前一个decoder层，而记忆value和key来自encoder的输出。这使得decoder中的每个位置都可以关注输入序列中的所有位置。这模仿了sequence-to-sequence模型中典型的encoder-decoder注意力机制。
* encoder包含self-attention层。在self-attention层中，所有的key、value和query都来自同一个地方，即encoder中前一层的输出。encoder中的每个位置都可以关注encoder前一层的所有位置。
* decoder中的self-attention层允许decoder中的每个位置关注decoder中该位置及其之前的所有位置。为了防止解码器中的信息向左流动以保持自回归特性，我们在缩放点积注意力机制中通过掩码（将 softmax 输入中对应非法连接的值设置为 −∞）来实现。

3.3 Position-wise Feed-Forward Networks(MLP, 多层感知机)

tranformer的大概结构

* “除了注意力子层外，我们的编码器和解码器中的每一层还包含一个全连接的前馈神经网络，该网络分别且相同地应用于每个位置。它由两个线性变换组成，中间通过 ReLU 激活函数连接。
* “尽管线性变换在不同位置上是相同的，但它们在层与层之间使用不同的参数。另一种描述方式是将其视为两个核大小为 1 的卷积。输入和输出的维度为 dmodel = 512，而中间层的维度为 df f = 2048。”

3.4 Embeddings and Softmax
	
 * “与其他序列转换模型类似，我们使用可学习的embedding将input tokens和output tokens转换为维度为 dmodel 的向量。我们还使用常规的可学习线性变换和 softmax 函数将decoder输出转换为预测的next-token的概率。
 * 在两个embedding和 softmax 前的线性变换之间共享相同的权重矩阵

3.5 Positional Encoding  
Attention没有时序信息，输出是value的加权和，权重是query和key之间的距离

* 模型不包含循环或卷积结构，为了让模型能够利用序列的顺序信息，必须注入一些关于序列中标记的相对或绝对位置的信息。
* 在encoder和decoder堆栈的底部向input embeddings中添加了‘位置编码’。位置编码的维度与嵌入相同，均为 dmodel，因此两者可以相加。”

7.总结
在这项工作中，我们提出了Transformer，这是第一个完全基于注意力机制的序列转换模型，它用多头自注意力机制（multi-headed self-attention）取代了编码器-解码器架构中最常用的循环层。
