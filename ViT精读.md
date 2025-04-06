0.摘要  
尽管Transformer架构已成为自然语言处理任务的实际标准，但其在计算机视觉领域的应用仍然有限。

1.引言  
* 受Transformer架构在自然语言处理（NLP）领域规模化成功的启发，我们尝试以尽可能少的修改将标准Transformer直接应用于图像处理。
* 为此，我们将图像分割为若干图块( patches)，并将这些图块的线性嵌入(linear embeddings)序列作为输入提供给Transformer。图像块的处理方式与自然语言处理应用中的词元(tokens)（单词）完全相同。我们采用监督学习方式训练该模型进行图像分类任务。【每个patch是16 x 16，每个pathc当成一个元素，通过一个fc layer得到一个linear embedding。一句话有多少个单词，相当于一个图片有多少个patch】
* 在中等规模数据集（如ImageNet）上训练时，若未采用强正则化(strong regularization)措施，这些模型的准确率会略低于同等规模的ResNet模型，差距约为几个百分点。
* Transformer 不像 CNN 那样天生具备“局部感知(locality)”和“平移不变性(translation equivariance)”等特性【一种先验知识，或者一种我们提前做好的假设】，所以在训练数据不够多的时候，它的表现就不如 CNN 那么好。（用更直白的语言解释：CNN 的设计让它更容易学会识别图像中的局部特征，比如边缘、纹理等，而 Transformer 需要更多数据才能达到同样的效果。如果数据量不足，Transformer 的表现就会差一些。）
* 卷积网络有2种归纳偏置：locality(图片上相邻的区域会有相邻的特征)；平移等变性[ f(g(x)) = g(f(x)) ]，无论先做g函数还是f函数，结果都是不变的，f可以理解为卷积，g可以理解为平移。

3.方法  
3.1ViT
![image](https://github.com/user-attachments/assets/ade2067a-d551-4b76-af5b-32ca9aea96cd)
给定一张图，分成很多个patch。将这些patch变成一个序列。每个patch会通过一个线性投射层的操作，得到一个特征(patch embedding)。再加上position embedding位置编码。  
整个token既包含原本有的图像信息，又包含这个图像块的所在位置信息。将token输入给transformer encoder。  
根据Class embedding的输出，做最后的判断。MLP Head通用的分类头，最后用交叉熵函数去做模型的训练。  

5.结论  
* 本文的ViT是用来做分类
* 虽然初步结果令人鼓舞，但仍存在许多挑战。其中之一是如何将ViT（视觉Transformer）应用到其他计算机视觉任务中，例如目标检测和图像分割。
* 另一个挑战是继续探索自监督预训练方法。我们的初步实验表明，自监督预训练已能带来性能提升，但其效果与大规模监督式预训练相比仍存在显著差距。

