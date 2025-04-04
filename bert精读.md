题目：用于语言理解的深度双向Transformer预训练

0.摘要  
* BERT通过同时利用左右两侧的上下文信息，学习深度的双向语言表示。
* 预训练的 BERT 模型只需添加一个额外的输出层进行微调，就可以为广泛的任务（如问答(question answering)和语言推理(language inference)）创建最先进的模型，而无需对任务特定的架构进行大量修改。

1.介绍  
* 语言模型预训练已被证明对改进许多自然语言处理任务非常有效。
* 句子级别(sentence-level)任务：关注句子之间的整体关系（如推理和复述）。
* 词元级别(token-level)任务：需要模型对每个词元进行细粒度的处理（如命名实体识别和问答）。
* 将预训练语言表示应用于下游任务有两种现有策略：基于特征的方法(feature-based)和微调方法(fine-tuning)。基于特征的方法：将预训练模型作为特征提取器，提取的特征用于下游任务。微调方法：在预训练模型的基础上，通过微调模型参数来适应下游任务。
* 基于特征的方法，例如ELMo（Peters等，2018a），使用任务特定的架构，将预训练表示作为额外的特征包含其中。
* 微调方法，例如生成式预训练Transformer（OpenAI GPT）（Radford等，2018），引入了最少的任务特定参数，并通过简单地微调所有预训练参数来在下游任务上进行训练。
* 这两种方法在预训练期间共享相同的目标函数，即使用单向语言模型来学习通用的语言表示。这句话指出，尽管基于特征的方法（如ELMo）和微调方法（如GPT）在下游任务中的应用方式不同，但它们在预训练阶段的目标是一致的：通过单向语言模型学习通用的语言表示。
* 主要限制在于标准语言模型是单向的，这限制了预训练期间可用的架构选择。例如，在OpenAI GPT中，作者使用了从左到右的架构，其中每个词元在Transformer的自注意力层中只能关注前面的词元。

* BERT通过使用‘掩码语言模型’（MLM）预训练目标来缓解前面提到的单向性限制，这一方法受到了完形填空任务（Cloze task）的启发。掩码语言模型（MLM）：通过随机掩码部分token并预测它们，BERT能够同时利用左右两侧的上下文信息。
* 除了掩码语言模型外，我们还使用了一个‘下一句预测(next sentence prediction)’任务，以联合预训练文本对(text-pair representations)的表示。这句话指出了BERT预训练的另一个关键组件：下一句预测任务：该任务通过预测两个句子是否连续，帮助模型学习句子之间的关系。联合预训练：MLM和下一句预测任务共同训练，使BERT能够同时学习词元和句子级别的表示。

贡献：
* 证明了双向预训练对语言表示的重要性。
* 证明了预训练的表示(pre-trained representations)减少了对许多高度工程化的任务特定架构的需求。

2.相关工作  
2.1非监督基于feature的方法  
ELMo  
2.2非监督微调的方法  
GPT  
2.3 来自监督数据的迁移学习  

3 BERT  
* 框架中有2个步骤：pre-training 和 fine-tuning
* 在预训练期间，模型通过不同的预训练任务在未标注的数据上进行训练。多种预训练任务：例如掩码语言模型（MLM）和下一句预测任务，帮助模型学习通用的语言表示。
* 对于微调，BERT模型首先使用预训练的参数进行初始化，然后使用下游任务的标注数据对所有参数进行微调。

模型架构  
* BERT的模型架构是一个多层的双向Transformer编码器，基于Vaswani等人（2017）提出的原始实现，并在tensor2tensor库中发布。
* 在这项工作中，我们将层数（即Transformer块的数量）表示为L，隐藏层大小表示为H，自注意力头的数量表示为A。
* transformer的可学习参数来源：嵌入层+transformer块

输入/输出Representations  
* 为了使BERT能够处理各种下游任务，输入表示(input representation)能够明确地在一个词元序列(token sequence)中表示单个句子(single sentence)或句子对（例如，⟨问题，答案⟩）。”明确表示：通过特殊的标记和分隔符，BERT能够清晰地区分句子或句子对的内容。
* sentence：在BERT中，句子不限于语言学上的句子，而是指任意连续的文本片段。
* 序列：BERT的输入序列可以是一个句子，也可以是两个句子（如问题和答案）的组合。
* WordPiece嵌入：BERT使用WordPiece分词方法，词表大小为30,000个词元(token)。
* [CLS]标记：每个输入序列的开头添加一个特殊的[CLS]标记，其对应的最终隐藏状态用于分类任务的整体序列表示。
* 句子对(Sentence pairs)被组合成一个单独的序列。我们通过两种方式区分句子：首先，用一个特殊标记（[SEP]）将它们分隔开；其次，为每个词元添加一个可学习的嵌入，用于指示它属于句子A还是句子B。
* 对于给定的词元，其输入表示是通过将对应的词元嵌入、段嵌入和位置嵌入(token, segment, and position embeddings)相加而构建的。

3.1 预训练bert  
(1)掩码的语言模型Masked LM  
* BERT的掩码策略：随机掩码15%的词元：在每个序列中，随机选择15%的词元进行掩码。仅预测掩码词元：BERT的目标是预测被掩码的词元，而不是重建整个输入序列。
* BERT掩码策略的一个潜在问题：预训练与微调的不匹配：在预训练阶段，模型会看到[MASK]标记并学习预测它们，但在微调阶段，输入数据中不会出现[MASK]标记，这可能导致模型在微调时的表现受到影响。
* 训练数据生成器随机选择15%的词元位置进行预测。如果选择了第i个词元，我们将其替换为：(1) 80%的概率替换为[MASK]标记；(2) 10%的概率替换为随机词元；(3) 10%的概率保持不变。

（2）Next Sentence Prediction (NSP)  
* 在为每个预训练样本选择句子A和B时，50%的概率B是A的实际下一句（标记为IsNext），50%的概率B是从语料库中随机选择的句子（标记为NotNext）。这句话解释了BERT的下一句预测任务的数据生成方式。

预训练数据  
BooksCorpus (800M words)  
English Wikipedia (2,500M words)  

3.2 微调bert  
* BERT在下游任务中的应用方式：任务特定输入输出：根据任务需求调整输入和输出的格式。端到端微调：所有参数（包括预训练参数）都会在微调过程中更新，以适配具体任务。
* BERT输入表示在不同任务中的通用性：复述任务：句子A和句子B分别对应需要复述的句子对。蕴含任务：句子A和句子B分别对应假设和前提。问答任务：句子A和句子B分别对应问题和相关段落。文本分类或序列标注：句子A是输入文本，句子B为空（∅）。
* BERT在不同任务中的输出处理方式：词元级别任务：使用每个词元的表示作为输出，适用于序列标注或问答任务。分类任务：使用[CLS]标记的表示作为输出，适用于蕴含、情感分析等任务。
