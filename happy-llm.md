# happy-llm

## 1. NLP基本概念

自然语言处理（NLP）是 一种让计算机理解、解释和生成人类语言的技术。

### 1.1 NLP任务

+ 中文分词：在处理中文文本时，由于中文语言的特点，词与词之间没有像英文那样的明显的分割（如空格），所有无法直接通过空格来确定词的边界。因此，中文分词成为了中文文本处理的首要步骤。

	```
	英文输入：The cat sits on the mat.
	英文切割输出：[The | cat | sits | on | the | mat]
	中文输入：今天天气真好，适合出去游玩.
	中文切割输出：["今天", "天气", "真", "好", "，", "适合", "出去", "游玩", "。"]
	```

+ 字词切分：将词汇进一步分解为更小的单位，即字词。字词切分特别适用于处理词汇稀疏问题，即当遇到罕见词或未见过的新词时，能够通过已知的字词单位来理解或生成这些词汇。

	+ 子词切分的方法有很多种，常见的有Byte Pair Encoding (BPE)、WordPiece、Unigram、SentencePiece等。这些方法的基本思想是将单词分解成更小的、频繁出现的片段，这些片段可以是单个字符、字符组合或者词根和词缀。

	```
	输入：unhappiness
	
	不使用子词切分：整个单词作为一个单位，输出：“unhappiness”
	使用子词切分（假设BPE算法）：单词被分割为：“un”、“happi”、“ness”
	
	即使模型未见过unhappiness，也能通过字词切分，形成已知的字词来理解词汇意思
	```

+ 词性标注：为每个词标注词性，更易理解文本含义，进而进行信息提取、情感分析、机器翻译等更复杂的处理。

+ 文本分类：理解文本含义和上下文，并基于此将文本映射到特定类别。文本分类任务成功的关键在于选择合适的特征表示和分类算法，以及拥有高质量的训练数据。

+ 实体识别：自动识别文本中具有特定意义的实体，并将它们分类为预定义的类别，如人名、地点、组织、日期、时间等。

+ 关系抽取： NLP 领域中的一项关键任务，它的目标是从文本中识别实体之间的语义关系。这些关系可以是因果关系、拥有关系、亲属关系、地理位置关系等，关系抽取对于理解文本内容、构建知识图谱、提升机器理解语言的能力等方面具有重要意义。

	```
	输入：比尔·盖茨是微软公司的创始人。
	
	输出：[("比尔·盖茨", "创始人", "微软公司")]
	
	```

	在这个例子中，关系抽取任务的目标是从文本中识别出“比尔·盖茨”和“微软公司”之间的“创始人”关系。通过关系抽取，我们可以从文本中提取出有用的信息，帮助计算机更好地理解文本内容，为后续的知识图谱构建、问答系统等任务提供支持。

+ 文本摘要：生成摘要，概括文本内容。

	+ 抽取式摘要：直接从文本内选取句子或短语来概括
	+ 生成式摘要：理解文本的深层含义，并能够以新的方式表达相同的信息

+ 机器翻译：为了提高机器翻译的质量，研究者不断探索新的方法和技术，如基于神经网络的Seq2Seq模型、Transformer模型等，这些模型能够学习到源语言和目标语言之间的复杂映射关系，从而实现更加准确和流畅的翻译。

+ 自动问答：旨在使计算机能够理解自然语言提出的问题，并根据给定的数据源自动提供准确的答案。

### 1.2 文本表示

+ 词向量

	```
	VSM 方法词向量：
	
	# "雍和宫的荷花很美"
	# 词汇表大小：16384，句子包含词汇：["雍和宫", "的", "荷花", "很", "美"] = 5个词
	
	vector = [0, 0, ..., 1, 0, ..., 1, 0, ..., 1, 0, ..., 1, 0, ..., 1, 0, ...]
	#                    ↑          ↑          ↑          ↑          ↑
	#      16384维中只有5个位置为1，其余16379个位置为0
	# 实际有效维度：仅5维（非零维度）
	# 稀疏率：(16384-5)/16384 ≈ 99.97%
	
	```

+ 语言模型：以前面出现词来判断下个词出现的概率。N-gram模型的核心思想是基于马尔可夫假设，即一个词的出现概率仅依赖于它前面的N-1个词。

+ Word2Vec：通过学习词与词之间的上下文关系来生成词的密集向量表示。利用词在文本中的上下文信息来捕捉词之间的语义关系，从而使得语义相似或相关的词在向量空间中距离较近。

+ ELMO：首先在大型语料库上训练语言模型，得到词向量模型，然后在特定任务上对模型进行微调，得到更适合该任务的词向量，ELMo首次将预训练思想引入到词向量的生成中，使用双向LSTM结构，能够捕捉到词汇的上下文信息，生成更加丰富和准确的词向量表示。

## 2. Transformer架构

### 2.1 注意力机制

#### 2.1.1 什么是注意力机制

由于循环神经网络（Recurrent Neutral Network，RNN）两个缺陷：

+ 序列依次计算的模式能够很好地模拟时序信息，但限制了计算机并行计算的能力。由于序列需要依次输入、依序计算，图形处理器（GPU）并行能力受到极大限制，导致以RNN为基础架构的模型虽然参数量不算特别大，但计算时间成本却很高。
+ RNN难以捕捉长序列的相关关系。在RNN架构中，距离越远的输入之间的关系就越难被捕捉，同时RNN需要将整个序列读入内存依次计算，也限制了序列的长度。虽然LSTM中通过门机制对此进行一定优化，但对于较远距离相关关系的捕捉，RNN依旧不如人意。

针对这样问题，提出了**注意力机制**，搭建了完全由注意力机制构成的神经网络——**Transformer**。

注意力机制最先源于计算机视觉领域，其核心思想为当我们关注一张图片，我们往往无需看清楚全部内容而仅将注意力集中在重点部分即可。而在自然语言处理领域，我们往往也可以通过将重点注意力集中在一个或几个 token，从而取得更高效高质的计算效果。

**注意力机制**的特点是通过计算 **Query** 与**Key**的相关性为真值加权求和，从而拟合序列中每个词同其他词的相关关系。

#### 2.1.2 深入理解注意力机制

注意力机制核心计算公式：$attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})V$

#### 2.1.3 注意力机制的实现

根据公式即可实现，我们假设输入的 q、k、v 是已经经过转化的词向量矩阵，也就是公式中的 Q、K、V。

```python
'''注意力计算函数'''
def attention(query, key, value, dropout=None):
    '''
    args:
    query: 查询值矩阵
    key: 键值矩阵
    value: 真值矩阵
    '''
    # 获取键向量的维度，键向量的维度和值向量的维度相同
    d_k = query.size(-1) 
    # 计算Q与K的内积并除以根号dk
    # transpose——相当于转置
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # Softmax
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
        # 采样
     # 根据计算结果对value进行加权求和
    return torch.matmul(p_attn, value), p_attn

```

#### 2.1.4 自注意力

所谓自注意力，即是计算本身序列中每个元素对其他元素的注意力分布，即在计算过程中，Q、K、V 都由同一个输入通过不同的参数矩阵计算得到。

通过自注意力机制，我们可以找到一段文本中每一个 token 与其他所有 token 的相关关系大小，从而建模文本之间的依赖关系。

#### 2.1.5 掩码自注意力

掩码自注意力，即 Mask Self-Attention，是指使用注意力掩码的自注意力机制。

使用注意力掩码的核心动机是让模型只能使用历史信息进行预测而不能看到未来信息。使用注意力机制的 Transformer 模型也是通过类似于 n-gram 的语言模型任务来学习的，也就是对一个文本序列，不断根据之前的 token 来预测下一个 token，直到将整个文本序列补全

+ 掩码的作用是遮蔽一些特定位置的token，模型在学习的过程中，会忽略被遮蔽的token。

例如：如果待学习的文本序列是 【BOS】I like you【EOS】，那么，模型会按如下顺序进行预测和学习：

```markup
Step 1：输入 【BOS】，输出 I
Step 2：输入 【BOS】I，输出 like
Step 3：输入 【BOS】I like，输出 you
Step 4：输入 【BOS】I like you，输出 【EOS】
```

但是，我们可以发现，上述过程是一个串行的过程，也就是需要先完成 Step 1，才能做 Step 2，接下来逐步完成整个序列的补全。

针对这个问题，Transformer 就提出了掩码自注意力的方法。掩码自注意力会生成一串掩码，来遮蔽未来信息。例如，我们待学习的文本序列仍然是 【BOS】I like you【EOS】，我们使用的注意力掩码是【MASK】，那么模型的输入为：

```markup
<BOS> 【MASK】【MASK】【MASK】【MASK】
<BOS>    I   【MASK】 【MASK】【MASK】
<BOS>    I     like  【MASK】【MASK】
<BOS>    I     like    you  【MASK】
<BOS>    I     like    you   </EOS>
```

在每一行输入中，模型仍然是只看到前面的 token，预测下一个 token。但是注意，上述输入不再是串行的过程，而可以一起并行地输入到模型中，模型只需要每一个样本根据未被遮蔽的 token 来预测下一个 token 即可，从而实现了并行的语言模型。

#### 2.1.6 多头注意力

### 2.2 Encoder-Decoder

在 Transformer 中，使用注意力机制的是其两个核心组件——Encoder（编码器）和 Decoder（解码器）。

#### 2.2.1 Seq2Seq模型

Seq2Seq，即序列到序列，是一种经典 NLP 任务。具体而言，是指模型输入的是一个自然语言序列，输出的是一个可能不等长的自然语言序列 。

机器翻译任务即是一个经典的 Seq2Seq 任务，例如，我们的输入可能是“今天天气真好”，输出是“Today is a good day.”。Transformer 是一个经典的 Seq2Seq 模型，即模型的输入为文本序列，输出为另一个文本序列。

![image-20260418215100738](C:/Users/user/AppData/Roaming/Typora/typora-user-images/image-20260418215100738.png)

对于 Seq2Seq 任务，就是编码后译码

+ 编码，就是将输入的自然语言序列通过隐藏层编码成能够表征语义的向量（或矩阵），可以简单理解为更复杂的词向量表示。
+ 解码，就是对输入的自然语言序列编码得到的向量或矩阵通过隐藏层输出，再解码成对应的自然语言目标序列

#### 2.2.2 前馈神经网络

前馈神经网络（Feed Forward Neural Network，下简称 FNN）。也就是我们在上一节提过的每一层的神经元都和上下两层的每一个神经元完全连接的网络结构。

每一个 Encoder Layer 都包含一个上文讲的注意力机制和一个前馈神经网络。

```python
class MLP(nn.Module):
    '''前馈神经网络'''
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        # 定义第一层线性变换，从输入维度到隐藏维度
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        # 定义第二层线性变换，从隐藏维度到输入维度
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        # 定义dropout层，用于防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 前向传播函数
        # 首先，输入x通过第一层线性变换和RELU激活函数
        # 最后，通过第二层线性变换和dropout层
        return self.dropout(self.w2(F.relu(self.w1(x))))
    
```

注意，Transformer 的前馈神经网络是由两个线性层中间加一个 RELU 激活函数组成的，以及前馈神经网络还加入了一个 Dropout 层来防止过拟合。Dropout 层只在训练时开启，推理/测试阶段关闭，所以许多Transformer结构示意图中不会画出该层。

#### 2.2.3 层归一化

归一化核心是为了让不同层输入的取值范围或者分布能够比较一致。

+ 由于深度神经网络中每一层的输入都是上一层的输出，因此多层传递下，对网络中较高的层，之间的所有神经层的参数变化会导致其输入的分布发生较大的改变。也就是说，随着神经网络参数的更新，各层的输出分布是不相同的，且差异会随着网络深度的增大而增大。但是，需要预测的条件分布始终是相同的，从而也就造成了预测的误差。——所以需要归一化

归一化操作

+ 批归一化
+ 层归一化

### 2.3 搭建一个Transformer

#### 2.3.1 Embedding层

 在 NLP 任务中，我们往往需要将自然语言的输入转化为机器可以处理的向量。在深度学习中，承担这个任务的组件就是 Embedding 层。

Embedding 层其实是一个存储固定大小的词典的嵌入向量查找表。

****

```
input: 我
output: 0

input: 喜欢
output: 1

input：你
output: 2
```

因此，Embedding 层的输入往往是一个形状为 （batch_size，seq_len，1）的矩阵，第一个维度是一次批处理的数量，第二个维度是自然语言序列的长度，第三个维度则是 token 经过 tokenizer 转化成的 index 值。例如，对上述输入，Embedding 层的输入会是：

```
[[[0],[1],[2]]]
```

其 batch_size 为1，seq_len 为3，转化出来的 index 如上。

#### 2.3.2 位置编码

注意力机制可以实现良好的并行计算，但同时，其注意力计算的方式也导致序列中相对位置的丢失。在 RNN、LSTM 中，输入序列会沿着语句本身的顺序被依次递归处理，因此输入序列的顺序提供了极其重要的信息，这也和自然语言的本身特性非常吻合。

```python
class PositionalEncoding(nn.Module):
    '''位置编码模块'''

    def __init__(self, args):
        super(PositionalEncoding, self).__init__()
        # Dropout 层
        # self.dropout = nn.Dropout(p=args.dropout)

        # block size 是序列的最大长度
        pe = torch.zeros(args.block_size, args.n_embd)
        position = torch.arange(0, args.block_size).unsqueeze(1)
        # 计算 theta
        div_term = torch.exp(
            torch.arange(0, args.n_embd, 2) * -(math.log(10000.0) / args.n_embd)
        )
        # 分别计算 sin、cos 结果
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # 将位置编码加到 Embedding 结果上
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return x
```

#### 2.3.3 一个完整的Transformer

![image-20260420221537552](C:/Users/user/AppData/Roaming/Typora/typora-user-images/image-20260420221537552.png)

基于之前所实现过的组件，我们实现完整的 Transformer 模型：

```python
class Transformer(nn.Module):
   '''整体模型'''
    def __init__(self, args):
        super().__init__()
        # 必须输入词表大小和 block size
        assert args.vocab_size is not None
        assert args.block_size is not None
        self.args = args
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(args.vocab_size, args.n_embd),
            wpe = PositionalEncoding(args),
            drop = nn.Dropout(args.dropout),
            encoder = Encoder(args),
            decoder = Decoder(args),
        ))
        # 最后的线性层，输入是 n_embd，输出是词表大小
        self.lm_head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

        # 初始化所有的权重
        self.apply(self._init_weights)

        # 查看所有参数的数量
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    '''统计所有参数的数量'''
    def get_num_params(self, non_embedding=False):
        # non_embedding: 是否统计 embedding 的参数
        n_params = sum(p.numel() for p in self.parameters())
        # 如果不统计 embedding 的参数，就减去
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
        return n_params

    '''初始化权重'''
    def _init_weights(self, module):
        # 线性层和 Embedding 层初始化为正则分布
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    '''前向计算函数'''
    def forward(self, idx, targets=None):
        # 输入为 idx，维度为 (batch size, sequence length, 1)；targets 为目标序列，用于计算 loss
        device = idx.device
        b, t = idx.size()
        assert t <= self.args.block_size, f"不能计算该序列，该序列长度为 {t}, 最大序列长度只有 {self.args.block_size}"

        # 通过 self.transformer
        # 首先将输入 idx 通过 Embedding 层，得到维度为 (batch size, sequence length, n_embd)
        print("idx",idx.size())
        # 通过 Embedding 层
        tok_emb = self.transformer.wte(idx)
        print("tok_emb",tok_emb.size())
        # 然后通过位置编码
        pos_emb = self.transformer.wpe(tok_emb) 
        # 再进行 Dropout
        x = self.transformer.drop(pos_emb)
        # 然后通过 Encoder
        print("x after wpe:",x.size())
        enc_out = self.transformer.encoder(x)
        print("enc_out:",enc_out.size())
        # 再通过 Decoder
        x = self.transformer.decoder(x, enc_out)
        print("x after decoder:",x.size())

        if targets is not None:
            # 训练阶段，如果我们给了 targets，就计算 loss
            # 先通过最后的 Linear 层，得到维度为 (batch size, sequence length, vocab size)
            logits = self.lm_head(x)
            # 再跟 targets 计算交叉熵
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # 推理阶段，我们只需要 logits，loss 为 None
            # 取 -1 是只取序列中的最后一个作为输出
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss
```

## 3. 预训练语言模型

### 3.1 Encoder-only PLM

针对 Encoder、Decoder 的特点，引入 ELMo 的预训练思路，开始出现不同的、对 Transformer 进行优化的思路。例如，Google 仅选择了 Encoder 层，通过将 Encoder 层进行堆叠，再提出不同的预训练任务-掩码语言模型（Masked Language Model，MLM），打造了一统自然语言理解（Natural Language Understanding，NLU）任务的代表模型——BERT。而 OpenAI 则选择了 Decoder 层，使用原有的语言模型（Language Model，LM）任务，通过不断增加模型参数和预训练语料，打造了在 NLG（Natural Language Generation，自然语言生成）任务上优势明显的 GPT 系列模型，也是现今大火的 LLM 的基座模型。当然，还有一种思路是同时保留 Encoder 与 Decoder，打造预训练的 Transformer 模型，例如由 Google 发布的 T5模型。

#### 3.1.1 BERT

BERT，全名为 Bidirectional Encoder Representations from Transformers。自 BERT 推出以来，预训练+微调的模式开始成为自然语言处理任务的主流，不仅 BERT 自身在不断更新迭代提升模型性能，也出现了如 MacBERT、BART 等基于 BERT 进行优化提升的模型。可以说，BERT 是自然语言处理的一个阶段性成果，标志着各种自然语言处理任务的重大进展以及预训练模型的统治地位建立，一直到 LLM 的诞生，NLP 领域的主导地位才从 BERT 系模型进行迁移。即使在 LLM 时代，要深入理解 LLM 与 NLP，BERT 也是无法绕过的一环。

#### （1）思想沿程

BERT 是一个统一了多种思想的预训练模型。其所沿承的核心思想包括：

+ Transformer 架构。BERT 正沿承了 Transformer 的思想，在 Transformer 的模型基座上进行优化，通过将 Encoder 结构进行堆叠，扩大模型参数，打造了在 NLU 任务上独居天分的模型架构；
+ 预训练+微调范式。同样在 2018年，ELMo 的诞生标志着预训练+微调范式的诞生。ELMo 模型基于双向 LSTM 架构，在训练数据上基于语言模型进行预训练，再针对下游任务进行微调，表现出了更加优越的性能，将 NLP 领域导向预训练+微调的研究思路。而 BERT 也采用了该范式，并通过将模型架构调整为 Transformer，引入更适合文本理解、能捕捉深层双向语义关系的预训练任务 MLM，将预训练-微调范式推向了高潮。

我们将从模型架构、预训练任务以及下游任务微调三个方面深入剖析 BERT，分析 BERT 的核心思路及优势，帮助大家理解 BERT 为何能够具备远超之前模型的性能，也从而更加深刻地理解 LLM 如何能够战胜 BERT 揭开新时代的大幕。

#### （2）模型架构——Encoder only

BERT 的模型架构是取了 Transformer 的 Encoder 部分堆叠而成，其主要结构如图3.1所示：

![image-20260422212154668](C:/Users/user/AppData/Roaming/Typora/typora-user-images/image-20260422212154668.png)

BERT 是针对于 NLU 任务打造的预训练模型，其输入一般是文本序列，而输出一般是 Label，例如情感分类的积极、消极 Label。但是，正如 Transformer 是一个 Seq2Seq 模型，使用 Encoder 堆叠而成的 BERT 本质上也是一个 Seq2Seq 模型，只是没有加入对特定任务的 Decoder，因此，为适配各种 NLU 任务，在模型的最顶层加入了一个分类头 prediction_heads，用于将多维度的隐藏状态通过线性层转换到分类维度（例如，如果一共有两个类别，prediction_heads 输出的就是两维向量）。

模型整体既是由 Embedding、Encoder 加上 prediction_heads 组成：

![image-20260422212430420](C:/Users/user/AppData/Roaming/Typora/typora-user-images/image-20260422212430420.png)

输入的文本序列会首先通过tokenizer（分组词）转化成 input_ids，然后进入 Embedding 层转化为特定维度的 hidden_states，再经过 Encoder 块。Encoder 块中是堆叠起来的 N 层 Encoder Layer，BERT 有两种规模的模型，分别是 base 版本，large 版本。通过Encoder 编码之后的最顶层 hidden_states 最后经过 prediction_heads 就得到了最后的类别概率，经过 Softmax 计算就可以计算出模型预测的类别。

BERT 采用 WordPiece 作为分词方法。WordPiece 是一种基于统计的子词切分算法，其核心在于将单词拆解为子词（例如，"playing" -> ["play", "##ing"]）。其合并操作的依据是最大化语言模型的似然度。对于中文等非空格分隔的语言，通常将单个汉字作为原子分词单位（token）处理。

prediction_heads 其实就是线性层加上激活函数，一般而言，最后一个线性层的输出维度和任务的类别数相等，如图3.3所示：

![image-20260422214516718](C:/Users/user/AppData/Roaming/Typora/typora-user-images/image-20260422214516718.png)

而每一层 Encoder Layer 都是和 Transformer 中的 Encoder Layer 结构类似的层，如图3.4所示：

![image-20260422214530007](C:/Users/user/AppData/Roaming/Typora/typora-user-images/image-20260422214530007.png)

如图3.5所示，已经通过 Embedding 层映射的 hidden_states 进入核心的 attention 机制，然后通过残差连接的机制和原输入相加，再经过一层 Intermediate 层得到最终输出。Intermediate 层是 BERT 的特殊称呼，其实就是一个线性层加上激活函数：

![image-20260422214540898](C:/Users/user/AppData/Roaming/Typora/typora-user-images/image-20260422214540898.png)

### 3.3 Decoder-Only PLM

Decoder-only PLM只使用Decoder堆叠而成的模型

事实上，Decoder-Only就是目前大火的LLM的基础架构，目前所有的LLM基本都是Decoder-only模型（RWKV、Mamba 等非 Transformer 架构除外）。ChatGpt就是Decoder-only系列的代表模型。

#### 3.3.1 GPT

##### 模型架构——Decoder-Only

1. 对于一个自然语言文本的输入，先通过 tokenizer 进行分词并转化为对应词典序号的 input_ids。

2. 输入的 input_ids 首先通过 Embedding 层，再经过 Positional Embedding 进行位置编码。不同于 BERT 选择了可训练的全连接层作为位置编码，GPT 沿用了 Transformer 的经典 Sinusoidal 位置编码，即通过三角函数进行绝对位置编码，可以参考Transformer 模型细节的解析。
3. 通过 Embedding 层和 Positional Embedding 层编码成 hidden_states 之后，就可以进入到解码器（Decoder），第一代 GPT 模型和原始 Transformer 模型类似，选择了 12层解码器层，但是在解码器层的内部，相较于 Transformer 原始 Decoder 层的双注意力层设计，GPT 的 Decoder 层反而更像 Encoder 层一点。由于不再有 Encoder 的编码输入，Decoder 层仅保留了一个带掩码的注意力层，并且将 LayerNorm 层从 Transformer 的注意力层之后提到了注意力层之前。hidden_states 输入 Decoder 层之后，会先进行 LayerNorm，再进行掩码注意力计算，然后经过残差连接和再一次 LayerNorm 进入到 MLP 中并得到最后输出。

##### 预训练任务——CLM

CLM 可以看作 N-gram 语言模型的一个直接扩展。N-gram 语言模型是基于前 N 个 token 来预测下一个 token，CLM 则是基于一个自然语言序列的前面所有 token 来预测下一个 token，通过不断重复该过程来实现目标文本序列的生成。也就是说，CLM 是一个经典的补全形式。例如，CLM 的输入和输出可以是：

```markup
input: 今天天气
output: 今天天气很

input: 今天天气很
output：今天天气很好
```

#### 3.3.2 LLaMA

与GPT类似，LLaMA模型的处理流程也始于将输入文本通过tokenizer进行编码，转化为一系列的input_ids。这些input_ids是模型能够理解和处理的数据格式。接下来，这些input_ids会经过embedding层的转换，这里每个input_id会被映射到一个高维空间中的向量，即词向量。同时，输入文本的位置信息也会通过positional embedding层被编码，以确保模型能够理解词序上下文信息。

这样，input_ids经过embedding层和positional embedding层的结合，形成了hidden_states。hidden_states包含了输入文本的语义和位置信息，是模型进行后续处理的基础，hidden_states随后被输入到模型的decoder层。

在decoder层中，hidden_states会经历一系列的处理，这些处理由多个decoder block组成。每个decoder block都是模型的核心组成部分，它们负责对hidden_states进行深入的分析和转换。在每个decoder block内部，首先是一个masked self-attention层。在这个层中，模型会分别计算query、key和value这三个向量。这些向量是通过hidden_states线性变换得到的，它们是计算注意力权重的基础。然后使用softmax函数计算attention score，这个分数反映了不同位置之间的关联强度。通过attention score，模型能够确定在生成当前词时，应该给予不同位置的hidden_states多大的关注。然后，模型将value向量与attention score相乘，得到加权后的value，这就是attention的结果。

在完成masked self-attention层之后，hidden_states会进入MLP层。在这个多层感知机层中，模型通过两个全连接层对hidden_states进行进一步的特征提取。第一个全连接层将hidden_states映射到一个中间维度，然后通过激活函数进行非线性变换，增加模型的非线性能力。第二个全连接层则将特征再次映射回原始的hidden_states维度。

最后，经过多个decoder block的处理，hidden_states会通过一个线性层进行最终的映射，这个线性层的输出维度与词表维度相同。这样，模型就可以根据hidden_states生成目标序列的概率分布，进而通过采样或贪婪解码等方法，生成最终的输出序列。这一过程体现了LLaMA模型强大的序列生成能力。

#### 3.3.3 GLM

GLM 的核心创新点主要在于其提出的 GLM（General Language Model，通用语言模型）任务，这也是 GLM 的名字由来。GLM 是一种结合了自编码思想和自回归思想的预训练方法。所谓自编码思想，其实也就是 MLM 的任务学习思路，在输入文本中随机删除连续的 tokens，要求模型学习被删除的 tokens；所谓自回归思想，其实就是传统的 CLM 任务学习思路，也就是要求模型按顺序重建连续 tokens。

GLM 通过优化一个自回归空白填充任务来实现 MLM 与 CLM 思想的结合。其核心思想是，对于一个输入序列，会类似于 MLM 一样进行随机的掩码，但遮蔽的不是和 MLM 一样的单个 token，而是每次遮蔽一连串 token；模型在学习时，既需要使用遮蔽部分的上下文预测遮蔽部分，在遮蔽部分内部又需要以 CLM 的方式完成被遮蔽的 tokens 的预测。例如，输入和输出可能是：

```markup
输入：I <MASK> because you <MASK>
输出：<MASK> - love you; <MASK> - are a wonderful person
```
