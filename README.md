## nlp-guide

<p align="center"><img width="120" src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/11/TensorFlowLogo.svg/2000px-TensorFlowLogo.svg.png" /> <img width="120" src="https://media-thumbs.golden.com/OLqzmrmwAzY1P7Sl29k2T9WjJdM=/200x200/smart/golden-storage-production.s3.amazonaws.com/topic_images/e08914afa10a4179893eeb07cb5e4713.png" /></p>

`nlp-guide` is a tutorial for who is studying NLP(Natural Language Processing) using **TensorFlow** and **Pytorch**.
This repository heavily refers [graykode's nlp-tutorial](https://github.com/graykode/nlp-tutorial)

- You can implement applications with data in [data folder](https://github.com/bzantium/nlp-guide/tree/master/data)

## Curriculum


#### 0. FFNN(Feed Forward Neural Network) with one-hot BoW
- 0-1. [FFNN](https://github.com/bzantium/nlp-guide/tree/master/0-1.FFNN) - **Binary Theme Classification**
  - Colab - [FFNN_Tensor.ipynb](https://colab.research.google.com/github/bzantium/nlp-guide/blob/master/0-1.FFNN/FFNN-Tensor.ipynb), [FFNN_Torch.ipynb](https://colab.research.google.com/github/bzantium/nlp-guide/blob/master/0-1.FFNN/FFNN-Torch.ipynb)

#### 1. Basic Embedding Model

- 1-1. [Word2Vec(Skip-gram)](https://github.com/bzantium/nlp-guide/tree/master/1-1.Word2Vec) - **Embedding Words and Show Graph**
  - Paper - [Distributed Representations of Words and Phrases
    and their Compositionality(2013)](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
  - Colab - [Word2Vec_Tensor.ipynb](https://colab.research.google.com/github/bzantium/nlp-guide/blob/master/1-1.Word2Vec/Word2Vec-Tensor.ipynb), [Word2Vec_Torch.ipynb](https://colab.research.google.com/github/bzantium/nlp-guide/blob/master/1-1.Word2Vec/Word2Vec-Torch.ipynb)
- 1-2. [Doc2Vec(Application Level)](https://github.com/bzantium/nlp-guide/tree/master/1-2.Doc2Vec) - **Sentence Classification**
  - Paper - [Distributed Representations of Sentences and Documents(2014)](https://arxiv.org/pdf/1405.4053.pdf)
  - Colab - [Doc2Vec_Tensor.ipynb](https://colab.research.google.com/github/bzantium/nlp-guide/blob/master/1-2.Doc2Vec/Doc2Vec-Tensor.ipynb), [Doc2Vec_Torch.ipynb](https://colab.research.google.com/github/bzantium/nlp-guide/blob/master/1-2.Doc2Vec/Doc2Vec-Torch.ipynb)



#### 2. RNN(Recurrent Neural Network)

- 2-1. [TextRNN](https://github.com/bzantium/nlp-guide/tree/master/2-1.TextRNN) - **Predict Next Step**
  - Paper - [Finding Structure in Time(1990)](http://psych.colorado.edu/~kimlab/Elman1990.pdf)
  - Colab - [TextRNN_Tensor.ipynb](https://colab.research.google.com/github/bzantium/nlp-guide/blob/master/2-1.TextRNN/TextRNN-Tensor.ipynb), [TextRNN_Torch.ipynb](https://colab.research.google.com/github/bzantium/nlp-guide/blob/master/2-1.TextRNN/TextRNN-Torch.ipynb)
- 2-2. [TextLSTM](https://github.com/bzantium/nlp-guide/tree/master/2-2.TextLSTM) - **Autocomplete**
  - Paper - [LONG SHORT-TERM MEMORY(1997)](https://www.bioinf.jku.at/publications/older/2604.pdf)
  - Colab - [TextLSTM_Tensor.ipynb](https://colab.research.google.com/github/bzantium/nlp-guide/blob/master/2-2.TextLSTM/TextLSTM-Tensor.ipynb), [TextLSTM_Torch.ipynb](https://colab.research.google.com/github/bzantium/nlp-guide/blob/master/2-2.TextLSTM/TextLSTM-Torch.ipynb)
  - Application(Colab) - [SamHangSi_Tensor.ipynb](https://colab.research.google.com/github/bzantium/nlp-guide/blob/master/2-2.TextLSTM/SamHangSi-Tensor.ipynb)
- 2-3. [biLSTM](https://github.com/bzantium/nlp-guide/tree/master/2-3.biLSTM) - **Binary Sentiment Classification**
  - Colab - [biLSTM_Tensor.ipynb](https://colab.research.google.com/github/bzantium/nlp-guide/blob/master/2-3.biLSTM/biLSTM-Tensor.ipynb), [biLSTM_Torch.ipynb](https://colab.research.google.com/github/bzantium/nlp-guide/blob/master/2-3.biLSTM/biLSTM-Torch.ipynb)


#### 3. Attention Mechanism

- 3-1. [Seq2Seq](https://github.com/bzantium/nlp-guide/tree/master/3-1.Seq2Seq) - **Change Word**
  - Paper - [Learning Phrase Representations using RNN Encoder–Decoder
    for Statistical Machine Translation(2014)](https://arxiv.org/pdf/1406.1078.pdf)
  - Colab - [Seq2Seq_Tensor.ipynb](https://colab.research.google.com/github/bzantium/nlp-guide/blob/master/3-1.Seq2Seq/Seq2Seq-Tensor.ipynb), [Seq2Seq_Torch.ipynb](https://colab.research.google.com/github/bzantium/nlp-guide/blob/master/3-1.Seq2Seq/Seq2Seq-Torch.ipynb)
  - Application(Colab) - [Conversation_Tensor.ipynb](https://colab.research.google.com/github/bzantium/nlp-guide/blob/master/3-1.Seq2Seq/Conversation-Tensor.ipynb)
- 3-2. [Seq2Seq with Attention](https://github.com/bzantium/nlp-guide/tree/master/3-2.Seq2Seq(Attention)) - **Translate**
  - Paper - [Neural Machine Translation by Jointly Learning to Align and Translate(2014)](https://arxiv.org/abs/1409.0473)
  - Colab - [Seq2Seq(Attention)_Tensor.ipynb](https://colab.research.google.com/github/bzantium/nlp-guide/blob/master/3-2.Seq2Seq(Attention)/Seq2Seq(Attention)-Tensor.ipynb), [Seq2Seq(Attention)_Torch.ipynb](https://colab.research.google.com/github/bzantium/nlp-guide/blob/master/3-2.Seq2Seq(Attention)/Seq2Seq(Attention)-Torch.ipynb)



#### 4. CNN(Convolutional Neural Network)

- 4-1. [TextCNN](https://github.com/bzantium/nlp-guide/tree/master/4-1.TextCNN) - **Binary Sentiment Classification**
  - Paper - [Convolutional Neural Networks for Sentence Classification(2014)](http://www.aclweb.org/anthology/D14-1181)
  - Colab - [TextCNN_Tensor.ipynb](https://colab.research.google.com/github/bzantium/nlp-guide/blob/master/4-1.TextCNN/TextCNN-Tensor.ipynb), [TextCNN_Torch.ipynb](https://colab.research.google.com/github/bzantium/nlp-guide/blob/master/4-1.TextCNN/TextCNN-Torch.ipynb)
  - Application(Colab) - [Sentiment_Tensor.ipynb](https://colab.research.google.com/github/bzantium/nlp-guide/blob/master/4-1.TextCNN/Sentiment-Tensor.ipynb)
  


#### 5. Model based on Transformer

- 5-1.  [The Transformer](https://github.com/bzantium/nlp-guide/tree/master/5-1.Transformer) - **Translate**
  - Paper - [Attention Is All You Need(2017)](https://arxiv.org/abs/1810.04805)
  - Colab - [Transformer_Torch.ipynb](https://colab.research.google.com/github/bzantium/nlp-guide/blob/master/5-1.Transformer/Transformer-Torch.ipynb), [Transformer(Greedy_decoder)_Torch.ipynb](https://colab.research.google.com/github/bzantium/nlp-guide/blob/master/5-1.Transformer/Transformer(Greedy_decoder)-Torch.ipynb)
- 5-2. [BERT](https://github.com/bzantium/nlp-guide/tree/master/5-2.BERT) - **Sentiment Analysis / Classification  Next Sentence**
  - Paper - [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding(2018)](https://arxiv.org/abs/1810.04805)
  - Colab - [BERT_Torch.ipynb](https://colab.research.google.com/github/bzantium/nlp-guide/blob/master/5-2.BERT/BERT-Torch.ipynb)
- 5-3. [OpenAI GPT-2](https://github.com/bzantium/nlp-guide/tree/master/5-3.GPT-2) - **Sample model-written texts**
  - Paper - [Language Models are Unsupervised Multitask Learners(2019)](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
  - Colab - [GPT2_Torch.ipynb](https://colab.research.google.com/github/bzantium/nlp-guide/blob/master/5-3.GPT-2/GPT2-Torch.ipynb)



## Dependencies

- Python 3.5+
- Tensorflow 1.12.0+
- Pytorch 0.4.1+



## Author

- Minho Ryu @bzantium
- Author Email : ryumin93@gmail.com
