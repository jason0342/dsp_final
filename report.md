# Digital Speech Processing Final Project Report
Deep Speech 2 and A Simplified Adaption to the Homework 2 Recognition Task  
Author: B03901052 王傑生  
Instructor: Prof. Lin-shan Lee

### I. Introduction
Traditional HMM approach has shown decent performance in speech recognition, while recently, deep learning has become more and more popular, and people started to wonder whether better result can be achieved with deep learning. In chapter 9 in the lecture, 2 different approaches, tendem and hybrid system, are introduced to show how neural networks can be combined with the traditional HMM in speech recognition. However, these methods are still mainly the traditional HMM model with some submodel supported or replaced by neural networks.

Today, some researches shows that it is possible to do the task end-to-end with neural networks and have performance not to far from the traditional HMM method. Several difficulties have to be encountered to fully get rid of the HMM model, for example, dealing with variable length sequences and unsegmented data, a differentiable loss function to do back propagation ,and huge training time needed. The following sections are going to summerize the model and techniques used to solve these problems in a 2017 Baidu paper Deep Speech 2, reproduce a simplified model for Homework 2 Chinese digit recognition task, and discuss the result.

### II. Model and Techniques

#### 1. Model Architecture Overview
The figure shows the architecture of the model. Utterances are converted to spectrograms as the input of the model. One or more layers of either 1D or 2D Convolutional Layer follows. And then one or more layers of uni or bidirectional RNN or GRU is connected, followed by a lookahead convolutional layer and a fully-connected layers. Finally, a softmax layer produces the outputs. At each time step, the RNN takes an input and produces an output. The set of output classes for English is alphabets, space, apostrophe and blank for CTC loss, which will be discussed in another part of the section. For Chinese, the output classes is the most frequently used ~6000 Chinese characters, and possibly alphabets for hybrid Chinese-English cases. The outputs are compared to the label sequences to compute CTC losses for back propagation training.

#### 2. RNN & GRU
By using Recurrent neural network (RNN), output of each timestep is determined by the output of previous timestep and the input, which can be think of as the network having memories, thus suitable for this sequence-to-sequence task. Like Long short-term memory (LSTM), GRU is a more complex version of RNN cell which use additional gates to control the behaviour of the cell. There are only 2 gate, reset and update gate, in GRU compared to 3 in LSTM. The reset gate determines the portion of the previous output used, combining with the input to compute the new memory. The output is the weighted average of the new memory and the previous output, where the weight is controled by the update gate. Due to the lack of output gate, GRU has less parameter than LSTM, and faster to train. In the paper, it is stated that GRU and LSTM has similar performance under experiment with smaller dataset but faster to train and less likely to diverge, and thus, GRU is preferred over LSTM.

#### 3. CTC Loss
The main issue in end-to-end training is that the data is unsegmented. That is, which label beloning to which time segment of the input sequence is unknown. CTC (Connectionist Temporal Classification) is a method to deal with the problem, allowing the network to output a sequence of probability vector longer than the target label sequence, which later being decoded to the target sequence. In the following paragraphs, how decoding and the loss calculation is done is discussed respectively.

For the decoding process, the output sequence of the network denotes the probability of each output class at each timestep. A many-to-one mapping from the sequence of classes to the shorter target label sequence is defined, where repeated labels and then blanks are removed. For example, "AAABB" will become "AB", and both "A_AA_BB" and "AA_ABB_" will become "AAB". The blank symbol "_" allows the occurence of repetition of a character in the target label sequence. Then, the probability of a label sequence can be defined: $p(l|x)=\sum_{M(\pi)=l}p(\pi|x)$, where $M$ is the mapping and $\pi$ is all possible network-output-length sequences. The goal is to find a label sequence that maximize the probability given input sequence $x$. $l^* = argmax_{l}\ p(l|x)$. While there does not exist a polynomial time algorithm, approximation can be found by picking the best output class at each timestep($l^*\approx M(\pi^*)$) or using beam search. For application in speech recognition, language model can also be integrated in the decoding process for better result.

The loss is defined as the negative log-likelihood: $-\log{p(l|x)}$. To calculate $p(l|x)$, a forward-backward dynamic programming algorithm similar the HMM training one can be used. First, define foward variable $\alpha_t(s) = \sum_{M(\pi_t)=l_s} p(\pi_t|x)$, where $\pi_t$means output-length class sequence up to timestep $t$, $l_s$ means the subsequence of the first $s$ units of the label sequence. A recursion of $\alpha$ can be established. $\alpha_t(s) = (\alpha_{t-1}(s)+\alpha_{t-1}(s-1)) p(y_t=l'_s|x)$ if $l'_s=blank\ or\ l'_{s-2}$. Else, $\alpha_t(s) = (\alpha_{t-1}(s)+\alpha_{t-1}(s-1)+\alpha_{t-1}(s-2)) p(y_t=l'_s|x)$, where $l'$ is $l$ with blank inserted between all symbols including beginning and end. Then, $p(l|x) = \alpha_T(|l'|) + \alpha_T(|l'-1|)$. The backward variable $\beta$ is nearly the same as $\alpha$ except for the opposite direction (i.e. $t$ and $s$ are starting from the position to end). As the product of $\alpha$ and $\beta$ is gives the probability of all $\pi$s corresponding to $l'$ with symbol $l'_s$ at time $t$. With some rearranging, for any $t$, $p(l|x) = \sum_s \frac{\alpha_t(s)\beta_t(s)}{p(y_t=l'_s)}. With the equation, differentiation for back propagation can be computed.

#### 4. Batch Normalization


#### 5. Other Techniques

### III. Performance

### IV. Adaption to the Homework 2 Recognition Task
Since directly reproducing the result in the paper, which trained an entire end-to-end recognition model for a language with tens of millions of parameters, would require tremendous computational resources and time, in the project, a simplified version of the model for the homework 2 Chinese digit recognition task is trained, tested and compared to the result of the HMM model in homework 2.

### V. Result and Discussion

### VI. Conclusion

### VII. Reference
1. Deep Speech 2 paper: https://arxiv.org/abs/1512.02595
2. CTC paper: https://www.cs.toronto.edu/~graves/icml_2006.pdf
3. GRU: https://en.wikipedia.org/wiki/Gated_recurrent_unit
4. Keras image_ocr example: https://github.com/keras-team/keras/blob/master/examples/image_ocr.py
5. HTK Speech Recognition Toolkit: http://htk.eng.cam.ac.uk/
6. PyHTK: https://github.com/danijel3/PyHTK
