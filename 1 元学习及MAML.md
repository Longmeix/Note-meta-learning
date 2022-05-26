# 元学习

约一年前了解到元学习，特别喜欢这个idea，因为“让机器学会如何学习”打破了传统机器学习的训练思维，很符合现实中人的学习过程。近期想把它研究社交网络，看了许多从实验和理论角度解释元学习有效的文章，在此梳理一下加强记忆，也希望能启发到他人。

本文主要参考李宏毅老师2020年机器学习ppt。

# 一、元学习介绍

## 1. 为什么要用meta-learning

传统的机器学习和深度学习的模式是：获取目标领域的大量的样本数据来训练一个模型，若任务发生变化则需重新训练模型。这会带来两个问题，从数据方面看，机器模型的好坏很大程度上依赖于训练数据的规模和质量，正所谓”garbage in, garbage out”；从资源消耗上看，切换一个任务就得从头训练十分耗时耗力，当模型参数庞大更是无法忍受这样的计算成本。

一种广泛的解决办法是迁移学习，将源域的知识迁移到目标域。不过元学习或许提供了一种更为有效的方法，它力图让模型获取一种“学会学习”的能力，而不是只掌握单个领域知识。回想一下人类的学习，一个小孩仅看过几张苹果和梨子的照片，就能将他从未见过的水果区分开。受此启发，我们是否能让模型也只通过少量数据就学会新领域知识，具有快速学习的能力呢？这就是元学习的动机。

## 2. Meta learning概念

元学习又称“learn to learn”，学习如何去学习，目的是成为一个拥有学习能力的学霸，而不是背题家。机器学习通过数据$D_{train}=\{{\mathbf{x},\mathbf{y}}\}$找一个函数f*，希望f*(x)尽可能接近y；元学习不直接学f*，而是根据数据学一个函数F，它能寻找到对某任务最佳的函数$f^*$，即$f^* = F(D_{train})$，描述特征和标签之间的关系。这里F是learnable的，比如模型网络结构、初始化参数和学习率等超参数，可用深度学习训练。总之，meta learning希望机器学会自己设计网络结构等配置，减少人为定义。

分类：根据F要学习的组件类型，元学习可主要分为三类：基于度量的方法（metric-based），基于模型的方法（model-based），基于优化的方法（optimization-based）。其中应用最广，也最适合初学的当属基于优化的MAML。

![元学习要学习的部件](%E5%85%83%E5%AD%A6%E4%B9%A0%2013aa8b4877db49f4b227d9bdb667e3d3/Untitled.png)

元学习要学习的部件

![元学习总体框架](%E5%85%83%E5%AD%A6%E4%B9%A0%2013aa8b4877db49f4b227d9bdb667e3d3/Untitled%201.png)

元学习总体框架

# 二、元学习的建模过程——以MAML为例

## 1. MAML框架

权重初始化的好坏很大程度上影响模型的最终性能。  ****MAML（Model-Agnostic Meta-Learning）是17年发表在ICML上的一种通用优化算法，适用于任何基于梯度学习的模型。它目的”learning to initialize“，为不同的任务提供初始化参数，以便面对新任务时能快速学习。

机器学习的数据分为训练集、验证集、测试集。训练过程一般经历三步：定义一系列函数f — 设计评价函数的好坏的指标（loss） — 挑选出最好的函数f*。如图，我们熟知的梯度下降法做图像分类的流程大致是：先定义一个网络结构$f_{\theta}$如CNN —> 初始化网络参数$\theta$ —> 输入第一个batch的训练数据，计算loss —> 计算梯度 —> 更新参数 —> 下一个batch训练 —> ……，训练完毕后得到最佳参数$\theta$^hat，对应最佳函数为$f^{*}_{\theta}$ 。

![机器学习过程](%E5%85%83%E5%AD%A6%E4%B9%A0%2013aa8b4877db49f4b227d9bdb667e3d3/Untitled%202.png)

机器学习过程

![元学习的数据以task为单位划分](%E5%85%83%E5%AD%A6%E4%B9%A0%2013aa8b4877db49f4b227d9bdb667e3d3/Untitled%203.png)

元学习的数据以task为单位划分

相比传统训练，元学习对数据的划分更有层次，引入了Task的概念，将数据装到一个个规模很小的task中，希望通过多次训练少量样本快速完成学习的目的。接下来介绍一些元学习的名词。元学习里的训练和测试阶段叫meta-training和meta-testing，对应的数据称为training tasks和testing tasks，也可以划分验证集即validation tasks。每一个task里又包含训练数据和测试数据，分别称为support set和query set。构建task时，N-way，K-shot指每个task中包含N个类别，每个类别下只有K个样本数据。如图是在做一个2-ways，1-shot的图像分类任务。一个Meta batch包含meta_bsz个tasks，和机器学习的batch概念相似，批处理数据。

所以，元学习流程是：定义网络结构$F_{\phi}$，$f_{\theta}$—> 初始化网络参数$\phi$ —> 输入第一个**meta batch**的tasks—> 在task 1上的support set计算loss和梯度，更新参数$\theta$ —> 根据query set计算$l_1$—> 训练task2 —> …… —> 对第一个meta batch的所有loss求和，计算梯度，更新参数$\phi$ —> …… —> 所有meta batch训练完毕。

（还没描述任务网络和外网络一样

![元学习中一个task的loss计算](%E5%85%83%E5%AD%A6%E4%B9%A0%2013aa8b4877db49f4b227d9bdb667e3d3/Untitled%204.png)

元学习中一个task的loss计算

![加总training loss](%E5%85%83%E5%AD%A6%E4%B9%A0%2013aa8b4877db49f4b227d9bdb667e3d3/Untitled%205.png)

加总training loss

值得注意的是，元学习与机器学习一个很大的不同是loss的计算。如图，我们是先用training example（support set）的loss对任务网络的参数更新过一次后，再在testing examples（query set）上计算loss，用这些loss计算的梯度更新参数$\phi$，学到”learning algorithm“$F_{\phi^*}$，当要解决一个新任务时，F能得到对任务的合适函数f。而机器学习如预训练，是直接在训练数据上计算loss和梯度，学习函数f。

![Untitled](%E5%85%83%E5%AD%A6%E4%B9%A0%2013aa8b4877db49f4b227d9bdb667e3d3/Untitled%206.png)

[https://www.notion.so](https://www.notion.so)

## 2. Code实现

关于maml的实现，[原作者的代码](https://github.com/cbfinn/maml)基于TensorFlow。我当时是看李宏毅老师视频学的元学习，[他的网站](https://speech.ee.ntu.edu.tw/~hylee/ml/2020-spring.php)上有相应作业和参考代码，包括图片处理到task构造到MAML模型的完整流程，注释比较多还有助教讲解。适合入门元学习的实验有[regression](https://colab.research.google.com/drive/1MFJwRdOTefd6UOYRsNjdc7BWuB7Qe3lY)、[Omniglot图像分类](https://colab.research.google.com/drive/1OcF5TQCCd7WNK0cbXyzYxAzWpMKW_r8B)****。****如果想先快速run起来，[开源库learn2learn](https://github.com/learnables/learn2learn)是个不错的选择，不过如果想更深入的理解MAML还是建议看更底层代码。

```python
maml = l2l.algorithms.MAML(model, lr=0.1)
opt = torch.optim.SGD(maml.parameters(), lr=0.001)
for iteration in range(10):
    opt.zero_grad()
    task_model = maml.clone()  # torch.clone() for nn.Modules
    adaptation_loss = compute_loss(task_model)
    task_model.adapt(adaptation_loss)  # computes gradient, update task_model in-place
    evaluation_loss = compute_loss(task_model)
    evaluation_loss.backward()  # gradients w.r.t. maml.parameters()
    opt.step()
```

## 3. 对比

# 三、Deeper insight

## 1. 类比人的学习

个人很喜欢元学习的一个原因是，我能将其与自己以往的学习经历结合，理解它的理念，直觉上感受它有效的原因。举个熟知的例子——高考，想想高中我们是如何在3年里快速掌握多门学科知识，提高学习能力的。最常见的高考模式要求考生的总分越高越好，数学满分但单科成绩落后并不是我们所期望的。那平时我们是怎么学习的呢？主要是自我练习+单元测验。平日里，学生会自己刷练习题，对比标准答案。每隔一段时间，会有班级的单元测验或月考，考察学生对新知识的掌握情况。你细品，元学习是不是和人类快速学习新知识的步骤有相似之处。

以3-ways 5-shot图像分类任务为例，每个batch大小meta_bsz=8。首先，目标是要在meta testing上取得高性能（高考总分高），testing data包含各种类别的图像（高考题有语数英等七个学科题目）。第二，在元训练阶段，对于每一个task（随机抽取3门学科各5 * 2道题），$\phi$先用support set微调出适应该任务的参数$\theta$（先写数学练习题找到”题感“），再在query set上计算loss（单元测验还错的题）。将一个meta batch内8个tasks在query set上的loss求和（所有测验的错题），计算梯度来更新$\phi$（从测验的错题中学习，改正）。利用梯度下降法，loss会越来越低（测试错得越少），最终模型通过元学习获得了快速学习能力。

可以看出，support set相当于平时自学做的练习题，而query set则是单元测试（往往是平时练习题的变形，表面不同但底层知识相似）。比如现在小明小红都想快速学会余弦函数的知识点，但仅已知10道余弦函数的题目及答案，怎么做到呢？小红机智地先对题目分门别类，5道当练习，5道当自我测试。结合以前的知识，她会去学习如何从题目已知信息得到正确答案。然而，她从练习中总结的规律未必是正确的，若她能在测试题上做对，我们才有理由相信她可能掌握了新知识，这是许多学霸的自学方式。另一种做法接近机器学习的流程，小明直接从10道题中总结规律，他也许学会或背会了这10道题，但我们很难判断他是否真正掌握了知识，如果直接上考场有极大概率翻车。这就好比在分辨牛和狗的图片时，模型是根据背景为绿色的草来判断图片为牛，而不是识别出了牛的鼻子等特征，因此背景一旦更换模型就失效了。

## 2. 学习能力如何获得

对比MAML和人类自学的流程，我认为MAML在”学会学习“的设计上有两个聪明之处（讲错了别打我，欢迎讨论><）

1）将机器学习的batches划分成若干tasks，tasks的构建

机器学习训练和更新参数的单位是batch，而元学习是在若干个数据量很小的task上训练，以meta batch为单位更新meta网络参数，task内会适应任务fine-tune子网络参数。虽然batch也能缩小size来达到和task相似的概念，但它俩**最大的不同是batch的数量内的数据是混杂的，而task的构建是有讲究的**！**task内的数据类别要比较相似**，”learn to compare“，经验上也是字形相近的单词一起记相比单词和数学公式糅杂记忆的效果更好。而一个元batch的**tasks间涉及的知识可以多样一些**，比如tasks全是中文英语法语等语言类，可能会使我们有了快速掌握新语言的学习能力，但数学类的学习能力很差，这并不是我们期待的全能型learner。

2）query set提供了检验学习能力的机会

元学习和机器学习的很大不同是，在task内再将数据划分成了support set 和 query set，其实或许training examples和testing examples 这两个名词能更好地解释设置它们的本来目的（可能怕和机器学习里的train和test混淆）。上一小节将query set类比成了单元测试，这看起来是合乎常识的，让模型从support set中找寻规律知识，再在query set上检验模型是否学到的是本质知识。如果模型仅学到了表面知识，如绿色背景的草地，那么在query set的表现就会很差，之后的task训练中它可能会思考怎样才能学到关键的核心知识，这个过程就好似人类的学习，学习能力也是在这样的反复检验中习得的。

## 3. maml训练注意的问题

接下来总结我目前从文献和实验中感受到的经验。

1. task内的样本数据要相似，一个meta batch内的tasks最好具有一定差异性。
2. task内的support set和query set需要有一定关联（底层知识相似），可以不一样也可以完全一样。

目前了解到task的划分有两种方式：support set和query set**不一样称为train-validation**，**一样的称为train-train**。在实践中，不同划分对训练结果的影响不同，具体可参见【】【】