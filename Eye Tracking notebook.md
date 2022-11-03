# 2眼动预测研究笔记

### 1（已参考文献）1+（已参考重点文献）0（未参考文献）0+（未参考重点文献）

##### #1 A novel deep learning approach for diagnosing Alzheimer's disease based on eye-tracking data

![image-20220921222112773](E:\研究生作业合集\计算机体系结构\PIC\image-20220921222112773.png)

核心思想：设计**三维视觉空间记忆任务**，目的是为患者提供视觉刺激，同时记录他们的眼动数据并用于构建研究追踪数据集，并基于特定任务提出新的**基于深度学习模型-嵌套自动编码网络**，通过生成的注视热图提取出特点的眼动特征，同时利用**权重自适应网络层**进行特征融合，这有利于最后进行**二元分类**。并进行比较验证，最后通过**四重交叉验证**进行评估，所提出的模型在AD识别中显示出**85%**的平均准确率，**优于机器学习方法和其他典型的深度学习模型**。



模型： 嵌套自动编码器网络 权重自适应网络层 **自动编码器**是一种**无监督的神经网络模型** 它可以学习到输入数据的隐含特征，这称为编码(coding)，同时用学习到的新特征可以重构出原始输入数据，称之为解码(decoding)。

自动编码器**（加入模型介绍）**

![image-20221008220918018](E:\研究生作业合集\计算机体系结构\PIC\image-20221008220918018.png)

##### #2 Abnormalities of saccadic eye movements in dementia due to Alzheimer’s disease and mild cognitive impairment

![image-20220921222024926](E:\研究生作业合集\计算机体系结构\PIC\image-20220921222024926.png)

核心思想：主要是描述了眼动追踪可以区分两种形式的MCI，进一步论证了眼动追踪可以作为评估痴呆症的有用诊断生物标志物

----    可以作为例证 放在背景里面 

##### #3 Altered visual entrainment in patients with Alzheimer’s disease: magnetoencephalography evidence

**阿尔茨海默病患者的视觉夹带改变：脑磁图证据** 

![image-20220921223743491](E:\研究生作业合集\计算机体系结构\PIC\image-20220921223743491.png)

不太懂，大致意思是 阿尔茨海默症患者相对于正常患者，其在初级视觉皮层中表现出比刺激前基线期明显更强的15Hz夹带。总结为夹带视觉刺激能力的变化可能是补偿性的。

##### #4 Computational techniques for eye movements analysis towards supporting early diagnosis of Alzheimer’s disease: a review

![image-20220921232158601](E:\研究生作业合集\计算机体系结构\PIC\image-20220921232158601.png)

尽管做出了许多努力，早期AD的无创诊断仍然没有解决。眼球运动异常是AD疾病进展过程中的慢性病和慢性病的标志。例如，使用眼球跟踪器，这是一种测量眼睛注视和眼跳运动的设备。然而眼球跟踪器仅在受控实验室环境中使用。

##### 主要任务：

- 收集眼球运动数据的技术工具和方法
- 回顾了眼动与AD之间的关系的现有研究
- 最早的研究转向更适合早期检测的自然主义场景以来，眼动和AD的研究进展。
- 自然主义场景中有助于补充眼球运动和AD分析的计算技术。

##### 数据收集：

眼球追踪提供了一种无禁忌症的非侵入性工具，适用于潜在的AD筛查和追踪[22]。眼球跟踪器具有数据敏感性，适合分析AD患者的眼球运动异常。然而，眼球跟踪仪的构建有不同的技术途径[21]。因此，正确选择适合本研究的眼球跟踪器特征非常重要。

##### 眼球运动与阿尔茨海默病：

几项关于痴呆症的研究报告显示，大脑视觉区域的老年斑块和缠结会产生视觉症状[38，39]。

神经退行性疾病引起的视觉系统病理变化见[40-43]。这些病理变化的例子包括视力变化、不典型瞳孔反应和动眼神经性能的改变[44]。

眼球运动涉及由广泛大脑区域形成的复杂的眼球运动控制系统[45]。通过尸检研究，有证据表明与AD相关的病理学影响动眼神经脑区[46,47]。眼球运动模式的改变反映了潜在的视觉空间和执行功能损害。

因此，运动模式与高级认知控制过程有关[48]。例如，这就是为什么眼球运动可以探索视觉搜索背后的认知过程，提供人们在执行视觉搜索任务时如何觅食和计划的信息[49]。

相关文章中19.7%（14篇）是在过去3年发表的。这表明分析眼球运动的研究在AD研究中越来越重要。此外，最近的研究表明，眼动障碍分析有助于早期发现AD，也有可能用于评估疾病进展[43，50]。【18年的数据，如今过去四年，可借鉴这种方式进行新的扩展】

**3.1** **眼跳与阿尔茨海默病**

**行为：**参与者需要从起始点跳到出现的外围目标，从而记录受试者从注视呈现的周边目标开始的反应时间或潜伏时间。

**挑战（局限点）：**眼动异常并不排除AD，开发正确区分AD与其他疾病的技术很重要。其中很重要的就是-SEM异常与衰老有关。将SEM异常与AD、“正常”老化和其他情况区分开来是SEM分析的主要挑战。

##### 3.2 受控场景中的眼动分析

**行为：**

- 参与者执行了一项更复杂的任务，只涉及一个目标点，即检测和分类自然场景中的特定对象。参与者在监视器中观察到两种视觉刺激，一种包括动物图像，另一种包括干扰物图像。参与者被要求扫视当他们成功地注视动物和正确的图像时，包含动物的图像被测量。

- 阅读 参与者阅读句子时的眼睛注视位置。
- AD对视觉探查的影响。 参与者、AD患者和对照组受试者探索半球屏幕并对呈现的目标做出反应。
- 目标检测时间和注视次数的差异。研究使用视频观看期间的眼球运动分析来推断人们的认知功能。研究人员从注视中定义了13个特征，并发现了这些特征与记忆能力之间的相关性。这些特征包括平均注视持续时间、注视计数和平均眼跳幅度。

**结果：**

- 与对照组相比，AD患者即使处于疾病的轻度阶段，也很难选择相关目标。
- 注视时间更长并且对于句子的可预测性更差，说明了记忆和记忆提取功能受损。
- AD患者和对照受试者在从屏幕上识别不同偏心率的靶点时存在差异。

**局限：**

- 研究实验室任务的视觉缺陷对于AD评估很有成效，但其应用需要合作场景

- 受试者必须有意识地合作执行眼球运动任务，以获得他们的评估。

- 以便在不需要受试者明确合作的情况下对眼部运动异常进行评估，例如，分析AD如何影响ADL中的眼部运动，例如烹饪或园艺。

  所以，我们需要尽量在分析自然主义场景的眼球运动相关的工作。

**3.3 自然主义任务中的眼动分析**

**行为：**试图通过调查眼睛模式和行动上的眼-手协调来了解活动执行和眼球运动之间的关系

- 铃木等人[28]的研究，运动过程中的眼球运动。一名AD患者、一名后皮质萎缩（PCA）患者和一名健康受试者在进行运动活动（沿着走廊行走、上下楼梯以及穿过有障碍物或无障碍物的房间）时使用了眼球跟踪装置。结果表明，PCA患者的运动活动最慢。此外，PCA患者的固定时间比普华永道和健康受试者长。普华永道要求在任务竞赛期间进行提示，显示记忆受损。

**结果：**ADS患者对预期使用的物体不看一眼，在任务期间对无关物体的注视次数增加。

- AD患者的整体固定次数少于对照组和ADS患者。

**局限：**

- 需要更多的参与者进行更多的实验
- 视力缺陷并非AD独有，也存在在其他病理中
- 考虑每个患者的临床和个人病史
- 需要使之成为一种普及的工具，使受试者自然的方式进行活动

**AD患者在定位物体时使用场景语义的能力较低。从这个意义上说，对场景的理解是至关重要的。**

##### 4.1 利用计算注意建模实现早期检测

- 计算视觉显著性模型。通过计算分析视频以估计引起视觉注意的图像区域的这种方法称为视觉显著性检测[86，87]。这一研究领域与神经科学、心理学和计算机视觉领域相匹配[16]。**由于深度学习需要大量数据，仍然需要更多关于不同场景的注释信息。**

  - 自下而上建模

    颜色、对比度、方向和纹理等信息

  - 自上而下建模

    他们预测刺激引起的注意力。自上而下的模型需要事先了解视觉搜索任务和上下文目前大多数工作都属于自底向上方法，但很少有针对于AD患者的实验。

- 计算视觉显著性模型和诊断。

##### 5 Conclusions and Future Directions

**挑战：**

- 评估与活动相关的组织结构
- 不同认知问题的人进行试验

##### # Early detection of cognitive decline in mild cognitive impairment and Alzheimer's disease with a novel eye tracking test（2021）

![image-20220921232505473](E:\研究生作业合集\计算机体系结构\PIC\image-20220921232505473.png)

核心思想：

- 作者通过三分钟的眼动追踪设备（暂无方法）对NC，MCI，AD进行了评估。

结果：通过记忆和演绎推理任务可以有效的区分NC、MCI以及AD。从而为筛查早期发现MCI和AD提供了优势。

##### #5 Early emotional attention is impacted in Alzheimer’s disease: an eye-tracking study（no need）

![image-20220922141837008](E:\研究生作业合集\计算机体系结构\PIC\image-20220922141837008.png)

核心思想：阿尔茨海默症缺乏早期情感关注。

##### #6 Eye movement behavior identification for Alzheimer’s disease diagnosis

![image-20220922143117043](E:\研究生作业合集\计算机体系结构\PIC\image-20220922143117043.png)

核心思想：作者通过捕捉受试者阅读行为的描述符组成试验信息，并且使用这些消息去训练一组去噪稀疏自动编码器，并使用训练后的自动编码器和softmax构建神经网络分类器，最终得到了89.78%的准确率。

<img src="E:\研究生作业合集\计算机体系结构\PIC\image-20220924194338355.png" alt="image-20220924194338355" style="zoom: 25%;" />

##### #7 Eye tracking dysfunction in Alzheimer-type dementia（1984）

![image-20220922143802714](E:\研究生作业合集\计算机体系结构\PIC\image-20220922143802714.png)

核心思想：作为前言参考文献之一，主要论证的是阿尔茨海默症患者中发现视觉跟踪异常的严重程序与痴呆的严重程度之间有极强的相关性。

##### #8 Eye-tracking indices of impaired encoding of visual short-term memory in familial Alzheimer’s disease

![image-20220922174219516](E:\研究生作业合集\计算机体系结构\PIC\image-20220922174219516.png)

##### #9 Eye-Tracking Technologies in Mobile Devices Using Edge Computing:  eview

![image-20220922174754641](E:\研究生作业合集\计算机体系结构\PIC\image-20220922174754641.png)

精读！主要是参考并用于眼动追踪的介绍以及可行性论证

##### #10 Impact of Cognitive Demand on Eye Movement Pattern in Patients with Alzheimer’s Disease

![image-20220922175506329](E:\研究生作业合集\计算机体系结构\PIC\image-20220922175506329.png)

核心思想：如今眼动行为可以用作是识别阿尔茨海默症认知和行为模式的可靠工具。通过眼动追踪时间的相关参数将其进行区分。

## #11 Non-Invasive classification of Alzheimer’s disease using eye tracking and language

![image-20220922180153018](E:\研究生作业合集\计算机体系结构\PIC\image-20220922180153018.png)

核心思想：通过眼球运动的效用及其语音的组合可用于分类该任务，两者结合的方式可以将准确率达到0.80

精读！  提供了一个思路，将两者进行结合，去进行分类。

### 机器学习

### 数据集来源

##### 论文数据来源

##### DementiaBank Corpus

- The Cookie Theft Picture Description Task - 语音评估数据集
- 眼动数据预处理 - 为了捕捉用户的眼球运动和瞳孔行为，我们遵循相关工作中的标准方法，计算了一组关于注视、扫视和瞳孔大小数据的汇总统计数据(Toker等人，2017,2019;D 'Mello等，2012;Lall´e et al, 2016;Mart´ajnez - g´omez and Aizawa, 2014)。我们使用眼动数据分析工具包(EMDAT3)处理眼球跟踪数据，这是一个用Python编写的开源库。

**公共数据库和基因库**

AI技术在AD和其他疾病的应用中需要大量数据集，在许多数据库中由由数百到数千个条目组成，描述了许多临床和生物变量的主题，可用于通过分析疾病特征来开发新算法。在过去的 20 年中，许多开放数据共享计划在神经退行性疾病研究领域得到发展。比如：

1. 阿尔茨海默病遗传学联盟（ADGC，[http://www.adgenetics.org](https://link.zhihu.com/?target=http%3A//www.adgenetics.org)）；
2. 阿尔茨海默病测序项目（ADSP，[http://www.niagads.org/adsp/content/home](https://link.zhihu.com/?target=http%3A//www.niagads.org/adsp/content/home)）；
3. 阿尔茨海默病神经影像学倡议（ADNI，[http://adni.loni.usc.edu/](https://link.zhihu.com/?target=http%3A//adni.loni.usc.edu/)）；
4. AlzGene（[http://www.alzgene.org](https://link.zhihu.com/?target=http%3A//www.alzgene.org)）；
5. 英国痴呆症平台（ DPUK，[https://portal.dementiasplatform.uk/](https://link.zhihu.com/?target=https%3A//portal.dementiasplatform.uk/)）；
6. 阿尔茨海默病遗传学数据库（NIAGADS，[http://www.niagads.org/](https://link.zhihu.com/?target=http%3A//www.niagads.org/)）；
7. 全球阿尔茨海默病协会互动网络（GAAIN， [http://www.gaain.org/](https://link.zhihu.com/?target=http%3A//www.gaain.org/)）
8. 国家阿尔茨海默病和相关痴呆症集中数据库（NCRAD，[https://ncrad.iu.edu](https://link.zhihu.com/?target=https%3A//ncrad.iu.edu)）

在“组学时代”中，其他多个综合类数据库也可以应用于神经退行性疾病—例如GEO数据库和UK biobank。在AD领域，公共和私有数据库的开发有助于更全面地了解疾病异质性，以及个性化医疗和药物的开发。

### 实验细节

- 分类器:我们测试了三种不同的分类算法，**逻辑回归(LR)、随机森林(RF)和高斯朴素贝叶斯(GNB)**，它们在最密切相关的工作中报告了最佳性能Masrani(2018)。我们使用**scikit-learn**(一个用于机器学习的python包)来执行分类。

  - 为了结合不同模式的数据进行研究，我们探索了早期融合和晚期融合方法。这是通常用于多模式方法的两种融合方案。

    在早期的融合中，我们将两种模式的特征串联起来，制作一个特征向量来学习分类器。这种方法很简单，它允许基于分类器的特征之间的建模交互。我们的早期融合模型如图2所示。

    ![image-20220925135927813](E:\研究生作业合集\计算机体系结构\PIC\image-20220925135927813.png)

    后期融合方案在决策级别结合了来自每种模态的预测(见图3)。我们使用了一种广泛建立的后期融合方法，称为“平均投票”，在这种方法中，预测是通过将不同学习算法(具有异构模型表示)的输出平均到单个数据集(Battiti和Colla, 1994)。在我们的案例中，我们使用了一种稍加修改的方法，即对单一学习算法产生的预测概率进行平均，但应用于不同的数据模式，如Fraser等人(2019)所述。

    ![image-20220925135934941](E:\研究生作业合集\计算机体系结构\PIC\image-20220925135934941.png)

- 基线:为了观察眼动特征本身是否是足够准确的分类器，我们使用一个**零规则分类器(B)**，它总是预测训练数据中的大多数类，作为比较的基线。

- 特征集:我们比较了仅使用任务不可知(E TA)特征、仅使用任务特定特征(E TS)以及两者的组合(E)的分类器的分类性能。

- 评估:我们使用分层**10倍交叉验证方法**评估分类器，该方法在不同的分层分段上重复10次(运行)，以加强结果的稳定性和可重复性。我们根据ROC曲线下的面积(AUC)报告分类性能，10次折叠和10次运行的平均值。在交叉验证的每一层，我们执行相关特征选择Hall(1999)，以删除高度两两相关的特征(Pearson r >.85)以及与结果相关性极低的特征(Pearson r <.2)。

### 结果

![image-20220925142225802](E:\研究生作业合集\计算机体系结构\PIC\image-20220925142225802.png)



### 局限

- **有限的数据集**
- 并没有精细的调优参数
- 没有探索更高级的特诊选择
- 数据质量比数据数量更重要
- 眼球追踪技术不够敏感，无法描述微眼跳微眼动运动的差异。（原因在于 如果想要捕捉高分辨率的眼球运动，头部运动需要使用下巴托）
- 健康对照组未经过类似于患者组的相关验证，导致有概率在健康人群中存在未诊断的神经退行性问题。
- AD经常被误诊
- 眼球运动、瞳孔以及语言的变化可能与一些神经系统疾病一起发生。

### 未来展望

- 队列的增加（数据量的增加）
- 机器学习模型的技术改进
- 多模态特征设计
- 学习算法
- 神经网络算法 

##### #12 Shortening of Saccades as a Possible Easy-to-Use Biomarker to Detect Risk of Alzheimer’s Disease

![image-20220922234955080](E:\研究生作业合集\计算机体系结构\PIC\image-20220922234955080.png)

##### #13 The feasibility of using virtual reality and eye tracking in research with older adults with and without Alzheimer's disease

![image-20220922235217167](E:\研究生作业合集\计算机体系结构\PIC\image-20220922235217167.png)

##### #14 The investigation of simultaneous EEG and eye tracking characteristics during fixation task in mild alzheimer’s disease

![image-20220923001216665](E:\研究生作业合集\计算机体系结构\PIC\image-20220923001216665.png)

核心思想：论证眼动追踪生物标志物可以作为AD的早期检测的非侵入性工具。

##### #15 Machine learning algorithms on eye tracking trajectories to classify patients with spatial neglect

![image-20221016093701388](E:\研究生作业合集\计算机体系结构\PIC\image-20221016093701388.png)

核心思想：扫视轨迹表征视觉空间忽视迹象的挑战。

方法：标准化的预处理管道，利用传统机器学习以及深度卷积网络来自动分析眼睛轨迹。（主要是参照于什么样的眼睛轨迹）结果呈现于一个什么样的图表

##### #16 Prediction of visual attention with deep CNN on artificially degraded videos for studies of attention of patients with Dementia

![image-20221016110542697](E:\研究生作业合集\计算机体系结构\PIC\image-20221016110542697.png)

论证了为视觉内容开发自动预测模型以用于设计痴呆症患者的心理视觉体验是一个研究方向。难点在于**训练数据量非常少。**利用迁移学习策略，深度学习。

##### #17 Prosaccade and Antisaccade Paradigms in Persons with Alzheimer’s Disease: A Meta-Analytic Review

![image-20221016112656785](E:\研究生作业合集\计算机体系结构\PIC\image-20221016112656785.png)

这些发现强调了反跳动错误率是区分 AD/MCI 和健康老年人之间抑制能力的可靠工具。

##### #18 Sensors | Free Full-Text | Analyzing Facial and Eye Movements to Screen for Alzheimer’s Disease

![image-20221016113733763](E:\研究生作业合集\计算机体系结构\PIC\image-20221016113733763.png)

我们的研究结果表明，客观和准确的面部和眼球运动测量可用于快速筛查此类患者。使用基于摄像头的测试来早期发现有神经退行性疾病迹象的患者可能会对痴呆症的公共护理产生重大影响。



## 多模态

##### （1+）#1 Predicting MCI Status From Multimodal Language Data Using Cascaded Classifiers

![image-20221026162457812](E:\研究生作业合集\计算机体系结构\PIC\image-20221026162457812.png)

关键在于给定了三个任务：图片描述、默读和朗读。并通过不同的数据来捕捉每个任务的信息。并从每个模式中提取特征，并用于一系列级联分类器（多模态机器学习）。可以看到的是通过任务级别结合数据来实现能达到最高的数据准确度。

 “Cookie Theft” 

LR SVM

局限性和未来展望

- 未来的工作将探索使用新兴技术来跟踪网络摄像头和其他便携式设备的眼睛运动
- 我们依赖于Cookie Theft任务的手动转录，并且自动提取了一些特征，然后手动进行了校正。
- 研究使用自动语音识别和其他技术来完全自动化处理管道
- 使用别的包括已被证明可以检测 MCI 的其他语言任务。
- 自动特征提取和分类可能有助于对认知功能进行敏感的纵向监测。

## 语言分析

##### （0+）#1 Alzheimer’s disease and automatic speech analysis: A review

![image-20221026234644594](E:\研究生作业合集\计算机体系结构\PIC\image-20221026234644594.png)

##### （0+）#2 NLP Based Automated Screening Tools for Alzheimer’s Disease

![image-20221028163342510](E:\研究生作业合集\计算机体系结构\PIC\image-20221028163342510.png)

NLP

##### （0+）#3 Detection of Alzheimer’s Disease Through Speech Features and Machine Learning Classifiers

![image-20221028164146413](E:\研究生作业合集\计算机体系结构\PIC\image-20221028164146413.png)

回顾了相关非侵入信号处理技术，并描述了AD检测中可行的各种分类器和机器学习算法

##### （0+）#4 Deep-Learning-Based System for Assisting People with Alzheimer’s Disease

![image-20221028164304332](E:\研究生作业合集\计算机体系结构\PIC\image-20221028164304332.png)

基于深度学习算法的人工智能系统，使用人工智能识别视频中的人类活动。

Paper主要是描述了利用人工智能帮助阿尔茨海默症患者以及护理方面的难题，填补了智能辅助软件领域的空白

##### （0+）#5 Deep learning approach for early detection of Alzheimer’s disease

![image-20221028164441464](E:\研究生作业合集\计算机体系结构\PIC\image-20221028164441464.png)

深度学习算法 特别是卷积神经网络（CNN）Paper主要还是针对与2D和3D的结构性脑扫描进行分析

##### （0+）#6 Automated speech based evaluation of mild cognitive impairment and Alzheimer’s disease detection using with deep belief network model

![image-20221028190635449](E:\研究生作业合集\计算机体系结构\PIC\image-20221028190635449.png)

使用自动语音识别（ASR）和DL模型创建检测模型。首先采用了高斯混合函数来检测自发语言中的ASR，随后利用DBN来讲识别的语言数据中提取特征向量，最后使用SoftMax分类器进行分类。

##### （0+）#7 Automated text-level semantic markers of Alzheimer's disease

![image-20221028171007118](E:\研究生作业合集\计算机体系结构\PIC\image-20221028171007118.png)

自然语音数据统计和机器学习分析。
