# 脑结构成像（CT、MRI）

##### CT（X线电子计算机断层扫描）主要是利用X线断层扫描，电光子探测器接收，并把信号转化为数字输入电子计算机，再由计算机转化为图像，是一种无痛苦、无损伤的辅助检查工具。

**MRI**  （**磁共振成像）**  **是根据有磁矩的原子核在磁场作用下，能产生能级间的跃迁的原理而采用的一项新检查技术，对人体无害。MRI对脑内低度星形胶质细胞瘤、神经节、神经胶质瘤、动静脉畸形和血肿等的诊断确认率极高，对脑实质和脑脊液的显示度极好。** 

**sMRI：   结构性磁共振成像 顾名思义，是为了产生某一组织结构的一项成像技术。科学家利用了氢原子，因为人体任何组织里都有水分子（不用氧原子后面说），自然用氢原子来定位比较准确干扰少。如下图，科学家可以利用MRI探究腹腔内情况。**

**fMRI：功能性磁共振成像   顾名思义，是要研究针对某个脑活动（脑功能定位）激发某个脑区活动进行研究。这一项大幅提升MRI技术的创始人是日本科学家 小川诚二，他突破性使用血氧浓度相依对比（Blood oxygen-level dependent, BOLD）。简单而言就是，在某个脑区剧烈活动时候必然消耗更多的能量，自然就要消耗更多氧气（相对应的，PET技术针对能量代谢的成像技术，不过需要服用同位素药物），所以说通过检测血氧对比程度可以发现脑区活动情况。如下图，科学家可以通过BOLD探究脑部活动区别。**

**PET**  **（正电子发射断层摄影）**   **是新发展起来的核医学检查方法。扫描前先给病人注射一种标记某种正电子的放射性制剂，从它们所参与的代谢过程来测定脑组织的代谢改变。是目前惟一可在活体上显示生物分子代谢、受体及神经介质活动的新型影像技术，现已广泛用于多种疾病的诊断与鉴别诊断、病情判断、疗效评价、脏器功能研究和新药开发等方面。** 



#####  # A deep learning framework with an embedded-based feature selection approach for the early detection of the Alzheimer's disease

##### 具有嵌入式特征选择方法的深度学习框架，用于阿尔茨海默病的早期检测

![image-20220929112559641](E:\研究生作业合集\计算机体系结构\PIC\image-20220929112559641.png)

**数据来源**：AD DNA**甲基化数据集**

**模型**

通过执行质量控制，归一化和下游分析对数据进行**预处理**

基于嵌入式特征选择模型作为分类模型　**增强型深度递归神经网络（ＥＤＲＮＮ）**

##### ＃Multi-modal data Alzheimer’s disease detection based on 3D convolution

##### 基于3D卷积的多模态数据阿尔茨海默病检测

![image-20220929114640516](E:\研究生作业合集\计算机体系结构\PIC\image-20220929114640516.png)

数据来源：**ADNI**

多模态融合数据　PET+MRI

3D卷积神经网络

##### # A deep learning MRI approach outperforms other biomarkers of prodromal Alzheimer’s disease

![image-20220929161107810](E:\研究生作业合集\计算机体系结构\PIC\image-20220929161107810.png)

核心思想在于论证了MRI对于其他生物标志物有更大的价值在根据生物标志物进行预测中。

##### # A Single Model Deep Learning Approach for Alzheimer’s Disease Diagnosis

![image-20220929162812307](E:\研究生作业合集\计算机体系结构\PIC\image-20220929162812307.png)

核心思想：1.不同卷积神经网络之间的评估

2.提出了一种数据增强策略

3.引入了可解释方法 使模型更透明

##### # Deep learning approach for early detection of Alzheimer’s disease

![image-20220929163417230](E:\研究生作业合集\计算机体系结构\PIC\image-20220929163417230.png)

迁移学习 准确率高达97%

##### # Deep Learning-Based Prediction of Alzheimer’s Disease from Magnetic Resonance Images

![image-20220929163735988](E:\研究生作业合集\计算机体系结构\PIC\image-20220929163735988.png)

MRI 使用了VGG迁移学习 ResNet等网络架构

##### # Detecting the stages of Alzheimer’s disease with pre-trained deep learning architectures

![image-20220929170316344](E:\研究生作业合集\计算机体系结构\PIC\image-20220929170316344.png)

MRI 评估了29种不同的预训练模型（迁移学习）

- **首先，第一个关键技术是 Transformer**。
- **第二个关键技术是自监督学习**。
- **第三个关键技术就是微调**。
- **总体来讲，预训练模型发展趋势：第一，模型越来越大。比如 Transformer 的层数变化，从12层的 Base 模型到24层的 Large 模型。导致模型的参数越来越大，比如 GPT 110 M，到 GPT-2 是1.5 Billion，图灵是 17 Billion，而 GPT-3 达到了惊人的 175 Billion。一般而言模型大了，其能力也会越来越强，但是训练代价确实非常大。第二，预训练方法也在不断增加，从自回归 LM，到自动编码的各种方法，以及各种多任务训练等。第三，还有从语言、多语言到多模态不断演进。最后就是模型压缩，使之能在实际应用中经济的使用，比如在手机端。这就涉及到[知识蒸馏](https://www.zhihu.com/search?q=知识蒸馏&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A1465037757})和 teacher-student models，把大模型作为 teacher，让一个小模型作为 student 来学习，接近大模型的能力，但是模型的参数减少很多。**

EfficientNet作为性能最高的模型，实现了较高的分类性能

##### # FDN-ADNet: Fuzzy LS-TWSVM based deep learning network for prognosis of the Alzheimer’s disease using the sagittal plane of MRI scans

![image-20220929172345973](E:\研究生作业合集\计算机体系结构\PIC\image-20220929172345973.png)



阿尔茨海默病（AD）是最普遍的不可逆的精神神经障碍，导致记忆力下降。

MRI的矢状面提供了大脑中段区域的更多视觉特征。

对MRI图像进行预处理，以进行矢状平面提取，图像配准和关键切片提取。

ResNet101 深度学习网络 （DLN） 用于从矢状平面切片中提取特征。

基于模糊的分类器用于对从DLN中提取的特征进行分类，以分类CN与AD，CN与MCI，AD与MCI。

##### # Improving Sensitivity of Arterial Spin Labeling Perfusion MRI in Alzheimer's Disease Using Transfer Learning of Deep Learning-Based ASL Denoising

![image-20220929172534407](E:\研究生作业合集\计算机体系结构\PIC\image-20220929172534407.png)

##### # Multimodal deep learning models for early detection of Alzheimer’s disease stage | Scientific Reports（nature）

![image-20220929172726178](E:\研究生作业合集\计算机体系结构\PIC\image-20220929172726178.png)

去噪自动编码器 提取特征

3D卷积神经网络成像

提出了一种新颖的数据解释方法，通过聚类和扰动分析来识别深度模型所学习的顶级特征

数据集采用了ADNI数据集 证明了深度学习优于机器学习

##### # Predicting clinical scores for Alzheimer’s disease based on joint and deep learning

![image-20220929173100194](E:\研究生作业合集\计算机体系结构\PIC\image-20220929173100194.png)

## 突出

一个联合和深度学习框架来预测AD的临床评分。

组 LASSO 和协熵通过特征选择进行降维。

了多层独立递归神经网络回归。

学习MRI和临床评分之间的关系来预测临床评分。