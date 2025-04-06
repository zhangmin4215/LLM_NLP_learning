Swin Transformer: Hierarchical Vision Transformer using Shifted Windows  
基于移位窗口的分层ViT

0.摘要  
* 将Transformer从语言领域迁移到视觉领域所面临的挑战，主要源于二者之间的本质差异，例如：视觉实体的尺度变化更为显著，且图像像素的分辨率远高于文本中的单词。
* 这种移位窗口的机制通过将自注意力计算限制在非重叠的局部窗口内，显著提升了计算效率，同时仍保留了跨窗口的连接能力。
* 这种分层架构具有多尺度建模的灵活性，并且其计算复杂度与图像大小呈线性关系。

1.引言  
![image](https://github.com/user-attachments/assets/ecf9ec81-e5fb-4448-a1b7-06acaace9e8e)
* 提出的Swin Transformer通过深层合并图像块（如灰色区域所示），构建分层特征图。由于自注意力计算仅在各局部窗口内进行（如红色区域所示），其计算复杂度与输入图像尺寸呈线性关系。因此，该模型可作为通用骨干网络，同时适用于图像分类和密集预测任务。
* 卷积神经网络通过pooling池化(文中的类似操作是patch merging)得到多尺度的特征，能增大每个卷积核看到的感受野
* 一旦有了多尺寸的特征信息，输给一个FPN就可以做检测任务；输给一个UNet，就可以做分割任务

![image](https://github.com/user-attachments/assets/677f957c-ef28-4987-b0c0-70711501a17a)
* 左图向右下移动2个patch得到右图
* 在第l层（左图）中，采用常规的窗口划分策略，自注意力计算仅在每个窗口内部进行。而在相邻的l+1层（右图）中，窗口划分发生偏移，形成新的窗口区域。此时新窗口的自注意力计算将跨越前一层（l层）的窗口边界，从而建立跨窗口的连接机制。

3.方法  
3.1整体架构  
![image](https://github.com/user-attachments/assets/dfaa3ceb-8432-4b2f-9418-0ff3f690378e)
Patch partition：将图像分块

![image](https://github.com/user-attachments/assets/f7695219-d8c4-4247-ade6-a0ffdc69e110)
Patch merging  
第一张图：每隔一个像素点选择一个  
第3张-->第4张：使用1x1卷积核  

3.2基于自注意力的移动窗口
![image](https://github.com/user-attachments/assets/5ec2011c-4f9b-4a14-b6b2-bba928a9c238)
Transformer block安排：做一次基于窗口的多头自注意力，再做一次基于移动窗口的多头自注意力。达到窗口和窗口之间的互相通信
