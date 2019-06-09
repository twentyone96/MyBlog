---
layout: post
title:  "空间旋转平移描述"
date:   2019-06-09
excerpt: "介绍旋转矩阵、欧拉角、旋转向量、四元数等描述方法"
tag:
- markdown 
- math
- jekyll
comments: true
---



## 空间旋转平移描述

为了描述空间中一点的**位置**，我们需要这个点在世界坐标系\\(C_W\\)(World)下的坐标\\((x,y,z)​\\)来描述。但是对于刚体而言，由于其形状不能忽略，除了描述其**位置**还需要描述其在空间中的**姿态**（也可以叫方位），这里**位置和姿态简称为位姿**。

数学上，刚体的位姿有6个自由度(dof)，其中刚体的位置3个，刚体的姿态3个。刚体的位置即其质心在世界坐标系下的坐标，然而刚体的姿态的描述却没那么简单。我们以刚体的质心为圆点，建立一个固连在刚体上的物体坐标系\\(C_R\\)(Rigid Body)，当刚体运动时，固连在其上的坐标系也随之运动，自然而然**物体坐标系\\(C_R\\)与世界坐标系\\(C_W\\)的关系就可以用于刻画刚体在世界坐标系下的姿态（还有位置）**。下文中的讨论的平移与选择都指的是坐标系之间的平移旋转。

物体坐标系\\(C_R\\)与世界坐标系\\(C_W​\\)的关系的描述有多种方法，为了介绍的方便，这里将其分成单独描述旋转、单独描述平移以及旋转和平移同时描述：

单独旋转描述：旋转矩阵R、欧拉角、泰特布莱恩角、RYP角、罗德里格向量、四元数表示法

单独平移描述：平移向量t

旋转+平移描述：变换矩阵T

### 1 变换矩阵T 旋转矩阵R 平移向量t

#### 旋转矩阵R与平移向量t

对于空间中一点\\(P=(x,y,z)^T\\)，在坐标系A下记为\\(^AP\\)。在坐标系B下记为\\(^BP\\)，那么\\(^AP\\)与\\(^BP\\)的关系可以用下式来表示：
\\[
^AP = ^A_BR^BP + ^A_Bt
\\]
矩阵\\(^A_BR_{3 \times 3}\\)即所谓的旋转矩阵，向量\\(^A_Bt_{3 \times 3}\\)即所谓的平移向量。

旋转矩阵\\(R_{3 \times 3}​\\)有很多漂亮的性质，比如\\(R^T=R^{-1}​\\)，\\(R​\\)的列向量为单位向量且两两正交，\\(detR=\pm 1​\\)，这里篇幅原因就不赘述。

同理我们有：
\\[
^BP = ^B_AR^AP + ^B_At
\\]
比对\\((1)(2)​\\)两式，我们可以得到：
\\[
^A_BR = ^B_AR^T =  ^B_AR^{-1}
\\]

\\[
^A_Bt = -^B_AR^B_At
\\]

\\(^A_Bt\\)几何上可以理解为从坐标系B的原点在坐标系A下的坐标，可以写成\\(^AO_B\\)。记坐标系B的基分别为\\(i,j,k\\)，那么旋转矩阵的几何解释为：
\\[
^A_BR=
\begin{bmatrix}
^Ai_B & ^Aj_B & ^Ak_B
\end{bmatrix}
\\]
![2019-04-17_094638](1.png)

#### 旋转平移的另一种描述

点\\(P\\)在两个坐标系下的坐标转换\\(^A P \rightarrow  ^BP\\)可以有两种方式，先旋转(坐标系)再平移，先平移再旋转(坐标系)，前一种方式即公式\\((1)\\)，后一种方式为下式：
\\[
^AP = ^A_BR(^BP - ^B_At)
\\]

\\(^B_At\\)几何上可以理解为从坐标系A的原点在坐标系B下的坐标，可以写成\\(^BO_A\\)。

#### 变换矩阵T

如果点\\(P\\)写成齐次坐标形式\\(\tilde P=(x,y,z,w)\\)，那么公式\\((1)\\)可以简化为：
\\[
^A\tilde{P} = ^A_BT^B\tilde{P}
\\]
其中，
\\[
^A_BT_{4 \times 4} = 
\begin{bmatrix}
^A_BR & ^A_Bt \\\\ 0^T & 1 
\end{bmatrix}
\\]



### 2 欧拉角

对于空间中两原点重合的坐标系\\(A​\\)和坐标系\\(B​\\)，矩阵\\(^A_BR​\\)作用于\\(^B P​\\)上得到\\(^A P​\\)的过程可以分解为以下形式：

\\[
\begin{align}
R 
&= R_z(z_A,\alpha)R_y(y_A,\beta)R_x(x_A,\gamma) \\\\\\
&= 
\begin{bmatrix}
\cos\alpha & -\sin\alpha & 0 \\\\\\
\sin\alpha & \cos\alpha & 0 \\\\\\
0 & 0 & 1 
\end{bmatrix}
\begin{bmatrix}
\cos\beta  & 0 & \sin\beta \\\\\\
0 & 1 & 0 \\\\\\
-\sin\beta  & 0 & \cos\beta 
\end{bmatrix}
\begin{bmatrix}
1 & 0 & 0 \\\\\\
0 & \cos\gamma & -\sin\gamma \\\\\\
0 & \sin\gamma & \cos\gamma  
\end{bmatrix} 
\end{align}
\\]
对于任何一个旋转矩阵这个分解是唯一的，既然如此，那么我们可以利用分解出来的三个角度\\(\alpha \beta \gamma​\\)来描述空间旋转了。这个三个角度就是广为人知的欧拉角了，虽然这个旋转矩阵的分解是唯一的，分解的过程可以用两种不同的理解方式。

这里我们考虑原点重合但坐标轴不重合的空间坐标系\\(A​\\)和坐标系\\(B​\\)，我们的目的是旋转坐标系B使得坐标系B与坐标系\\(A​\\)完全重合。这个过程可以用下列公式来描述：
\\[
R _B^AC = R_z(z_A,\alpha)R_y(y_A,\beta)R_x(x_A,\gamma) _B^AC
\\]

\\[
_B^AC = 
\begin{bmatrix}
^Ai_B & ^Aj_B & ^Ak_B
\end{bmatrix}
\\]

这里矩阵\\(_B^AC​\\)是一个单位正交阵，其三个列向量为单位向量，表示在坐标系\\(A​\\)下坐标系\\(B​\\)的三个轴的方向。矩阵\\(R_z(z_A,\alpha)R_y(y_A,\beta)R_x(x_A,\gamma)​\\)作用于矩阵\\(_B^AC​\\)有两种不同的结合方式：
\\[
R _B^AC = R_z(z_A,\alpha) \overbrace{R_y(y_A,\beta) \underbrace{R_x(x_A,\gamma) _B^AC}_1}^2
\\]

\\[
R _B^AC = \underbrace{ R_z(z_A,\alpha) \overbrace{R_y(y_A,\beta) R_x(x_A,\gamma) _B^AC}^2}_1
\\]

第一种方式很直观，遵循**从右到左**的法则，矩阵结合的方式可以理解为先坐标系\\(B​\\)(三个轴)绕着坐标系\\(A​\\)的X轴转\\(\gamma​\\)角度，再绕着坐标系\\(A​\\)的Y轴转\\(\beta​\\)角度，最后再绕着坐标系A的Z轴转\\(\alpha​\\)角度。

第二种方式不容易理解，遵循**从左到右**的法则矩阵结合的方式可以理解为坐标系\\(B\\)(三个轴)绕着坐标系\\(B\\)的Z轴转\\(\alpha\\)角度，再绕着新的坐标系\\(B_1\\)的Y轴转\\(\beta\\)角度，最后再绕着新的坐标系\\(B_2\\)的X轴转\\(\gamma\\)角度。

这里引用维基百科上的一个图来解释第二种方式，这张图表达的旋转是绕自身坐标系旋转(坐标系B)，只是这里旋转只用到了自身坐标系的两个轴。后面我们会看到工作原理是一样的，只是欧拉角分解的方式不同。

![euler2](euler2.gif)

第一种理解方式是绕着静态坐标系A旋转的，旋转轴实际上是\\((1,0,0)(0,1,0)(0,0,1)\\)这三个轴；而第二种理解方式是绕着动态坐标系B旋转的，在坐标系B的旋转过程中，这个动态坐标系会不断改变。

换一种角度考虑，我们刚刚的讨论都是绕着坐标系的三个轴进行旋转的，那么如果要绕坐标系中的任意一个向量\\(k=(l,m,n)\\)何进行旋转呢？思路是首先将这个向量\\(k\\)利用仅有的三种旋转方式使之与任意一个坐标轴重合，连同要旋转的那个量也一起旋转过去，然后在那完成我们需要的旋转，最后怎么旋转过来的就怎么旋转回去。利用这个思路可以将坐标轴B动态旋转的过程写成下式：
\\[
R _B^AC =  
\underbrace{R_x(x_A,\gamma)}_3
\underbrace{R_x(x_A,-\gamma) R_y(y_A,\beta) R_x(x_A,\gamma)}_2 \\\\\\
\underbrace{R_x(x_A,-\gamma) R_y(y_A,-\beta) R_z(z_A,\alpha) R_y(y_A,\beta) R_x(x_A,\gamma)}_1 {_B^AC}
\\]
第一步使坐标系B的Z轴与坐标系A的Z轴重合，绕着坐标系A(B)的Z轴转\\(\alpha\\)角度，然后再旋回去；第二步使坐标系B的Y轴与坐标系A的Y轴重合，绕着坐标系A(B)的Y轴旋转\\(\beta\\)角度，然后再旋回去；第三步绕着坐标系A(B)的X轴旋转。这个过程有点儿绕，好吧不是一点点绕，是很绕，读者可以停下来拿笔画一画来理解。

有些文献中根据这两种不同的理解方式来定义欧拉角，绕固定坐标系A旋转称为**显式定义(extrinsic definition)**，绕旋转坐标系B旋转称为**隐式定义(intrinsic definition)**。

之前的讨论旋转矩阵\\(R\\)的分解方式是按照\\(R_zR_yR_x \\)的顺序分解的，但实际上对\\(R_zR_yR_x\\)这三个矩阵进行组合有6种不同的分解方法。除此之外，如果我们允许只用到两个坐标轴进行旋转的话还可以有\\(R_zR_yR_z\\)这种类型的组合，这种类型的组合也有6种，比如下面这种：
\\[
\begin{align}
R 
&= R_z(\alpha)R_y(\beta)R_z(\gamma) \\\\\\
&= \begin{bmatrix}
\cos\alpha & -\sin\alpha & 0 \\\\\\
\sin\alpha & \cos\alpha & 0 \\\\\\
0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
\cos\beta  & 0 & \sin\beta \\\\\\
0 & 1 & 0 \\\\\\
-\sin\beta  & 0 & \cos\beta 
\end{bmatrix}
\begin{bmatrix}
1 & 0 & 0 \\\\\\
0 & \cos\gamma & -\sin\gamma \\\\\\
0 & \sin\gamma & \cos\gamma  
\end{bmatrix} 
\end{align}
\\]
读者可以自行验算。这样的话欧拉角就有12种分解方法了。

很多文献会提到这几个概念：欧拉角、泰特布莱恩角、RPY角。实际上它们看作是欧拉角的子集，本文中欧拉角指的的广义的欧拉角，也就是包含了所有12种情况。**泰特布莱恩角(Tait–Bryan angles)**指的是绕3个不同的旋转轴进行旋转，而**纯欧拉角(Pure Euler angles)**指的是旋转过程只涉及两个旋转轴。**RPY角**是英文**Roll, Pitch, Yaw**的缩写，它用于描述飞行器的姿态。

这里以z-y-x 欧拉角为例来介绍欧拉角与矩阵的转换问题，其他的情况类似。

#### 欧拉角变换为旋转矩阵

将\\(\alpha \beta \gamma\\)带入公式中，将矩阵相乘记即可：
\\[
R = 
\begin{bmatrix}
\cos\alpha & -\sin\alpha & 0 \\\\\\
\sin\alpha & \cos\alpha & 0 \\\\\\
0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
\cos\beta  & 0 & \sin\beta \\\\\\
0 & 1 & 0 \\\\\\
-\sin\beta  & 0 & \cos\beta \\
\end{bmatrix}
\begin{bmatrix}
1 & 0 & 0 \\\\\\
0 & \cos\gamma & -\sin\gamma \\\\\\
0 & \sin\gamma & \cos\gamma  
\end{bmatrix} 
\\]

#### 旋转矩阵变换为欧拉角

将上式相乘得到，
\\[
R =
\begin{bmatrix}
r_{11} & r_{12} & r_{13} \\\\\\
r_{21} & r_{22} & r_{23} \\\\\\
r_{31} & r_{32} & r_{33} 
\end{bmatrix}
=
\begin{bmatrix}
\cos\alpha\cos\beta & * & * \\\\\\
\sin\alpha\cos\beta & * & * \\\\\\
-\sin\beta & \cos\beta\sin\gamma & \cos\beta\cos\gamma
\end{bmatrix}
\\]
当\\(\cos\beta \neq 0​\\)时，不难得到：
\\[
\beta = -\arcsin r_{31}
\\]

\\[
\alpha = atan2 (\frac{r_{21}}{\cos\beta},\frac{r_{11}}{\cos\beta})
\\]

\\[
\gamma = atan2 (\frac{r_{32}}{\cos\beta},\frac{r_{31}}{\cos\beta})
\\]

当\\(\cos\beta = 0​\\)时，需要用到带*号的那几个式子来计算，读者可以自行计算下，此处省略。

最后结论是当\\(\beta = \frac{\pi}{2}​\\)时，
\\[
\gamma - \alpha = atan2(r_{12},r_{13})
\\]
当\\(\beta = -\frac{\pi}{2}\\)时，
\\[
\gamma + \alpha = atan2(-r_{12},-r_{13})
\\]

#### 万向锁

万向锁(Gimbal Lock)问题是欧拉角一个硬伤，网上关于它的介绍非常多。它对应的数学描述是：当\\(\cos\beta = 0\\)是，欧拉角的解\\(\alpha\\)与\\(\gamma\\)无穷多个，这就是刚刚上文推导的公式。以z-y-x 欧拉角为例，从一种不严谨的角度来理解，当\\(\cos\beta = 0\\)时，运动坐标系B绕自身的Y轴转过\\(\beta\\)角度(\\(\frac{\pi}{2}\\)或\\(\frac{-\pi}{2}\\))，这时绕X轴与

绕Z轴能达到的效果是一样的。把绕XYZ三个轴旋转单位角度看作是旋转变换的一组基，正常情况下这组即线性无关，但当\\(\cos\beta = 0​\\)会让这组基线性相关，这就是万向锁的数学本质。前文中，我们提到了纯欧拉角，只需绕不同的两个轴旋转便可以到达任意位置，这是因为它们的三次旋转也构成了一组线性无关的基。

![gimbal lock](gimbal lock.gif)



### 3 旋转向量

旋转向量表示(Rotation representation)又名轴角表示(Axis-angle representation)。对于空间中两个原点重合的坐标系\\(A​\\)和\\(B​\\)，为了得到\\(P​\\)点在坐标系\\(A​\\)下的坐标\\(^A P​\\)只需将\\(_B^AR​\\)左乘\\(^B A​\\)。这个过程可以看作是利用某种操作将两个坐标系重合，因此描述两个坐标系之间的变换的还可以用旋转向量，将点\\(P​\\)与坐标系\\(A​\\)的三个轴整体绕着向量\\(^A r​\\)旋转一个角度可以得到同样的结果。

旋转向量\\(r\\)有3个参数，这个向量代表旋转轴的方向，而其模长代表旋转的角度(一般是弧度制)。

那么对于对于空间中两个原点重合的坐标系\\(A\\)和\\(B\\)，将坐标系\\(B\\)旋转得到坐标系\\(A\\)的旋转向量是否唯一？ 考虑一个简单的例子，对于一个平放在桌面上的一本书，绕垂直于桌面的轴顺时针旋转90度与逆时针旋转270度得到的书的摆放位置是一样的。所以，答案是不唯一。

实际上，全体旋转向量的集合在空间中是一个半径为\\(\pi\\)的实心球：
\\[
\{ r\lvert  \Vert r\Vert \le \pi\}
\\]
任何一个旋转矩阵都对应于这个集合的两个元素，为了消除二义性，可以将旋转角度的范围定义为\\([0,\pi]\\)之间，将这个集合一分为二，得到新的集合几何描述为空间中一个半球和平面上一个半圆，如下图所示。

![2](2.png)



给定一个旋转向量\\(r\\)，我们很容易可以求得旋转角度\\(\theta\\)和单位旋转轴的方向\\(u\\)；
\\[
u = \frac{r}{||r||}
\\]

\\[
\theta = ||r||
\\]

将旋转向量转换为旋转矩阵称为罗德里格(Rodrigues)公式，反过来将旋转矩阵转换为旋转向量称为罗德里格逆公式。

#### 罗德里格公式

罗德里格公式有两种推导方式，几何推导和代数推导，代数推导过程十分繁琐，要进行大量的三角学的公式变换，有兴趣的读者可以参考文献，这里介绍几何推导。

假设我们向量\\(p\\)绕着向量\\(u\\)旋转\\(\theta\\)角度，将向量\\(p\\)分解为平行于\\(u\\)的分量\\(a\\)和垂直于\\(u\\)的分量\\(b\\)：
\\[
p = a + b
\\]
旋转后的向量为：
\\[
p' = a + b'
\\]
利用下图的几何关系很容易得到下列关系：
\\[
a = uu^Tp
\\]

\\[
b = p - a  = (I - uu^T)p
\\]

\\[
c = u \times p
\\]

\\[
b' = b\cos\theta + c\sin\theta
\\]

由以上几式可以得到式(35)，该公式便是著名的罗德里格公式。
\\[
p' = [I\cos\theta + (1-\cos\theta)uu^T + u_{\times}\sin\theta]p
\\]
![3](3.png)

#### 罗德里格逆公式

上面通过几何推导得到了罗德里格斯公式：
\\[
R = I\cos\theta + (1-\cos\theta)uu^T + u_{\times}\sin\theta
\\]
注意到罗德里格公式，前面两项\\(I\cos\theta + (1-\cos\theta)uu^T\\)为对称矩阵，后面一项\\(u_{\times}\sin\theta\\)为反对称矩阵。利用一些矩阵的性质，可以得到，
\\[
trace(R) = 2\cos\theta + 1
\\]

\\[
B 
= \frac{R - R^T}{2} = u_{\times}\sin\theta 
= \sin\theta \begin{bmatrix}
0 & -u_3 & u_2 \\\\\\
u_3 & 0 & -u_1 \\\\\\
-u_2 & u_1 & 0
\end{bmatrix}
\\]

进而，
\\[
\cos\theta = \frac{r_{11} + r_{22} + r_{33}}{2}
\\]

\\[
\sin\theta = \sqrt{b_{12}^2 + b_{13}^2 + b_{23}^2}
\\]

(1)当\\(\sin\theta \neq 0\\)时

我们可以由式(38)得到单位化的\\(u\\)，由式(39)和式(40)得到\\(\theta\\)的值，那么：
\\[
r = u\theta
\\]
(2)当\\(\sin\theta = 0\\)以及\\(\cos\theta = 1\\)时
\\[
r = 0
\\]
(3)当\\(\sin\theta = 0\\)以及\\(\cos\theta = -1\\)时

罗德里格公式简化为：
\\[
R = -I + 2uu^T
\\]
因此，
\\[
uu^T = \frac{R+I}{2}
\\]
令\\(v\\)为\\(\frac{R+I}{2}\\)的任意一列，将\\(v\\)单位化得到\\(u\\)，
\\[
u = \frac{v}{||v||}
\\]
最后得到：
\\[
r = u\pi
\\]



### 4 四元数

哈密尔顿发现四元数时，绝对没想到自己的成果会在100年后的计算机视觉、计算机图形学领域内应用如此广泛，其实很多人也是因此才知道哈密尔顿苦思数十年才悟到四元数的故事。其实四元数描述与轴角描述有着千丝万缕的联系，只是因为四元数的晦涩难懂给空间旋转的四元数描述蒙上了一层神秘的面纱。

这里首先简单介绍四元数的基本概念，再利用四元数描述空间旋转，最后探索四元数描述与轴角描述的联系与区别。

#### 四元数的基本概念

**四元数的定义**

四元数定义为：
\\[
q = s + ai + bj + ck \quad s,a,b,c \in  \mathbb{R} 
\\]
并满足下列运算规则，
\\[
{i}^2 = {j}^2 = {k}^2 = {ijk} = -1
\\]

\\[
{ij} = {k},\quad {jk} = {i},\quad {ki} = {j}
\\]

我们发现i,j,k与矢量的\\(\mathbf{i},\mathbf{j},\mathbf{k}\\)运算规则是一样的，
\\[
\mathbf{i} \times \mathbf{j} = \mathbf{k}, \quad 
\mathbf{j} \times \mathbf{k} = \mathbf{i}, \quad
\mathbf{k} \times \mathbf{i} = \mathbf{j}
\\]
![4](4.png)

我们也可以用有序对的形式，来表示四元数，
\\[
[s,\mathbf {v}] \quad  s \in \mathbb{R},v \in \mathbb{R^{3}}
\\]
或者写成，
\\[
q = [s,x\mathbf i + y\mathbf j + z\mathbf k] \quad s,x,y,z\in \mathbb{R}
\\]
**四元数的加减**
\\[
q_{a} = [s_{a},\mathbf {a}]  \quad q_{b} = [s_{b},\mathbf {b}] \\\\\\
q_{a} + q_{b} = [s_{a} + s_{b},\mathbf {a} + \mathbf {b}] \\\\\\
q_{a} - q_{b} = [s_{a} - s_{b},\mathbf {a} - \mathbf {b}]
\\]
**四元数与四元数的积**
\\[
q_{a}q_{b} 
= [s_{a},\mathbf {a}][s_{b},\mathbf {b}]
= [s_{a} + x_{a}\mathbf i + y_{a}\mathbf j +z_{a}\mathbf k][s_{b} + x_{b}\mathbf i + y_{b}\mathbf j +z_{b}\mathbf k]
\\]
将其展开化简可以得到：
\\[
q_{a}q_{b} = [s_{a}s_{b} - \mathbf a\cdot \mathbf b, s_{a}\mathbf b + s_{b}\mathbf a + \mathbf a\times \mathbf b]
\\]
**四元数与标量的积**
\\[
q = [s,\mathbf{v}] \\\\
\lambda q = \lambda[s,\mathbf{v}] = [\lambda s,\lambda \mathbf{v}]
\\]
**单位四元数**

当s=0时，可以定义单位四元数为：
\\[
\hat {q} = [0, \mathbf { \hat { \mathbf v } }] \quad | \hat { \mathbf v } | = 1
\\]
**四元数的共轭**

将四元数的虚向量取反：
\\[
q^{*} = [s,-\mathbf {v}]
\\]
**四元数的范数**

我们可以定义四元数的范数为：
\\[
|\mathbf q| = \sqrt {s^{2} + v^{2}}
\\]
**四元数单位化**

有了范数便可以单位化四元数了，
\\[
\mathbf q' = \frac {\mathbf q}{\sqrt {s^{2}+v^{2}}}
\\]
**四元数的逆**
\\[
\mathbf q^{-1} = \frac {\mathbf q^{*}}{|\mathbf q|^{2} }
\\]
这个式子不难证明，只需根据逆元的定义即可。

**四元数的点积**

类似向量的点积，我们可以定义四元数的点积为：
\\[
cos\theta = \frac {s_{1}s_{2}+x_{1}x_{2}+y_{1}y_{2}+z_{1}z_{2}}{|\mathbf q_{1}||\mathbf q_{2}|}
\\]

#### 四元数与旋转

上文简要介绍了四元数的定义以及一些基本的运算规则，下面我们将看到四元数是如何巧妙地运用在空间旋转中。

**平面旋转**

在介绍空间旋转前，我们先考虑平面上的旋转。考虑平面上一点\\(p=(a,b)^T\\)，使其绕原点逆时针旋转\\(\theta​\\)角度。

我们定义复数p为：
\\[
p = a + b \mathbf{i}
\\]
定义转子为：
\\[
q = \cos\theta + \mathbf{i}\theta
\\]
那么旋转的过程可以表示为：
\\[
p' = qp = (a + b \mathbf{i}) (\cos\theta + \mathbf{i}\theta) 
\\]
得到新的复数为：
\\[
a' + b'\mathbf i = acos\theta -bsin\theta + (asin\theta +bcos\theta )\mathbf i
\\]
这个过程也可以用矩阵与向量相乘表达为：
\\[
\left[ 
\begin{matrix} 
a' & -b'\\\\\\
b' & a'
\end{matrix} 
\right] 
= \left[
\begin{matrix} 
cos\theta & -sin\theta \\\\\\
sin\theta & cos\theta 
\end{matrix} 
\right] 
\left[
\begin{matrix} 
a & -b \\\\\\
b & a
\end{matrix} 
\right]
\\]
啊哈，平面旋转竟然可以用复数来描述，很神奇吧。

**空间旋转**

既然平面旋转可以用复数描述，那么空间旋转也应该可以复数在三维空间的推广来描述，这个数学工具就是四元数。

这里给出公式，空间上一点\\(p=(a,b,c)\\)绕向量\\(v\\)旋转角度\\(\theta\\)的四元数表达为：
\\[
\begin{align}
qpq^{-1} 
& = [\cos\frac{1}{2}\theta,\sin\frac{1}{2}\theta \hat{\mathbf{v}}]
[0,p]
[ \cos\frac{1}{2}\theta,-\sin\frac{1}{2}\theta \hat{\mathbf{v}}] \\\\\\
& = [0,(1-\cos\theta) \hat{\mathbf{v}} \hat{\mathbf{v}}^T \mathbf{p} 
+ \cos\theta \mathbf{p} + \sin\theta \hat{\mathbf{v}} \times \mathbf{p} ]
\end{align}
\\]
这里的\\(\hat{\mathbf{v}}\\)是单位化后的向量\\(v\\)，读者可以利用四元数的乘法将其展开，这里惊喜地发现得到的结果与罗德里格公式一模一样，是的没错！

#### 四元数变换为旋转矩阵

为了寻找四元数与旋转矩阵的关系，我们得将四元数与四元数相乘的过程用矩阵来描述。回忆前文中，复数与复数相乘可以用矩阵来描述，这个推广过去完成不成问题。
\\[
\begin{align}
q_{a}q_{b} 
& = [s_{a} + x_{a}\mathbf i + y_{a}\mathbf j +z_{a}\mathbf k \][s_{b} + x_{b}\mathbf i + y_{b}\mathbf j +z_{b}\mathbf k \] \\\\\\
& = \begin{bmatrix}
s_a & -x_a & -y_a & -z_a \\\\\\
x_a & s_a & -z_a & y_a \\\\\\
y_a & z_a & s_a & -x_a \\\\\\
z_a & -y_a & x_a & s_a
\end{bmatrix} = Aq_b \\\\\\
& = \begin{bmatrix}
s_b & -x_b & -y_b & -z_b \\\\\\
x_b & s_b & z_b & -y_b \\\\\\
-y_b & z_b & s_b & x_b \\\\\\
z_b & -y_b & x_b & s_b
\end{bmatrix} = q_aB
\end{align}
\\]
利用上面两式，可以得到：

\\[
qpq^{-1} 
= Rp 
= \begin{bmatrix}
2(s^2+x^2)-1 & 2(xy-sz) & 2(xz+sy) \\\\\\
2(xy+sz) & 2(s^2+y^2)-1 & 2(yz+sx) \\\\\\
2(xz-sy) & 2(yz+sx) & 2(s^2+z^2)-1
\end{bmatrix} p
\\]

#### 旋转矩阵变换为四元数

\\[
s = \pm \frac{1}{2} \sqrt{1 + r_{11} + r_{22} + r_{33}}
\\]

(1)当\\(s=0\\)时

思路和罗德里格逆公式一样，任取旋转矩阵的任意一列单位化即可。

(2)当\\(s \neq 0\\)时
\\[
\begin{align}
s = \pm \frac{1}{2} \sqrt{1 + r_{11} + r_{22} + r_{33}} \\\\\\
x = \frac{1}{4s}(r_{32} - r_{23}) \\\\\\
y = \frac{1}{4s}(r_{13} - r_{31}) \\\\\\
z = \frac{1}{4s}(r_{21} - r_{12}) \\\\\\
\end{align}
\\]

#### 四元数与旋转向量

我们可以看到，四元数与旋转矩阵的转换和旋转向量与旋转矩阵的转换公式十分相似，很多公式推导都是一样的。都是绕空间中一个轴旋转一个角度，这里面需要四个量，只不过这两种表达对四个量的处理方式不同。实际上罗德里格在哈密尔顿发现四元数之前就找到了四元数表达旋转的系数；而哈密尔顿发现四元数只是想推广复数，没考虑过利用四元数来描述空间旋转，直到最近几十年在计算机图形学中为了对空间旋转进行插值才发现四元数是如此好用。



### 5 总结

本文详细介绍了空间中旋转平移的描述方法，其中空间平移描述十分简单，只涉及x,y,z三个量，而空间旋转描述比较复杂，本文介绍了旋转矩阵、欧拉角、旋转向量以及四元数四种描述方法，及其这些描述之间的转换关系。

这些转换关系非常重要，因为不同的场合需要用到不同的描述方法，比如欧拉角比较直观，常常用于表达描述两个坐标系直接的，但四元数表达便于插值，机器人内部的运动解算实际上是利用四元数表达的。

这些转换关系如下图所示，红色箭头的转换公式本文已经详细介绍了。四元数与旋转向量的转化十分简单，这里就不赘述了；欧拉角与旋转向量或四元数的转化公式复杂些(这里就不介绍了，公式太难打了)，可以直接转化也可以利用旋转矩阵作为中介来实现。

![convert](convert.png)



### 参考(Reference)

维基百科[1]给出了关于欧拉角很全面的介绍，[2]是介绍了欧拉角与旋转矩阵如果转换，[4]很细致地介绍了旋转向量表示法。关于四元数的这部分稍稍复杂些，[5]和[6]是关于四元数入门极好的入门资料，虽然[5]篇幅很长，但是相信我一两天就能看完，[5]中还介绍了四元数的历史，十分有趣。[6]是一篇博客，在网上能找到与其对应的中文翻译。[7]是一篇四元数的技术报告，十分深入地探讨了四元数的数学基础与在旋转表达方面的应用。

[[1]Wikipedia: Euler angles](https://en.wikipedia.org/wiki/Euler_angles)

[[2]Gregory G. Slabaugh: Computing Euler angles from a rotation matrix](http://www.gregslabaugh.net/publications/euler.pdf)

[[3]Learning opencv: Rotation Matrix To Euler Angles](http://www.learnopencv.com/rotation-matrix-to-euler-angles/)

[[4]Carlo Tomasi: Vector Representation of Rotations](http://www.learnopencv.com/rotation-matrix-to-euler-angles/)

[[5]John Vince: Quaternions for Computer Graphics](https://book.douban.com/subject/6831294/)

[[6]Jeremiah van Oosten: Understanding Quaternions](https://www.3dgep.com/understanding-quaternions/)

[[7]Erik B. Dam: Quaternions, Interpolation and Animation](http://web.mit.edu/2.998/www/QuaternionReport1.pdf)

