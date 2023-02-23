# 1. 单变量线性回归
## 1.1 模型表示（Model Representation）
### 1.11房价预测训练集
|Size in $feet^2(x)$ |Price(\$) in 1000$'$s $y$|
:-:|:-:
2104 |460
1416|231
1534|315
952|189
$\cdots$|$\cdots$
1.12 Hypothesis假设函数
>$h_\theta = \theta_0+\theta_1x$

1.13 Cost Function
**损失函数**：计算**单个**样本的误差
**代价函数**：计算整个训练集**所有损失函数之和的平均值**  
>代价函数：
$$J(\theta_0,\theta_1)=\frac{1}{2m}\sum_{i=1}^m(h_\theta(i)-y_i)^2$$
问题$\implies$求解$\sum_{i=0}^m(h_\theta(x^{(i)})-y^{(i)})$的最小值
$m$:训练集中的样本总数
$y$:目标变量/输出变量
$(x,y)$:训练集中的实例
$(x^{(i)},y{(i)})$:训练集中的第$i$个样本实例
---
## 1.2 梯度下降
### 1.21 线性回归模型
* $$h_\theta(x)=\theta_0+\theta_1x$$
* $$J(\theta_0,\theta_1)=\frac{1}{2m}\sum_{i=1}^m(h_\theta(x^{(i)}-y^{(i)}))^2$$ 
>梯度下降算法：
Repeat until convergence：{
    $$\theta_j:=\theta_j-\alpha\frac{\partial}{\partial\theta_j}J(\theta_0,\theta_1)$$}

当 $j = 0, j = 1$ 时，线性回归中代价函数求导的推导过程（看不懂请查阅导数基本公式）： $$ \begin{align*} \frac{\partial}{\partial\theta_j} J(\theta_1, \theta_2)&=\frac{\partial}{\partial\theta_j} \left(\frac{1}{2m}\sum\limits_{i=1}^{m}{{\left( {{h}_{\theta }}\left( {{x}^{(i)}} \right)-{{y}^{(i)}} \right)}^{2}} \right)\\ &=\left(\frac{1}{2m}*2\sum\limits_{i=1}^{m}{{\left( {{h}_{\theta }}\left( {{x}^{(i)}} \right)-{{y}^{(i)}} \right)}} \right)*\frac{\partial}{\partial\theta_j}{{\left( {{h}_{\theta }}\left( {{x}^{(i)}} \right)-{{y}^{(i)}} \right)}}\\ &=\left(\frac{1}{m}\sum\limits_{i=1}^{m}{{\left( {{h}_{\theta }}\left( {{x}^{(i)}} \right)-{{y}^{(i)}} \right)}} \right)*\frac{\partial}{\partial\theta_j}{{\left(\theta_0{x_0^{(i)}} + \theta_1{x_1^{(i)}}-{{y}^{(i)}} \right)}} \end{align*} $$

所以当 $j = 0$ 时：

$$ \frac{\partial}{\partial\theta_0} J(\theta)=\frac{1}{m}\sum\limits_{i=1}^{m}{{\left( {{h}_{\theta }}\left( {{x}^{(i)}} \right)-{{y}^{(i)}} \right)}} *x_0^{(i)} $$

所以当 $j = 1$ 时：

$$ \frac{\partial}{\partial\theta_1} J(\theta)=\frac{1}{m}\sum\limits_{i=1}^{m}{{\left( {{h}_{\theta }}\left( {{x}^{(i)}} \right)-{{y}^{(i)}} \right)}} *x_1^{(i)} $$ 上文中所提到的梯度下降，都为批量梯度下降（Batch Gradient Descent），即每次计算都使用所有的数据集 $\left(\sum\limits_{i=1}^{m}\right)$ 更新。

>由于线性回归函数呈现碗状，且只有一个全局的最优值，所以函数一定总会收敛到全局最小值（学习速率不可过大）。同时，函数 $J$ 被称为凸二次函数，而线性回归函数求解最小值问题属于凸函数优化问题。    
>另外，使用循环求解，代码较为冗余，后面会讲到如何使用向量化（Vectorization）来简化代码并优化计算，使梯度下降运行的更快更好。