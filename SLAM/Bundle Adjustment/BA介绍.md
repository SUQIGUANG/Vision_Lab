# 1. 起源

Adjustment computation最早是由geodesy的人搞出来的。19世纪中期的时候，geodetics的学者就开始研究large scale triangulations了。20世纪中期，随着camera和computer的出现，photogrammetry也开始研究adjustment computation，所以他们给起了个名字叫bundle adjustment，bundle的意思我不知道中文怎么解释好，大家意会吧。20世纪下半段，computer vision community开始兴起reconstruction，也开始研究bundle adjustment，一开始重复造轮子，后来triggs的modern synthesis及时出现。21世纪前后，robotics领域开始兴起SLAM，最早用的recursive bayesian filter，后来把问题搞成个graph然后用least squares方法解。

这些东西归根结底就是Gauss大神“发明”的least squares method。当年天文学家Piazzi整天闲得没事看星星，在1801年1月1号早上发现了一个从来没观测到的星星，再接下来的42天里做了19次观测之后这个星星就消失了。当时的天文学家为了确定这玩意到底是什么绞尽了脑汁，这时候Gauss出现了，（最初）只用了3个观察数据，就用least squares算出了这个小行星的轨道，接下来天文学家根据Gauss的预测，也重新发现了这个小行星（虽然有小小的偏差），并将其命名为Ceres，也就是谷神星。Google的[**Ceres-solver**](https://github.com/ceres-solver/ceres-solver)就是根据这个来命名的。  

> Ceres solver 是谷歌开发的一款用于非线性优化的库，在谷歌的开源激光雷达slam项目cartographer中被大量使用。Ceres官网上的文档非常详细地介绍了其具体使用方法，相比于另外一个在slam中被广泛使用的图优化库G2O，ceres的文档可谓相当丰富详细。

关于究竟是谁发明了Least Squares历史上有争论，Legendre是最早publish这个方法的（1805），但是几年后（1809）Gauss跳出来说：“你太naive了，我1795年就开始用least squares了，微不足道”。虽然有人认为Gauss这样的大神没必要说谎来和Legendre这种小叼丝（相对，长了一副反派脸）抢成果，但是至今没有definitive的证据证明确实是Gauss最早发明Least Squares。有人认为least squares的方法对Gauss来说太简单以至于Gauss根本没觉得有必要把它publish出来。

我们再来看看19，20世纪连电脑都没有的情况下，geodesy的人们面对的是什么样的问题吧。1927年的North American Datum有25000个观测塔，1983年的NAD有270000个观测塔。再来看看robotics community，最大的real-world SLAM datasets有21,000个pose[Johannsson13]，最大的simulation datasets有200,000个pose[Grisetti10]。看看时间，都是2010年以后的。拿NAD83来说，虽然1983年已经有电脑了，但是优化这样一个network，需要解1,800,000个equations。也就是说如果如果用Least Squares的话，每一步的normal equation $J^T * J * x = J^T y$ 或者 $Ax = b$，里面这个A的dimension是900,000 x 900,000。用dense method光是存这么一个matrix就要3000 Gb[Agarwal14]。

# 2. 问题描述

在SFM（structure from motion）的计算中BA（Bundle Adjustment，可译为**光束法平差**）作为最后一步优化具有很重要的作用，在近几年兴起的基于图的SLAM（simultaneous localization and mapping）算法里面使用了图优化替代了原来的滤波器，这里所谓的图优化其实也是指BA。所谓bundle，来源于bundle of light，其本意就是指的光束，这些光束指的是三维空间中的点投影到像平面上的光束，而重投影误差正是利用这些光束来构建的，因此称为光束法强调光束也正是描述其优化模型是如何建立的。剩下的就是平差，那什么是平差呢？借用一下百度词条 **测量平差** 中的解释吧。

> 由于测量仪器的精度不完善和人为因素及外界条件的影响，测量误差总是不可避免的。为了提高成果的质量，处理好这些测量中存在的误差问题，观测值的个数往往要多于确定未知量所必须观测的个数，也就是要进行多余观测。有了多余观测，势必在观测结果之间产生矛盾，测量平差的目的就在于消除这些矛盾而求得观测量的最可靠结果并评定测量成果的精度。测量平差采用的原理就是“最小二乘法”。

用一句话来描述BA那就是，**BA的本质是一个优化模型，其目的是最小化重投影误差**

---
**那么什么是重投影误差呢？**
![enter image description here](http://ww1.sinaimg.cn/large/8c1d6d59gy1fmml45w77sj20gm0ctmxs.jpg)

这些五颜六色的线就是我们讲的光束。那现在就该说下什么叫重投影误差了，重投影也就是指的第二次投影，那到底是怎么投影的呢？我们来整理一下吧：

 - 其实第一次投影指的就是相机在拍照的时候三维空间点投影到图像上
 - 然后我们利用这些图像对一些特征点进行三角定位
 - 最后利用我们计算得到的三维点的坐标（注意不是真实的）和我们计算得到的相机矩阵（当然也不是真实的）进行第二次投影，也就是重投影

现在我们知道什么是重投影了，那重投影误差到底是什么样的误差呢？这个误差是指的真实三维空间点在图像平面上的投影（也就是图像上的像素点）和重投影（其实是用我们的计算值得到的虚拟的像素点）的差值，因为种种原因计算得到的值和实际情况不会完全相符，也就是这个差值不可能恰好为0，此时也就需要将这些差值的和最小化获取最优的相机参数及三维空间点的坐标。

---

Bundle adjustment优化的是sum of reprojection error，这是一个geometric distance（为什么要minimize geometric distance可以参考[Hartley00]），问题可以formulate成一个least squares problem， 如果nosie是gaussian的话，那就是一个maximum likelihood estimator，是这种情况下所能得到的最优解了。

这个reprojection error的公式是非线性的，所以这个least squares problem得用iterative method来求解。最简单的是Newton， 但是要算Hessian，并不是很好算，所以pass。接下来是Gauss-Newton，用$J^T J$ 来近似Hessian，但是convergence速度不给力，也pass。再下来是Levenberg-Marquardt，是一个damping method，改一个lambda来控制到底是偏向steepest descent还是偏向Gauss-Newton，如果算出来更渣的一步就不接受，改个lambda重算，这样一来做了很多无用功，所以也pass。再下来还有Powell's dogleg，是一个trust region method，在这个小区间内搞一个新的function来近objective function，不论好坏都走一步，但是步幅不会超过所谓的trust region，再根据表现调整这个region，总的来说在large scale问题上比Levenberg-Marquardt的表现要好。再往后问题再大就得用前面所说的算法的sparse版本，再大下去得换conjugate gradient方法，这块我就不怎么了解了。

不论GN,LM,DL，中间都要解一个$Ax=b$形式的linear system，一般情况下算法的效率就取决于解这个linear system的效率。所以**说到底这些nonlinear least squares problem最后也就是解一个linear system**。这个linear system你可以**直接inverse，也可以用QR，或者Choleskey**，或者**Schur complement trick**来解，爱谁谁。说到这个Choleskey decomposition，当初就是为了Geodetic mapping而发明的。

现实中，并不是所有observation都服从 gaussian noise的（或者可以说几乎没有），遇到有outlier的情况，这些方法非常容易挂掉，这时候就得用到robust statistics里面的robust cost (*cost也可以叫做loss, 统计学那边喜欢叫risk) function了，比较常用的有huber, cauchy等等。

其实least square problem一般都是用Gauss-Newton 法或者LM算法迭代求解。bundle adjustment本质也是lm算法。由于是特定的形式，所以可以化成sparse matrix 的形式，这样计算量大大减小了。

总的来说bundle adjustment这个东西搞了这么多年也搞得差不多了，不能说state-of-the-art，但是可以算是gold standard。大点的问题诸如earth-scale reconstruction也被google的人搞过了，用了2 billion张谷歌街景，reconstruct了整个地球，真不知道接下来还能reconstruct什么。

数学计算可参考[CSDN博文](https://blog.csdn.net/OptSolution/article/details/64442962#_39)

---
Bundle Adjustment的参考书和论文《[Bundle Adjustment - A Modern Synthesis](https://lear.inrialpes.fr/pubs/2000/TMHF00/Triggs-va99.pdf)》（比较全面的介绍）、《[Multiple View Geometry in Computer Vision](http://cvrs.whu.edu.cn/downloads/ebooks/Multiple%20View%20Geometry%20in%20Computer%20Vision%20%28Second%20Edition%29.pdf)》、《[Temporally scalable visual slam using a reduced pose graph](http://ais.informatik.uni-freiburg.de/longtermoperation_ws12rss/johannsson2012rss_ws.pdf)》、《[A tutorial on graph-based SLAM](http://www2.informatik.uni-freiburg.de/~stachnis/pdf/grisetti10titsmag.pdf)》及其[PPT](http://www.dis.uniroma1.it/~labrococo/tutorial_icra_2016/icra16_slam_tutorial_stachniss.pdf)


