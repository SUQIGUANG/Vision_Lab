参考：[ICP算法理解](https://blog.csdn.net/eric_e/article/details/80908162)、[ICP（迭代最近点）算法](https://www.cnblogs.com/21207-iHome/p/6038853.html)

## 1. 什么是ICP？
ICP是Iterative Closest Point迭代最近点的简称，主要用于求解两堆点云之间的变换关系。如下图所示，PR（红色点云）和RB（蓝色点云）是两个点集，该算法就是计算怎么把PB平移旋转，使PB和PR尽量重叠。
![enter image description here](https://note.youdao.com/yws/public/resource/51744046d92fd3b4b07cf72defc5ebf2/4B2643D7F568428EB8B79580CC564D3B?ynotemdtimestamp=1551862453977)

ICP算法一个经典的应用是场景的重建，比如说一张茶几上摆了很多杯具，用深度摄像机进行场景的扫描，通常不可能通过一次采集就将场景中的物体全部扫描完成，只能是获取场景不同角度的点云，然后将这些点云融合在一起，获得一个完整的场景。

## 2. 如何求解ICP问题
假设有两堆点云，分别记为两个集合$X=x_1,x_2,...,x_m$和$Y=y_1,y_2,...,y_m$(两个m并不一定相等)。假设这两堆点云的变换由R（旋转）和t（平移）描述，求解这个问题可以通过最小均方误差（LMS，Least Mean Square）：

$$
e(X,Y)=\sum_{i=1}^m(Rx_i+t-y_i)^2
$$

### 2.1 初始化
我们事先不知道R和t的值，那我们就“随便”赋一个值，这一步就叫初始化。但是“随便”也不是真的“随便”，初始值选的不好最后的结果可能就会收敛到局部最优解里。ICP在这么多年的发展历史中已经有很多方法来估计初始R与t了，PCL中是使用内置的SampleConsensusInitialAlignment函数以及TransformationEstimationSVD函数进行初始化。

### 2.2 迭代
得到初始化值后，就可以进行迭代了：对X和Y的每一个点用当前的R和t求解（二范数），逐渐优化R和t使误差降到允许范围内。

### 2.3 怎么收敛？
主要有四元数法、奇异值分解法等，下图是奇异值分解法进行求解。

![enter image description here](https://note.youdao.com/yws/public/resource/51744046d92fd3b4b07cf72defc5ebf2/6D0264C39DE346148952676937F804C7?ynotemdtimestamp=1551862453977)

如果X经变化后与Y“足够近”，则可保证收敛。

![enter image description here](https://note.youdao.com/yws/public/resource/51744046d92fd3b4b07cf72defc5ebf2/B4ABD5F5C22C493FA0468BC896E83042?ynotemdtimestamp=1551862453977)

## 3. ICP算法优缺点

**优点：**
- 可以获得非常精确的配准效果
- 不必对处理的点集进行分割和特征提取
- 在较好的初值情况下，可以得到很好的算法收敛性

**缺点：**
- 在搜索对应点的过程中，计算量非常大，这是传统ICP算法的瓶颈
- 标准ICP算法中寻找对应点时，认为欧氏距离最近的点就是对应点。这种假设有不合理之处，会产生一定数量的错误对应点。针对标准ICP算法的不足之处，许多研究者提出ICP算法的各种改进版本，主要涉及如下所示的6个方面。

![enter image description here](https://note.youdao.com/yws/public/resource/51744046d92fd3b4b07cf72defc5ebf2/598CBEEF60B145799CC0018BA61A3233?ynotemdtimestamp=1551862453977)

## 4. ICP算法（Python版）
[代码来自Github](https://github.com/ClayFlannigan/icp/blob/master/icp.py)

```
import numpy as np

def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform between corresponding 3D points A->B
    Input:
      A: Nx3 numpy array of corresponding 3D points
      B: Nx3 numpy array of corresponding 3D points
    Returns:
      T: 4x4 homogeneous transformation matrix
      R: 3x3 rotation matrix
      t: 3x1 column vector
    '''

    assert len(A) == len(B)

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    W = np.dot(BB.T, AA)
    U, s, VT = np.linalg.svd(W)
    R = np.dot(U, VT)

    # special reflection case
    if np.linalg.det(R) < 0:
       VT[2,:] *= -1
       R = np.dot(U, VT)


    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = t

    return T, R, t

def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nx3 array of points
        dst: Nx3 array of points
    Output:
        distances: Euclidean distances (errors) of the nearest neighbor
        indecies: dst indecies of the nearest neighbor
    '''

    indecies = np.zeros(src.shape[0], dtype=np.int)
    distances = np.zeros(src.shape[0])
    for i, s in enumerate(src):
        min_dist = np.inf
        for j, d in enumerate(dst):
            dist = np.linalg.norm(s-d)
            if dist < min_dist:
                min_dist = dist
                indecies[i] = j
                distances[i] = dist    
    return distances, indecies

def icp(A, B, init_pose=None, max_iterations=50, tolerance=0.001):
    '''
    The Iterative Closest Point method
    Input:
        A: Nx3 numpy array of source 3D points
        B: Nx3 numpy array of destination 3D point
        init_pose: 4x4 homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation
        distances: Euclidean distances (errors) of the nearest neighbor
    '''

    # make points homogeneous, copy them so as to maintain the originals
    src = np.ones((4,A.shape[0]))
    dst = np.ones((4,B.shape[0]))
    src[0:3,:] = np.copy(A.T)
    dst[0:3,:] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbours between the current source and destination points
        distances, indices = nearest_neighbor(src[0:3,:].T, dst[0:3,:].T)

        # compute the transformation between the current source and nearest destination points
        T,_,_ = best_fit_transform(src[0:3,:].T, dst[0:3,indices].T)

        # update the current source
    # refer to "Introduction to Robotics" Chapter2 P28. Spatial description and transformations
        src = np.dot(T, src)

        # check error
        mean_error = np.sum(distances) / distances.size
        if abs(prev_error-mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculcate final tranformation
    T,_,_ = best_fit_transform(A, src[0:3,:].T)

    return T, distances
    
if __name__ == "__main__":
    A = np.random.randint(0,101,(20,3))    # 20 points for test
    
    rotz = lambda theta: np.array([[np.cos(theta),-np.sin(theta),0],
                                       [np.sin(theta),np.cos(theta),0],
                                       [0,0,1]])
    trans = np.array([2.12,-0.2,1.3])
    B = A.dot(rotz(np.pi/4).T) + trans
    
    T, distances = icp(A, B)

    np.set_printoptions(precision=3,suppress=True)
    print T
```

上面代码创建一个源点集A（在0-100的整数范围内随机生成20个3维数据点），然后将A绕Z轴旋转45°并沿XYZ轴分别移动一小段距离，得到点集B。结果如下，可以看出ICP算法正确的计算出了变换矩阵。

![enter image description here](https://note.youdao.com/yws/public/resource/51744046d92fd3b4b07cf72defc5ebf2/FDB4A862E7274B7BA3DF3140A86504A6?ynotemdtimestamp=1551862453977)

需要注意几点：

- 首先需要明确公式里的变换是T（P→X）， 即通过旋转和平移把点集P变换到X。我们这里求的变换是T（A→B），要搞清楚对应关系。

- 本例只用了20个点进行测试，ICP算法在求最近邻点的过程中需要计算20×20次距离并比较大小。如果点的数目巨大，那算法的效率将非常低。

- 两个点集的初始相对位置对算法的收敛性有一定影响，最好在“足够近”的条件下进行ICP配准。


## 5.其他优秀教程汇总

[【Python】ICP迭代最近点算法](https://blog.csdn.net/jsgaobiao/article/details/78873718)
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTA4OTg3NjkwM119
-->