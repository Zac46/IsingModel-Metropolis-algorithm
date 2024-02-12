import numpy as np
import matplotlib.pyplot as plt
from numba import jit

n = int(input("输入格子的边长(整数)"))  #正则系综n越大，相对涨落越小
J = 30
k = 1

# 初始化用于存放结果的列表
kbTlist = []
average_energies_list = []
average_moments_list=[]
average_C_list=[]
energy_test=[]
iter=[]
# 设置迭代次数
iteration = 30000

# 计算格点相邻能量的函数, flip用于传递翻转情况
@jit
def energy_in_vicinity(lattice, i, j, flip):
    rows, cols = lattice.shape
    spin = lattice[i, j]
    if flip == 1:
        spin *= -1

    # 求出最近邻格点坐标，采用周期性边界条件
    left = i, (j - 1) % cols
    right = i, (j + 1) % cols
    up = (i - 1) % rows, j
    down = (i + 1) % rows, j
    # 获取最邻近格点的自旋
    left_spin = lattice[left]
    right_spin = lattice[right]
    up_spin = lattice[up]
    down_spin = lattice[down]
    # 计算总能量
    energy = -J * spin * (left_spin + right_spin + up_spin + down_spin)

    return energy

@jit
def get_total_energy(lattice,n):
     total_energy=0
     for i in range(n):
        for j in range(n):
            total_energy += energy_in_vicinity(lattice, i, j, 0)
     total_energy/=(2*J)
     return total_energy

@jit
def get_total_moment(lattice,n):
     total_moment=0
     for i in range(n):
        for j in range(n):
            total_moment+=lattice[i,j]
     return total_moment

# 开始计算
temperatures = np.arange(0.0001, 160, 1)

"""
#取一个样例温度，看迭代次数要到多少次后才有稳定结果
T=100
lattice = np.ones((n,n))

for i in range(iteration):
     x, y = np.random.randint(0, n, size=2)
     delta_energy = energy_in_vicinity(lattice, x, y, 1) - energy_in_vicinity(lattice, x, y, 0)
     # 若能量降低就翻转
     if delta_energy <= 0:
          lattice[x, y] *= -1
     # 若不降低以e^(\beta*delta_E)的概率翻转
     else:
          probability = np.exp(-delta_energy / (k * T))
          random_number = np.random.rand()        
          if probability > random_number:
                lattice[x, y] *= -1
     energy_test.append(get_total_energy(lattice,n))
     iter.append(i)
    
plt.plot(iter,energy_test,marker='o', linestyle='-',markersize=3)
plt.ylabel(r'total energy/$J$')
plt.xlabel(r'iteration')
plt.title('total energy-interation time(lattice 20*20)')
plt.show()
"""

for T in temperatures:
    # 生成初始位形，初始态取全部朝向一致，这很重要，因为此算法很难区分大块长程序磁矩反向和真正能量最低的态。
    lattice = np.ones((n,n))
    total_energy = 0
    total_moment = 0
    total_energy_sq = 0
    for i in range(iteration):
        x, y = np.random.randint(0, n, size=2)
        delta_energy = energy_in_vicinity(lattice, x, y, 1) - energy_in_vicinity(lattice, x, y, 0)
        # 若能量降低就翻转
        if delta_energy <= 0:
            lattice[x, y] *= -1
        # 若不降低以e^(\beta*delta_E)的概率翻转
        else:
            probability = np.exp(-delta_energy / (k * T))
            random_number = np.random.rand()

            if probability > random_number:
                lattice[x, y] *= -1
        if i>=5000: #5000步迭代之后的都作为系综的样本，共计15000个样本
            total_energy+=get_total_energy(lattice,n)
            total_moment+=get_total_moment(lattice,n)
            total_energy_sq+=(get_total_energy(lattice,n))**2
    average_E = total_energy / (25000)
    average_m = total_moment / (25000)
    average_E_sq =  total_energy_sq/(25000)
   #结果以J为单位 
    kbTlist.append(k*T/J)
    #total_energy()返回的就是以J为单位的能量，没有必要再除
    average_energies_list.append(average_E/(n**2))
   #m天然就是归一的
    average_moments_list.append(abs(average_m/(n**2))) 
   #显示进程
    average_C_list.append((average_E_sq-average_E**2)/(n**2))
    print(T,(average_E_sq-average_E**2))

#创建三个子图
fig, (ax1, ax2,ax3) = plt.subplots(3, 1, sharex=True, figsize=(8, 12))

# 绘制第一幅图
ax1.plot(kbTlist, average_energies_list, marker='o', linestyle='-', color='b')
ax1.set_ylabel(r'$\bar{E}/J$')
# 绘制第二幅图
ax2.plot(kbTlist, average_moments_list, marker='o', linestyle='-', color='r')
ax2.set_ylabel(r'$\bar{m}$')

# 绘制第三幅图
ax3.plot(kbTlist, average_C_list, marker='o', linestyle='-', color='g')
ax3.set_ylabel(r'$\bar{C}$')
ax3.set_xlabel(r'$k_BT/J$')

# 调整布局
plt.tight_layout()
plt.suptitle('20*20 lattice', y=1.02, fontsize=16)
# 显示图形
plt.show()
