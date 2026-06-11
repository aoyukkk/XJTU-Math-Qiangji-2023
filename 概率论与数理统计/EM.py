#设 \( Y = (Y_1, Y_2, \dots, Y_n)^T \) 来自混合正态分布总体 \( p_1 N(\mu_1, \sigma_1^2) + p_2 N(\mu_2, \sigma_2^2) \) 的随机样本，其中 \( p_1 + p_2 = 1 \)，且 \( p_1 > 0, p_2 > 0 \)。
# 随机样本对应的观测数据为 \( y = (y_1, \dots, y_n)^T \)。计算参数 \( \theta = (\mu_1, \sigma_1^2, \mu_2, \sigma_2^2, p_1, p_2) \) 的极大似然估计。
#   计算E步的Q函数，并计算M步的最大值点。
#   设参数的真实值分别为p_1=0.3, p_2=0.7,mu_1=0,\sigma^2=1, mu_2=4,\sigma^2=1，
# 用计算机抽取随机样本，通过EM算法去估计这些参数，提交程序和最后的估计结果。
import numpy as np
from scipy.stats import norm

#初始化
p_1_true = 0.3
p_2_true = 0.7
mu_1_true = 0
sigma_1_true = 1
mu_2_true = 4
sigma_2_true = 1

n = 1000 

n1 = int(n * p_1_true) 
n2 = n - n1         

y1 = np.random.normal(mu_1_true, sigma_1_true, size=n1)
y2 = np.random.normal(mu_2_true, sigma_2_true, size=n2)

y = np.concatenate([y1, y2])
np.random.shuffle(y)

mu1 = np.random.choice(y)
mu2 = np.random.choice(y)
sigma1 = np.std(y)
sigma2 = np.std(y)
p1 = 0.5
p2 = 0.5

#EM
max_iter = 1000   
tol = 1e-6  
log_likelihood_old = -np.inf

for iteration in range(max_iter):
    resp1 = p1 * norm.pdf(y, loc=mu1, scale=sigma1)
    resp2 = p2 * norm.pdf(y, loc=mu2, scale=sigma2)
    gamma = resp1 / (resp1 + resp2)  # 对于每个样本，γ是属于第一分量的概率

#更新参数
    p1_new = np.mean(gamma)
    p2_new = 1 - p1_new
    mu1_new = np.sum(gamma * y) / np.sum(gamma)
    mu2_new = np.sum((1 - gamma) * y) / np.sum(1 - gamma)
    sigma1_new = np.sqrt(np.sum(gamma * (y - mu1_new)**2) / np.sum(gamma))
    sigma2_new = np.sqrt(np.sum((1 - gamma) * (y - mu2_new)**2) / np.sum(1 - gamma))
    
    log_likelihood = np.sum(np.log(resp1 + resp2))
    
    if np.abs(log_likelihood - log_likelihood_old) < tol:
        break
    log_likelihood_old = log_likelihood
    
    p1, p2 = p1_new, p2_new
    mu1, mu2 = mu1_new, mu2_new
    sigma1, sigma2 = sigma1_new, sigma2_new
    
    # 打印迭代过程
    print(f"迭代次数 {iteration}: p1={p1:.4f}, mu1={mu1:.4f}, sigma1={sigma1:.4f}, mu2={mu2:.4f}, sigma2={sigma2:.4f}")


print("最终估计的参数：")
print(f"p1: {p1:.4f}, p2: {p2:.4f}")
print(f"mu1: {mu1:.4f}, sigma1^2: {sigma1**2:.4f}")
print(f"mu2: {mu2:.4f}, sigma2^2: {sigma2**2:.4f}")
