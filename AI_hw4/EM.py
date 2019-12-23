#-*-coding:utf-8-*-
from scipy import stats
import numpy as np
import math

# 图a:将已经观测到的结果保存下来 (H = 1, T = 0)
observations = np.array([[1, 0, 0, 0, 1, 1, 0, 1, 0, 1],
                         [1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                         [1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
                         [1, 0, 1, 0, 0, 0, 1, 1, 0, 0],
                         [0, 1, 1, 1, 0, 1, 1, 1, 0, 1]])
# 图b：Θ_A = 0.6, Θ_B = 0.4
demo_theta_A = 0.6
demo_theta_B = 0.4

#  ִ执行一次EM过程并更新Θ_A和Θ_B
def EM_once(priors):

    theta_A = priors[0]
    theta_B = priors[1]

    counts = {'A': {'H': 0, 'T': 0}, 'B': {'H': 0, 'T': 0}}

    global observations
    # E step
    for observation in observations:
        lens = len(observation)
        num_heads = observation.sum()   
        num_tails = lens - num_heads

        # 求二项分布的PMF，并将其归一化
        observaton_pmf_A = stats.binom.pmf(num_heads, lens, theta_A)
        observaton_pmf_B = stats.binom.pmf(num_heads, lens, theta_B)
        weight_A = observaton_pmf_A / (observaton_pmf_A + observaton_pmf_B)
        weight_B = observaton_pmf_B / (observaton_pmf_A + observaton_pmf_B)
        
        # 更新在当前参数下A、B硬币产生的正反面次数
        counts['A']['H'] += weight_A * num_heads
        counts['A']['T'] += weight_A * num_tails
        counts['B']['H'] += weight_B * num_heads
        counts['B']['T'] += weight_B * num_tails

    # M step
    # 更新Θ_A和Θ_B
    new_theta_A = counts['A']['H'] / (counts['A']['H'] + counts['A']['T'])
    new_theta_B = counts['B']['H'] / (counts['B']['H'] + counts['B']['T'])

    return [new_theta_A, new_theta_B]

# 执行EM循环，误差小于阈值或循环次数大于一定次数后停止
def EM(prior, threshod = 1e-6, max_loops = 10000):
    i = 0
    while i < max_loops:
        new_prior = EM_once(prior)
        delta_change = np.abs(prior[0] - new_prior[0])

        if delta_change < threshod:
            break
        else:
            prior = new_prior
            i += 1
    
    return [new_prior, i]
 
if __name__ == "__main__":
    result = EM([demo_theta_A, demo_theta_B])
    print "theta_A = ",result[0][0] 
    print "theta_B = ",result[0][1]
    print "Loops = ",result[1],"times"