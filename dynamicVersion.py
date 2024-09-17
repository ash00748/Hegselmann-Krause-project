#%%
import numpy as np
import math
import matplotlib.pyplot as plt

#%%
def compare(x, y, eps):
    result = 1
    for k in range(len(x)):
        if abs(x[k] - y[k]) >= eps:
            result = 0
    return bool(result)


def new_op(x, i, eps):
    count = 1
    summ = x[i]
    for k in range(len(x)):
        if (abs(x[i] - x[k]) < eps) and (i != k):
            # print(i, j, x[i], x[j])
            summ += x[k]
            count += 1
    # print(summ, count)
    return summ/count, count


def pair_count(x0):
    count = [0, 0]
    for i in range(len(x0)):
        if x0[i] >= 0.5:
            count[0] += 1
        else:
            count[1] += 1
    # count[0] = count[0]/len(x0)
    # count[1] = count[1] / len(x0)
    return count


def Hen_Kr_constant_speed(x0, eps, eps2, fix, v):
    X = [np.array(x0)]
    t = 0
    while True:
        X.append(np.zeros(len(x0)))
        t += 1
        for j in range(len(x0)):
            if j not in fix:
                X[t][j], _ = new_op(X[t - 1], j, eps)
            else:
                if X[t - 1][j] + v <= 1:
                    X[t][j] = X[t - 1][j] + v
                else:
                    X[t][j] = 1
        if compare(X[t], X[t - 1], eps2):
            break
    return X, t


def Hen_Kr_dynamic_speed(x0, eps, eps2, fix):
    X = [np.array(x0)]
    t = 0
    while True:
        X.append(np.zeros(len(x0)))
        t += 1
        for j in range(len(x0)):
            buf_data, count = new_op(X[t - 1], j, eps)
            if j not in fix:
                X[t][j] = buf_data
            else:
                v_cur = eps / count
                if X[t - 1][j] + v_cur <= 1:
                    X[t][j] = X[t - 1][j] + v_cur
                else:
                    X[t][j] = 1
        if compare(X[t], X[t - 1], eps2):
            break
    return X, t


def Hen_Kr_dynamic_speed_consensus_statistics(x0, eps, eps2, fix):
    X = [np.array(x0)]
    t = 0
    consensus_value = 0
    consensus_time = 0
    consensus_flag = True
    pos0_6_time = 0
    pos0_6_flag = True
    while True:
        X.append(np.zeros(len(x0)))
        t += 1
        for j in range(len(x0)):
            buf_data, count = new_op(X[t - 1], j, eps)
            if j not in fix:
                X[t][j] = buf_data
            else:
                v_cur = eps / count
                if X[t - 1][j] + v_cur <= 1:
                    X[t][j] = X[t - 1][j] + v_cur
                else:
                    X[t][j] = 1
        if consensus_flag and is_consensus(X[t], fix):
            consensus_flag = False
            consensus_time = t
            consensus_value = X[t][0]
        if not consensus_flag and pos0_6_flag and X[t][0] >= 0.6:
            pos0_6_time = t
            pos0_6_flag = False
        if compare(X[t], X[t - 1], eps2):
            break
    return X, t, consensus_value, consensus_time, pos0_6_time


def is_consensus(x, fix):
    res = True
    count = 0
    for i in range(1, len(x)):
        if x[i-1] != x[i]:
            count += 1
    if count > 2:
        res = False
    return res


def find_fix(x0):
    diff = 1
    ind = 0
    for j in range(len(x0)):
        if (x0[j] - 0.5 < diff) and (x0[j] - 0.5 >= 0):
            ind = j
            diff = abs(0.5 - x0[j])
    return ind


def find_dynamic_index(x0, spy_op):
    buf = x0.copy()
    for j in range(len(x0)):
        if buf[j] >= spy_op:
            buf = np.insert(buf, j, spy_op)
            return buf, j

print('number of agents: ')
#n = int(input())
n = 50
print(n)
eps = 0.2
eps2 = 0.001

# np.random.seed(111)
x0 = np.random.sample(n)
x0.sort()

start = pair_count(x0)
print('Initial set: ', x0)

# X = [np.array(x0)]
# t = 0

# ind = find_fix(x0)+2
# fix = {ind}
fix_control = set({})
# print(fix, x0[ind])

spy_opinion = x0[0] + eps  # the lowest opinion + eps
x0, dyn_ind = find_dynamic_index(x0, spy_opinion)
dyn_fix = {dyn_ind}
# print(x0[0], spy_opinion, dyn_ind)

v = eps/50
v_c = eps/50

X, t, cons_value, cons_time, positive_time = Hen_Kr_dynamic_speed_consensus_statistics(x0, eps, eps2, dyn_fix)
# X_control, t_control = Hen_Kr(x0, eps, eps2, fix_control, v_c)
X_control, t_control = Hen_Kr_constant_speed(x0, eps, eps2, dyn_fix, v_c)

finish = pair_count(X[t])
finish_c = pair_count(X_control[t_control])
print('Consensus time:', cons_time)
print('Consensus value in consensus time:', cons_value)
print('Time when consensus reaches value 0.6:', positive_time)
print(start, finish_c, finish)

#%%
cons_value_data = np.asarray([])
cons_time_data = np.asarray([])
positive_time_data = np.asarray([])
count_experiments = 500

for i in range(count_experiments):
    x0 = np.random.sample(n)
    x0.sort()
    X, t, cons_value, cons_time, positive_time = Hen_Kr_dynamic_speed_consensus_statistics(x0, eps, eps2, dyn_fix)
    cons_value_data = np.append(cons_value_data, cons_value)
    cons_time_data = np.append(cons_time_data, cons_time)
    positive_time_data = np.append(positive_time_data, positive_time)
print('Average consensus time:', np.mean(cons_time_data))
print('Average consensus value in consensus time:', np.mean(cons_value_data))
print('Average time when consensus reached 0.6:', np.mean(positive_time_data))
count_coin = 0.
for i in range(count_experiments):
    if cons_time_data[i] == positive_time_data[i]:
        count_coin += 1
print(count_coin)
print('Percentage of consensus formation with value above 0.6:', count_coin / count_experiments)

# p = np.zeros(100)
# success = 0

# for i in range(1):
#     X, t = Hen_Kr(x0, eps, eps2, fix, v)
#     X_control, t_control = Hen_Kr(x0, eps, eps2, fix_control, v_c)
#     finish = pair_count(X[t])
#     finish_c = pair_count(X_control[t_control])
#     print(start, finish_c, finish)
#     if finish[0] > finish_c[0]:
#         success += 1
#     p[i] = finish[0]
#     x0 = np.random.sample(n)
#     x0.sort()
#     start = pair_count(x0)

# average = np.mean(p)
# print(p, average, success)


#%%
t_list = np.linspace(0, t, t + 1)
t_c_list = np.linspace(0, t_control, t_control + 1)
#print(t_list)

# finish = pair_count(X[t])
# finish_c = pair_count(X_control[t_control])
# print(start, finish_c, finish)

# fig = plt.figure(facecolor='white')
# ax = fig.add_subplot(111)
# #ax1 = fig.add_subplot(111)
# ax.plot(t_list, X, linewidth=2)
# #ax1.plot(t_c_list, X_control, linewidth=2)
# plt.show()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
ax1.plot(t_list, X, linewidth=2)
ax2.plot(t_c_list, X_control, linewidth=2)
plt.show()
