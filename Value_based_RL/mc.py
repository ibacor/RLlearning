import random

from Value_based_RL.robot_mdp import *
from Value_based_RL.policy import epsilon_greedy


def mc_policy_evaluation(robot_mdp, state_sample, action_sample, reward_sample):
    vfunc = dict()
    qfunc = dict()
    n_v = dict()
    n_q = dict()
    for s in robot_mdp.states:
        vfunc[s] = 0.0
        n_v[s] = 0
        for a in robot_mdp.actions:
            key = "%d_%s" % (s, a)
            qfunc[key] = 0.0
            n_q[key] = 0

    for i in range(len(state_sample)):
        G = 0.0
        for step in range(len(state_sample[i]) - 1, -1, -1):  # 从后往前迭代计算G
            G *= robot_mdp.gamma
            G += reward_sample[i][step]

        # 获取采样后的值
        for step in range(len(state_sample[i])):
            state = state_sample[i][step]
            vfunc[state] += G
            vfunc[state] += 1

            action = action_sample[i][step]
            key = "%d_%s" % (state, action)
            n_q[key] += 1
            # 这里对q值对平均看不懂，可能有错
            qfunc[key] = (qfunc[key] * (n_q[key] - 1) + G) / n_q[key]

            # G往后退化
            G = (G - reward_sample[i][step]) / robot_mdp.gamma

    # 每次访问Monte-Carlo
    for s in robot_mdp.states:
        if n_v[s] > 1e-6:
            vfunc[s] /= n_v[s]

    print("mc_policy_evaluation")
    print("vfunc")
    print(vfunc)
    print("qfunc")
    print(qfunc)
    return vfunc, qfunc


# mc方法: epsilon-greedy产生数据，其实和上面差不多，只是这里用epsilon-greedy产生数据、得到q值
# 以后再优化整体设计
def mc_epsilon_greedy(iter_num, robot_mdp, epsilon):
    qfunc = dict()
    n_q = dict()
    for s in robot_mdp.states:
        for a in robot_mdp.actions:
            key = "%d_%s" % (s, a)
            qfunc[key] = 0.0
            n_q[key] = 0

    for iter in range(iter_num):
        state_episode = []
        action_episode = []
        reward_episode = []

        # 探索性初始化
        state = robot_mdp.states[int(random.random() * len(robot_mdp.states))]

        is_terminal = False
        count = 0
        while False == is_terminal and count < 100:  # 一个episode不应超过100个
            action = epsilon_greedy(qfunc, state, epsilon, robot_mdp.actions)
            is_terminal, next_state, reward = robot_mdp.transform(state, action)
            state_episode.append(state)
            action_episode.append(action)
            reward_episode.append(reward)

            state = next_state
            count += 1

        G = 0.0
        for i in range(len(state_episode) - 1, -1, -1):
            G *= robot_mdp.gamma
            G += reward_episode[i]

        for i in range(len(state_episode)):
            key = "%d_%s" % (state_episode[i], action_episode[i])
            n_q[key] += 1
            qfunc[key] = (qfunc[key] * (n_q[key] - 1) + G) / n_q[key]

            G = (G - reward_episode[i]) / robot_mdp.gamma

    return qfunc


# epsilon-greedy产生数据
def sarsa(robot_mdp, alpha, epsilon):
    qfunc = dict()
    for s in robot_mdp.states:
        for a in robot_mdp.actions:
            key = "%d_%s" % (s, a)
            qfunc[key] = 0.0
    # 探索性初始化
    state = robot_mdp.states[int(random.random() * len(robot_mdp.states))]
    action = robot_mdp.actions[int(random.random() * len(robot_mdp.actions))]

    is_terminal = False
    count = 0
    while False == is_terminal and count < 100:
        key = "%d_%s" % (state, action)
        is_terminal, next_state, reward = robot_mdp.transform(state, action)
        next_action = epsilon_greedy(qfunc, state, epsilon, robot_mdp.actions)
        next_key = "%d_%s" % (next_state, next_action)

        # q值更新
        qfunc[key] = qfunc[key] + alpha * (reward + robot_mdp.gamma * qfunc[next_key] - qfunc[key])
        state = next_state
        action = next_action
        count += 1
    return qfunc


def qlearning(num_iter, robot_mdp, alpha, epsilon):
    qfunc = dict()
    for s in robot_mdp.states:
        for a in robot_mdp.actions:
            key = "%d_%s" % (s, a)
            qfunc[key] = 0.0

    for iter in range(num_iter):
        state = robot_mdp.states[int(random.random() * len(robot_mdp.states))]
        action = robot_mdp.actions[int(random.random() * len(robot_mdp.actions))]

        is_terminal = False
        count = 0
        while False == is_terminal and count < 100:
            is_terminal, next_state, reward = robot_mdp.transform(state, action)
            key = "%d_%s" % (state, action)
            # find q_max
            q_max = qfunc[0]
            for a in robot_mdp.actions:
                if q_max < qfunc["%d_%s" % (state, a)]:
                    q_max = qfunc["%d_%s" % (state, a)]
            # update q function
            qfunc[key] = qfunc[key] + alpha * (reward + robot_mdp.gamma * q_max - qfunc[key])

            state = next_state
            action = epsilon_greedy(qfunc, state, epsilon, robot_mdp.actions)
            count += 1

    return qfunc


if __name__ == "__main__":
    dic = [[0, 1], [2, 3], [4, 5]]
    print(len(dic))
