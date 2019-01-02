#机器人找金币MDP过程
class RobotMdp:
    def __init__(self):
        # 八种状态，以列表形式方便获取元素，字典形式只写出终止状态
        self.states = [1,2,3,4,5,6,7,8]
        self.terminal = dict()
        self.terminal[6] = 1
        self.terminal[7] = 1
        self.terminal[8] = 1

        # 动作集合
        self.actions = ['e','s','w','n']

        # R(s,a) 集合，为了方便终止状态在后面补充
        self.rewards = dict()
        self.rewards['1_n'] = -1.0
        self.rewards['3_n'] = 1.0
        self.rewards['5_n'] = -1.0

        # 转移概率矩阵为1，得到如下转移关系
        self.t = dict()
        self.t['1_e'] = 2
        self.t['1_s'] = 6
        self.t['2_w'] = 1
        self.t['2_e'] = 3
        self.t['3_w'] = 2
        self.t['3_s'] = 7
        self.t['3_e'] = 4
        self.t['4_w'] = 3
        self.t['4_e'] = 5
        self.t['5_w'] = 4
        self.t['5_s'] = 8

        # gama
        self.gama = 0.8

    # 转移函数，补充Mdp类，返回 is_terminal, next_state, reward
    def transform(self, state, action):
        # 异常情况，本身就是终止状态
        if state in self.terminal:
            return True, state, 0
        key = '%d_%s' % (state, action)
        next_state = self.states[0]
        if key in self.t:
            next_state = self.t[key]
        else:
            next_state = state

        is_terminal = False
        if next_state in self.terminal:
            is_terminal = True

        reward = 0.0
        if key in self.rewards:
            reward = self.rewards[key]

        return is_terminal, next_state, reward
