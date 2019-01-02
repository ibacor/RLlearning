from robot_mdp import RobotMdp


class ValueIteration:
    def __init__(self, robot_mdp):
        self.values = [0.0 for i in range(len(robot_mdp.states) + 1)]

        self.greedy_pi = dict()
        for state in robot_mdp.states:
            if state in robot_mdp.terminal:
                continue

            self.greedy_pi[state] = robot_mdp.actions[0]

    def value_iteration(self, robot_mdp):
        for i in range(10000):
            delta = 0.0
            for state in robot_mdp.states:
                a = robot_mdp.actions[0]
                is_terminal, next_state, reward = robot_mdp.transform(state, a)
                # 由于使用greedy策略，q值也是新的值函数值
                q = reward + robot_mdp.gama * self.values[next_state]

                for action in robot_mdp.actions:
                    is_terminal, next_state, reward = robot_mdp.transform(state, action)
                    if q < reward + robot_mdp.gama * self.values[next_state]:
                        a = action
                        q = reward + robot_mdp.gama * self.values[next_state]

                delta += abs(self.values[state] - q)
                self.values[state] = q
                self.greedy_pi[state] = a

            if delta < 1e-6:
                break


if __name__ == "__main__":
    robot_mdp = RobotMdp()
    value_iteration = ValueIteration(robot_mdp)
    value_iteration.value_iteration(robot_mdp)

    print("value")
    for i in range(1, 6):
        print("%d: %f\t" % (i, value_iteration.values[i]))

    print("policy")
    for i in range(1, 6):
        print("%d->%s" % (i, value_iteration.greedy_pi[i]))
