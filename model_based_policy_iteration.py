from copy import deepcopy

from robot_mdp import RobotMdp


class PolicyIteration:
    def __init__(self, robot_mdp):
        # init value
        self.values = [0.0 for i in range(len(robot_mdp.states) + 1)]

        # init pi
        self.greedy_pi = dict()
        for state in robot_mdp.states:
            if state in robot_mdp.terminal:
                continue
            self.greedy_pi[state] = robot_mdp.actions[0]

    def pi(self, state):
        return self.greedy_pi[state]

    def policy_evaluate(self, robot_mdp):
        for i in range(10000):
            delta = 0.0
            for state in robot_mdp.states:
                if state in robot_mdp.terminal:
                    continue

                action = self.pi(state)
                is_terminal, next_state, reward = robot_mdp.transform(state, action)
                new_value = reward + robot_mdp.gama * self.values[next_state]
                delta += abs(new_value - self.values[state])
                self.values[state] = new_value

            if delta < 1e-6:
                break

    def policy_improve(self, robot_mdp):
        for state in robot_mdp.states:
            if state in robot_mdp.terminal:
                continue

            a = robot_mdp.actions[0]
            is_terminal, next_state, reward = robot_mdp.transform(state, a)
            q = reward + robot_mdp.gama * self.values[next_state]
            for action in robot_mdp.actions:
                is_terminal, next_state, reward = robot_mdp.transform(state, action)
                if q < reward + robot_mdp.gama * self.values[next_state]:
                    a = action
                    q = reward + robot_mdp.gama * self.values[next_state]
            self.greedy_pi[state] = a

    def cmp_pi(self, robot_mdp, pi1, pi2):
        for state in robot_mdp.states:
            if pi1[state] != pi2[state]:
                return False

        return True

    def policy_iteration(self, robot_mdp):
        old_pi = deepcopy(self.greedy_pi)
        for i in range(10000):
            self.policy_evaluate(robot_mdp)
            self.policy_improve(robot_mdp)
            if self.cmp_pi(robot_mdp, old_pi, self.greedy_pi):
                print("end at %d iteration." % (i + 1))
                break


if __name__ == "__main__":
    robot_mdp = RobotMdp()
    policy_iteration = PolicyIteration(robot_mdp)
    policy_iteration.policy_iteration(robot_mdp)

    print("value")
    for i in range(1, 6):
        print("%d: %f\t" % (i, policy_iteration.values[i]))

    print("policy")
    for i in range(1, 6):
        print("%d->%s\t" % (i, policy_iteration.pi(i)))
