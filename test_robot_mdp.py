import random as ran
ran.seed(0)
from robot_mdp import RobotMdp


def random_pi():
    actions = ['e', 's', 'w', 'n']
    return actions[int(ran.random() * 4)]


def test():
    values = [0.0 for i in range(8)]
    num = 100000

    for n in range(1, num):
        for i in range(1, 6):
            mdp = RobotMdp()
            s = i
            is_terminal = False
            gama = 1.0
            G = 0.0
            while not is_terminal:
                action = random_pi()
                is_terminal, s, reward = mdp.transform(s, action)
                G += gama * reward
                gama = gama * mdp.gama
            # values[i] = (values[i] * (n-1) + G) / n
            values[i-1] = values[i-1] + G

        if n%10000 == 0:
            print(values)
    for i in range(6):
        values[i] /= num
    print(values)

test()
