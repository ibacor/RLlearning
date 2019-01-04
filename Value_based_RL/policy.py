import random


def epsilon_greedy(qfunc, state, epsilon, actions):
    # select argmax a q_max(s,a)
    a_max = greedy(qfunc, state, actions)

    probability = dict()
    for action in actions:
        probability[action] = epsilon / len(actions)
    probability[a_max] += 1 - epsilon

    ran = random.random()
    sum = 0.0
    for action in actions:
        if sum >= ran:
            return action
        sum += probability[action]
    return actions[len(actions) - 1]


def greedy(qfunc, state, actions):
    a_max = actions[0]
    key = "%d_%s" % (state, a_max)
    q_max = qfunc[key]
    for action in actions:
        key = "%d_%s" % (state, action)
        q = qfunc[key]
        if q_max < q:
            q_max = q
            a_max = action

    return a_max


if __name__ == "__main__":
    dic = dict()
