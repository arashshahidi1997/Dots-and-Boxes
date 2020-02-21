import numpy as np

w = 2
h = 3
index_list = np.load('symmetries/index_list' + str(h) + '_' + str(w) + '.npy')
reduced_index_list = list(set(index_list[:, 0]))
folder = 'action_value_function/'


class Agent:

    def __init__(self, w, h, experience='', learning=True, alpha=0.5, gamma=0.8, epsilon=0.1, policy='e-greedy'):
        self.w = w
        self.h = h
        self.policy = policy
        self.learning = learning
        self.wallet = 0  # sum of rewards
        self.score = 0  # score in a dots and boxes match
        self.action = 0  # action
        self.state = 0
        self.previous_action = 0
        self.previous_state = 0
        self.Q = 0

        if experience:
            self.Q = np.load(experience)

        else:
            self.Q = np.load(folder + 'initial_Q' + str(h) + '-' + str(w) + '.npy')

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def choose_action(self, vector):  # epsilon-greedily
        if self.policy == 'e-greedy':
            # print('choose action:')
            s = self.state
            # print 'state:', self.previous_state, self.state
            # print(s, self.Q[s], vector)
            # print(self.Q[s])
            if np.random.rand() < self.epsilon:
                a_index = np.where(vector != 1)[0]
            else:
                a_index = np.where(self.Q[s] == np.max(self.Q[s]))[0]

            # print(a_index)
            self.previous_action = self.action
            self.action = a_index[np.random.randint(len(a_index))]
            # print 'action:', self.previous_action, self.action

        elif self.policy == 'random':
            a_index = np.where(vector != 1)[0]
            self.previous_action = self.action
            self.action = a_index[np.random.randint(len(a_index))]

    def update_rule(self, reward, m):
        if self.learning:
            s0 = self.previous_state
            a0 = self.previous_action
            # print(self.Q[s0, a0])
            # print(s0, a0)
            self.Q[s0, a0] += self.alpha * (reward + self.gamma * m - self.Q[s0, a0])
            # print(self.Q[s0, a0])
            # print('next')

    def target(self, vector):
        return np.max(self.Q[self.state, np.where(vector != 1)])


class Game:

    def __init__(self, w, h, player1, player2):
        self.w = w
        self.h = h
        self.dim = h * (w - 1) + (h - 1) * w

        self.total_score = (w - 1) * (h - 1)
        self.half_score = (w - 1) * (h - 1) / 2

        self.p = [player1, player2]

        self.state_array = [np.zeros((h, w - 1), dtype='int'), np.zeros((h - 1, w), dtype='int')]
        self.vector = vector_form(self.state_array)

        self.boxes = 2 * np.ones((h - 1, w - 1), dtype='int')
        self.turn = 0  # players indices
        self.scored = 0
        self.terminal_state = False
        self.extra_turn = False

    def initialize_episode(self):
        for q in range(2):
            self.p[q].score = 0
            self.p[q].score = 0
            self.p[q].action = 0
            self.p[q].state = 0
            self.p[q].previous_action = 0
            self.p[q].previous_state = 0

        self.state_array = [np.zeros((h, w - 1), dtype='int'), np.zeros((h - 1, w), dtype='int')]
        self.vector = vector_form(self.state_array)

        self.boxes = 2 * np.ones((h - 1, w - 1), dtype='int')
        self.turn = 0  # players indices
        self.scored = 0
        self.terminal_state = False
        self.extra_turn = False

    def episode(self):  # player1 and player2 play n matches of dots and boxes
        self.initialize_episode()
        # print(self.state_array)

        self.update()
        # print(self.p[self.turn].state)

        # print 'turn:', self.turn
        self.p[self.turn].choose_action(self.vector)
        choice = self.action(self.p[self.turn].action)  # choose action

        self.move(choice)
        # print(self.p[self.turn].state)
        # print(self.state_array)

        self.turn = 1 - self.turn

        self.update()

        # print 'turn:', self.turn
        self.p[self.turn].choose_action(self.vector)
        choice = self.action(self.p[self.turn].action)  # choose action

        self.move(choice)
        # print(self.p[self.turn].state)
        # print(self.state_array)

        self.turn = 1 - self.turn

        while not self.terminal_state:

            self.update()

            # print 'turn:', self.turn
            self.p[self.turn].choose_action(self.vector)
            choice = self.action(self.p[self.turn].action)  # choose action
            # self.info(choice)

            self.move(choice)
            # print(self.p[self.turn].state)
            # print(self.state_array)

            if self.terminal_state:
                reward = self.reward()
                if self.turn != 0:
                    reward = -reward

                m = 0
                for turn in range(2):
                    # # print(self.p[0].Q)
                    self.p[turn].previous_state = self.p[turn].state
                    self.p[turn].previous_action = self.p[turn].action
                    self.p[turn].update_rule(reward, m)
                    # # print(self.p[0].Q)
            else:
                reward = 0
                m = self.p[self.turn].target(self.vector)

                # # print(self.p[0].Q)
                self.p[self.turn].update_rule(reward, m)
                s0 = self.p[self.turn].previous_state
                a0 = self.p[self.turn].previous_action
                # # print(self.p[0].Q)

                if not self.extra_turn:
                    self.turn = 1 - self.turn

    def reward(self):
        reward = 0

        if self.p[0].score > self.half_score:
            reward = 1

        elif self.p[0].score < self.half_score:
            reward = -1

        self.p[0].wallet += reward
        self.p[1].wallet -= reward

        return reward

    def move(self, choice):
        score = 0
        self.extra_turn = False
        h, x, y = choice

        self.state_array[h][x, y] = 1  # update state
        self.vector = vector_form(self.state_array)

        if h == 0:

            if x != 0:
                upper = self.state_array[0][x - 1, y] == 1
                upper_left = self.state_array[1][x - 1, y] == 1
                upper_right = self.state_array[1][x - 1, y + 1] == 1
                if upper and upper_left and upper_right:
                    score += 1
                    self.boxes[x - 1, y] = self.turn

            if x != self.h - 1:
                lower = self.state_array[0][x + 1, y] == 1
                lower_left = self.state_array[1][x, y] == 1
                lower_right = self.state_array[1][x, y + 1] == 1
                if lower and lower_left and lower_right:
                    score += 1
                    self.boxes[x, y] = self.turn

        else:

            if y != 0:
                left = self.state_array[1][x, y - 1] == 1
                upper_left = self.state_array[0][x, y - 1] == 1
                lower_left = self.state_array[0][x + 1, y - 1] == 1
                if left and upper_left and lower_left:
                    score += 1
                    self.boxes[x, y - 1] = self.turn

            if y != self.w - 1:
                right = self.state_array[1][x, y + 1] == 1
                upper_right = self.state_array[0][x, y] == 1
                lower_right = self.state_array[0][x + 1, y] == 1
                if right and lower_right and upper_right:
                    score += 1
                    self.boxes[x, y] = self.turn

        self.scored += score

        if self.scored == self.total_score:
            self.terminal_state = True

        if score != 0 and not self.terminal_state:
            self.extra_turn = True

        self.p[self.turn].score += score

    def update(self):

        s = index_list[enumerate(self.vector), 0]
        self.vector = vectorize(s, self.dim)
        self.state_array = matrix_form(self.vector, self.h, self.w)
        self.p[self.turn].previous_state = self.p[self.turn].state
        self.p[self.turn].state = reduced_index_list.index(s)

    def action(self, a):
        v = np.zeros(self.dim, dtype='int')
        v[a] = 1
        h_arr, v_arr = matrix_form(v, self.h, self.w)

        if np.sum(h_arr) == 1:
            h = 0
            i, j = np.where(h_arr == 1)
            x = i[0]
            y = j[0]

        else:
            h = 1
            i, j = np.where(v_arr == 1)
            x = i[0]
            y = j[0]

        return h, x, y

    def info(self, choice):
        # print('state_array:', self.state_array)
        # print('state_vector:', self.vector)
        # print('scores:', self.p[0].score, self.p[1].score)
        # print('scored:', self.scored)
        # print('turn and action:', self.turn, self.p[self.turn].action)
        print('choice', choice)


def enumerate(vector):
    return int("".join(str(v) for v in vector), 2)


def vectorize(num, dim):
    vector = np.zeros(dim, dtype='int')
    v = np.array([int(s) for s in bin(num)[2:]])
    vector[dim - np.size(v):] = v
    return vector


def vector_form(array_list):
    return np.concatenate((np.concatenate(array_list[0]), np.concatenate(array_list[1])))


def matrix_form(vector, h, w):
    a = h * (w - 1)
    vector_list = [vector[:a], vector[a:]]
    array_list = [np.ndarray((h, w - 1), dtype='int'), np.ndarray((h - 1, w), dtype='int')]

    for i in range(w - 1):
        array_list[0][:, i] = vector_list[0][i::(w - 1)]

    for j in range(w):
        array_list[1][:, j] = vector_list[1][j::w]

    return array_list


def self_history_training(theta, w, h, p1, p2):

    #print("Agents are ready")
    game = Game(w, h, p1, p2)

    if p1.learning:
        learner = p1
    else:
        learner = p2

    # # print(index_list)
    # # print(len(reduced_index_list))
    # print("Game is set")

    epsiode_counter = 0

    k = 0
    while True:
        q = np.copy(learner.Q)
        game.episode()
        epsiode_counter += 1
        # print('round', epsiode_counter)
        Delta = np.max(abs(learner.Q - q))
        # print('delta:', Delta)

        if Delta < theta:
            k += 1
            if k == 10:
                # print('converged')
                break


num = 1
exp0 = ''
exp1 = folder + 'sh' + str(num) + '-action_value_function' + str(h) + '-' + str(w) + '.npy'

agents = [Agent(w, h, experience=exp0, learning=False)]
agent = Agent(w, h, experience=exp1, learning=True)

for num in range(1, 20):

    for opponent in agents:
        self_history_training(0.0001, w, h, opponent, agent)
        self_history_training(0.0001, w, h, agent, opponent)

    new_opponent = Agent(w, h, experience=exp1, learning=False)
    agents.append(new_opponent)
    new_file = 'sh'+str(num+1)+'-action_value_function' + str(h) + '-' + str(w) + '.npy'
    np.save(folder + new_file, agent.Q)
    exp1 = folder + new_file
    print(num)
