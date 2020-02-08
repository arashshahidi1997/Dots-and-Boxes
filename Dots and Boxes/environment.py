import numpy as np

# initialize
w = 5
h = 5
points = np.ndarray((h, w))
h_edges = np.zeros((h, w - 1))
v_edges = np.zeros((h - 1, w))
unoccupied_edges = [[], []]

for i in range(h - 1):
    for j in range(w - 1):
        unoccupied_edges[0].append([i, j])
        unoccupied_edges[1].append([i, j])

unoccupied_edges[0].append([h - 1, w - 2])
unoccupied_edges[1].append([h - 2, w - 1])


def draw(choice, player):
    extra_turn = False
    score = 0
    h, x, y = choice
    unoccupied_edges[h].remove([x, y])

    if h == 0:
        h_edges[x, y] = player
        if x != 0:
            upper = h_edges[x - 1, y] == player
            upper_left = v_edges[x - 1, y] == player
            upper_right = v_edges[x - 1, y + 1] == player
            if upper and upper_left and upper_right:
                score += 1
                extra_turn = True

        if x != h - 1:
            lower = h_edges[x + 1, y] == player
            lower_left = v_edges[x, y] == player
            lower_right = v_edges[x, y + 1] == player
            if lower and lower_left and lower_right:
                score += 1
                extra_turn = True

    else:
        v_edges[x, y] = player
        if y != 0:
            left = v_edges[x, y - 1] == player
            upper_left = h_edges[x, y - 1] == player
            lower_left = h_edges[x + 1, y - 1] == player
            if left and upper_left and lower_left:
                score += 1
                extra_turn = True

        if y != w - 1:
            right = v_edges[x, y + 1] == player
            upper_right = h_edges[x, y] == player
            lower_right = h_edges[x + 1, y] == player
            if right and lower_right and upper_right:
                score += 1
                extra_turn = True

    return score, extra_turn


# game
def play(player1, player2, env, n):
    for i in range(n):

        reward_1 = 0
        score = dict({1: 0, 2: 0})
        end = False
        player = 1
        initial_state =  # should be added after building the agents and defining MDP states and rewards
        choice = player1(initial_state)

        while True:
            if not draw(choice, player, score):
                player = (player + 1) % 2

            if unoccupied_edges == [[], []]:
                if score[0] > score[1]:
                    reward_1 = 1
                elif score[0] < score[1]:
                    reward_1 = -1
                else:
                    reward_1 = 0
                reward_2 = -reward_1
                break

            else:
                choice = player(state)

        return reward_1

