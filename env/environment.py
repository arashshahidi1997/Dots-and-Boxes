import numpy as np
import pygame
import math
import copy


class Agent:

    def __init__(self, w, h, experience='', agent=False):
        self.agent = agent
        self.w = w
        self.h = h
        if agent:
            if experience:
                self.policy = copy.deepcopy()
            else:
                self.policy = Policy(w, h)

        self.wallet = 0  # sum of rewards
        self.score = 0  # score in a dots and boxes match
        self.action = 0  # next action

    def choose_action(self, state_vector):
        self.action = self.policy.distribution[state_vector[0], state_vector[1]].generate()


class Game:

    def __init__(self, w, h, player1, player2, graphics=False):
        self.w = w
        self.h = h
        self.total_score = (w-1) * (h-1)
        self.half_score = (w-1) * (h-1) / 2
        self.p = [player1, player2]
        self.state = [np.zeros((h, w - 1), dtype='int'), np.zeros((h - 1, w), dtype='int')]
        self.boxes = 2 * np.ones((h-1, w-1), dtype='int')

        self.turn = 0   # players indices
        self.scored = 0
        self.terminal_state = False
        self.extra_turn = False

        if player1.agent and player2.agent:
            self.graphics = graphics
        else:
            self.graphics = True

        if self.graphics:
            self.board = Board(self)

    def episode(self):  # player1 and player2 play n matches of dots and boxes

        while not self.terminal_state:
            if not self.p[self.turn].agent:
                choice = self.board.choose_action()
                self.p[self.turn].score += self.move(choice=choice)
            else:
                self.p[self.turn].score += self.move()

            if not self.extra_turn:
                self.turn = 1 - self.turn

        reward = 0

        if self.p[0].score > self.half_score:
            reward = 1
            self.board.didiwin = True

        elif self.p[0].score < self.half_score:
            reward = -1

        self.p[0].wallet += reward
        self.p[1].wallet -= reward

        self.board.finished()

    def move(self, choice=[]):
        score = 0
        self.extra_turn = False
        if self.p[self.turn].agent:
            h, x, y = self.p[self.turn].choose_action(self.state)   # choose action
        else:
            h, x, y = choice

        self.state[h][x, y] = 1     # update state

        if h == 0:

            if x != 0:
                upper = self.state[0][x - 1, y] == 1
                upper_left = self.state[1][x - 1, y] == 1
                upper_right = self.state[1][x - 1, y + 1] == 1
                if upper and upper_left and upper_right:
                    score += 1
                    self.boxes[x-1, y] = self.turn

            if x != self.h - 1:
                lower = self.state[0][x + 1, y] == 1
                lower_left = self.state[1][x, y] == 1
                lower_right = self.state[1][x, y + 1] == 1
                if lower and lower_left and lower_right:
                    score += 1
                    self.boxes[x, y] = self.turn

        else:

            if y != 0:
                left = self.state[1][x, y - 1] == 1
                upper_left = self.state[0][x, y - 1] == 1
                lower_left = self.state[0][x + 1, y - 1] == 1
                if left and upper_left and lower_left:
                    score += 1
                    self.boxes[x, y-1] = self.turn

            if y != self.w - 1:
                right = self.state[1][x, y + 1] == 1
                upper_right = self.state[0][x, y] == 1
                lower_right = self.state[0][x + 1, y] == 1
                if right and lower_right and upper_right:
                    score += 1
                    self.boxes[x, y] = self.turn

        if score != 0:
            self.extra_turn = True
            self.scored += score

        if self.scored == self.total_score:
            self.terminal_state = True

        return score


def enumerate(array):
    h = np.size(array, axis=0)
    w = np.size(array, axis=1)
    enumerator_matrix = np.load('enumerator_matrix.npy')
    return np.sum(array * enumerator_matrix[:h, :w])


def matrix_form(num, a, b):
    array = np.ndarray((a, b), dtype='int')
    k = 1
    for i in range(a):
        for j in range(b):
            array[i, j] = bin(num)[-k]
            k += 1
    return array


class Board:

    def __init__(board, game):
        pass
        pygame.init()
        pygame.font.init()

        # initialize pygame clock
        board.clock = pygame.time.Clock()

        board.game = game

        board.w = game.w
        board.h = game.h

        if board.w < 8 and board.h < 8:
            board.width, board.height = 389, 489
        else:
            board.width = (board.w-1) * 64 + board.w
            board.height = board.width + 100

        # initialize the screen
        board.screen = pygame.display.set_mode((board.width, board.height))
        pygame.display.set_caption("Boxes")
        board.initGraphics()

        board.turn = True

        board.didiwin = False

        board.indicator = [board.greenindicator, board.redindicator]
        board.distance = 64

    def initGraphics(board):
        board.normallinev = pygame.image.load("normalline.png")
        board.normallineh = pygame.transform.rotate(pygame.image.load("normalline.png"), -90)
        board.bar_donev = pygame.image.load("bar_done.png")
        board.bar_doneh = pygame.transform.rotate(pygame.image.load("bar_done.png"), -90)
        board.hoverlinev = pygame.image.load("hoverline.png")
        board.hoverlineh = pygame.transform.rotate(pygame.image.load("hoverline.png"), -90)
        board.separators = pygame.image.load("separators.png")
        board.redindicator = pygame.image.load("redindicator.png")
        board.greenindicator = pygame.image.load("greenindicator.png")
        board.marker = pygame.image.load("greenplayer.png")
        board.othermarker = pygame.image.load("redplayer.png")
        board.winningscreen = pygame.image.load("youwin.png")
        board.gameover = pygame.image.load("gameover.png")
        board.score_panel = pygame.image.load("score_panel.png")

    def drawBoard(board):
        for x in range(board.w-1):
            for y in range(board.h):
                if not board.game.state[0][y, x]:
                    board.screen.blit(board.normallineh, [(x) * board.distance + 5, (y) * board.distance])
                else:
                    board.screen.blit(board.bar_doneh, [(x) * board.distance + 5, (y) * board.distance])

        for x in range(board.w):
            for y in range(board.h-1):
                if not board.game.state[1][y, x]:
                    board.screen.blit(board.normallinev, [(x) * board.distance, (y) * board.distance + 5])
                else:
                    board.screen.blit(board.bar_donev, [(x) * board.distance, (y) * board.distance + 5])
        # draw separators
        for x in range(board.w):
            for y in range(board.h):
                board.screen.blit(board.separators, [x * board.distance, y * board.distance])

    def drawOwnermap(board):
        for x in range(board.w-1):
            for y in range(board.h-1):
                if board.game.boxes[y, x] != 2:
                    if board.game.boxes[y, x] == 0:
                        board.screen.blit(board.marker, (x * board.distance + 5, y * board.distance + 5))
                    if board.game.boxes[y, x] == 1:
                        board.screen.blit(board.othermarker, (x * board.distance + 5, y * board.distance + 5))

    def drawHUD(board):
        # draw the background for the bottom:
        board.screen.blit(board.score_panel, [int((board.width - 389)/2), board.width])

        # create font
        myfont = pygame.font.SysFont(None, 32)

        # create text surface
        label = myfont.render("Turn:", 1, (255, 255, 255))

        # draw surface
        board.screen.blit(label, (int((board.width - 389)/2)+10, board.height - 89))
        board.screen.blit(board.indicator[board.game.turn], (int((board.width - 389)/2)+170, board.height-94))
        # same thing here
        myfont64 = pygame.font.SysFont(None, 64)
        myfont20 = pygame.font.SysFont(None, 20)

        scoreme = myfont64.render(str(board.game.p[0].score), 1, (255, 255, 255))
        scoreother = myfont64.render(str(board.game.p[1].score), 1, (255, 255, 255))
        scoretextme = myfont20.render("Green Player", 1, (255, 255, 255))
        scoretextother = myfont20.render("Red Player", 1, (255, 255, 255))

        board.screen.blit(scoretextme, (int((board.width - 389)/2)+10, board.height-64))
        board.screen.blit(scoreme, (int((board.width - 389)/2)+10, board.height-54))
        board.screen.blit(scoretextother, (int((board.width - 389)/2)+280, board.height-64))
        board.screen.blit(scoreother, (int((board.width - 389)/2)+340, board.height-54))

    def choose_action(board):
        while True:
            # sleep to make the game 60 fps
            board.clock.tick(60)

            # clear the screen
            board.screen.fill(0)

            # draw the board
            board.drawBoard()
            board.drawHUD()
            board.drawOwnermap()

            for event in pygame.event.get():
                # quit if the quit button was pressed
                if event.type == pygame.QUIT:
                    exit()

                # 1
                mouse = pygame.mouse.get_pos()

                # 2
                xpos = int(math.ceil((mouse[0] - board.distance/2) / board.distance))
                ypos = int(math.ceil((mouse[1] - board.distance/2) / board.distance))

                # 3
                is_horizontal = abs(mouse[1] - ypos * board.distance) < abs(mouse[0] - xpos * board.distance)

                # 4
                ypos = ypos - 1 if mouse[1] - ypos * board.distance < 0 and not is_horizontal else ypos
                xpos = xpos - 1 if mouse[0] - xpos * board.distance < 0 and is_horizontal else xpos

                # 5
                Board = board.game.state[0] if is_horizontal else board.game.state[1]
                isoutofbounds = False

                # 6
                try:
                    if not Board[ypos, xpos]:
                        board.screen.blit(board.hoverlineh if is_horizontal else board.hoverlinev,
                                          [xpos * board.distance + 5 if is_horizontal else xpos * board.distance,
                                           ypos * board.distance if is_horizontal else ypos * board.distance + 5])

                except:
                    isoutofbounds = True
                    pass

                if not isoutofbounds:
                    alreadyplaced = Board[ypos, xpos]

                else:
                    alreadyplaced = False

                if pygame.mouse.get_pressed()[0] and not alreadyplaced and not isoutofbounds:
                    if is_horizontal:
                        pygame.display.flip()
                        return [0, ypos, xpos]
                    else:
                        pygame.display.flip()
                        return [1, ypos, xpos]

            # update the screen
            pygame.display.flip()

    def finished(board):
        board.drawBoard()
        board.drawHUD()
        board.draw0Ownermap()
        board.screen.blit(board.gameover if not board.didiwin else board.winningscreen, (0, 0))
        while 1:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit()
            pygame.display.flip()


class Distribution:

    def __init__(distr, state):
        distr.state = state
        distr.h = [np.size(state[0], axis=1), np.size(state[1], axis=1)]
        distr.w = [np.size(state[0], axis=0), np.size(state[0], axis=0)]
        distr.p = []
        distr.action_space = []

    def initialize(distr):
        for h in range(2):
            for i in range(distr.h[h]):
                for j in range(distr.w[h]):
                    distr.action_space.append((h, i, j))
                    s = 0
                    if distr.state[h][i, j] == 0:
                        s = 1
                    distr.p.append(s)

        distr.normalize()

    def normalize(distr):
        distr.p = np.concatenate((distr.h_array, distr.v_array))
        distr.p = distr.p / np.sum(distr.p)

    def generate(distr):
        s = np.random.choice(np.size(distr.p), p=distr.p)
        return distr.action_space[s]


class Policy:

    def __init__(policy, w, h):
        policy.w = w
        policy.h = h
        policy.terminal_state_vector = [enumerate(np.ones(h, w-1)), enumerate(np.ones(h-1, w))]
        policy.distribution = np.ndarray((policy.terminal_state_vector[0], policy.terminal_state_vector[1]),
                                         dtype=object)

    def initialize(policy):

        for i in range(policy.terminal_state_vector[0]):
            for j in range(policy.terminal_state_vector[1]):

                policy.distribution[i, j] = Distribution([matrix_form(i, policy.h, policy.w-1),
                                                         matrix_form(j, policy.h-1, policy.w)])

'''
w = 4
h = 4
p1 = Agent()
p2 = Agent()
game = Game(w, h, p1, p2)
game.episode()
'''