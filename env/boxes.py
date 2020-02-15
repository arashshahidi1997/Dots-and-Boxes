import pygame
import math

class BoxesGame():
    def __init__(self):
        pass
        # 1
        pygame.init()
        pygame.font.init()

        width, height = 389, 489
        # 2
        # initialize the screen
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Boxes")
        # 3
        # initialize pygame clock
        self.clock = pygame.time.Clock()
        self.boardh = [[False for x in range(6)] for y in range(7)]
        self.boardv = [[False for x in range(7)] for y in range(6)]
        self.initGraphics()
        self.turn = True
        self.me=0
        self.otherplayer=0
        self.didiwin=False
        self.owner = [[0 for x in range(6)] for y in range(6)]

    def initGraphics(self):
        self.normallinev=pygame.image.load("normalline.png")
        self.normallineh=pygame.transform.rotate(pygame.image.load("normalline.png"), -90)
        self.bar_donev=pygame.image.load("bar_done.png")
        self.bar_doneh=pygame.transform.rotate(pygame.image.load("bar_done.png"), -90)
        self.hoverlinev=pygame.image.load("hoverline.png")
        self.hoverlineh=pygame.transform.rotate(pygame.image.load("hoverline.png"), -90)
        self.separators=pygame.image.load("separators.png")
        self.redindicator=pygame.image.load("redindicator.png")
        self.greenindicator=pygame.image.load("greenindicator.png")
        self.greenplayer=pygame.image.load("greenplayer.png")
        self.blueplayer=pygame.image.load("blueplayer.png")
        self.winningscreen=pygame.image.load("youwin.png")
        self.gameover=pygame.image.load("gameover.png")
        self.score_panel=pygame.image.load("score_panel.png")

    def drawBoard(self):
        for x in range(6):
            for y in range(7):
                if not self.boardh[y][x]:
                    self.screen.blit(self.normallineh, [(x)*64+5, (y)*64])
                else:
                    self.screen.blit(self.bar_doneh, [(x)*64+5, (y)*64])
        for x in range(7):
            for y in range(6):
                if not self.boardv[y][x]:
                    self.screen.blit(self.normallinev, [(x)*64, (y)*64+5])
                else:
                    self.screen.blit(self.bar_donev, [(x)*64, (y)*64+5])
        #draw separators
        for x in range(7):
            for y in range(7):
                self.screen.blit(self.separators, [x*64, y*64])

    def drawOwnermap(self):
        for x in range(6):
            for y in range(6):
                if self.owner[x][y]!=0:
                    if self.owner[x][y]=="win":
                        self.screen.blit(self.marker, (x*64+5, y*64+5))
                    if self.owner[x][y]=="lose":
                        self.screen.blit(self.othermarker, (x*64+5, y*64+5))

    def drawHUD(self):
        #draw the background for the bottom:
        self.screen.blit(self.score_panel, [0, 389])

        #create font
        myfont = pygame.font.SysFont(None, 32)

        #create text surface
        label = myfont.render("Your Turn:", 1, (255, 255, 255))

        #draw surface
        self.screen.blit(label, (10, 400))
        self.screen.blit(self.greenindicator, (130, 395))
        #same thing here
        myfont64 = pygame.font.SysFont(None, 64)
        myfont20 = pygame.font.SysFont(None, 20)

        scoreme = myfont64.render(str(self.me), 1, (255, 255, 255))
        scoreother = myfont64.render(str(self.otherplayer), 1, (255, 255, 255))
        scoretextme = myfont20.render("You", 1, (255, 255, 255))
        scoretextother = myfont20.render("Other Player", 1, (255, 255, 255))

        self.screen.blit(scoretextme, (10, 425))
        self.screen.blit(scoreme, (10, 435))
        self.screen.blit(scoretextother, (280, 425))
        self.screen.blit(scoreother, (340, 435))

    def update(self):
        #sleep to make the game 60 fps
        self.clock.tick(60)

        #clear the screen
        self.screen.fill(0)
        #draw the board
        self.drawBoard()
        self.drawHUD()

        for event in pygame.event.get():
            #quit if the quit button was pressed
            if event.type == pygame.QUIT:
                exit()

            #1
            mouse = pygame.mouse.get_pos()

            #2
            xpos = int(math.ceil((mouse[0]-32)/64.0))
            ypos = int(math.ceil((mouse[1]-32)/64.0))

            #3
            is_horizontal = abs(mouse[1] - ypos*64) < abs(mouse[0] - xpos*64)

            #4
            ypos = ypos - 1 if mouse[1] - ypos*64 < 0 and not is_horizontal else ypos
            xpos = xpos - 1 if mouse[0] - xpos*64 < 0 and is_horizontal else xpos

            #5
            board = self.boardh if is_horizontal else self.boardv
            isoutofbounds=False

            #6
            try:
                if not board[ypos][xpos]:
                    self.screen.blit(self.hoverlineh if is_horizontal else self.hoverlinev, [xpos*64+5 if is_horizontal else xpos*64, ypos*64 if is_horizontal else ypos*64+5])

            except:
                isoutofbounds=True
                pass
            if not isoutofbounds:
                alreadyplaced=board[ypos][xpos]
            else:
                alreadyplaced=False

            if pygame.mouse.get_pressed()[0] and not alreadyplaced and not isoutofbounds:
                if is_horizontal:
                    self.boardh[ypos][xpos]=True
                else:
                    self.boardv[ypos][xpos]=True

        # update the screen
        pygame.display.flip()

    def check_for_boxes(self):
        score = 0
        h, x, y = self.p[self.turn].choose_action(self.state)   # choose action
        self.state[h][x, y] = 1     # update state

        if self.step == self.total_score:
            self.terminal_state = True

        if h == 0:

            if x != 0:
                upper = self.state[0][x - 1, y] == 1
                upper_left = self.state[1][x - 1, y] == 1
                upper_right = self.state[1][x - 1, y + 1] == 1
                if upper and upper_left and upper_right:
                    score += 1

            if x != self.h - 1:
                lower = self.state[0][x + 1, y] == 1
                lower_left = self.state[1][x, y] == 1
                lower_right = self.state[1][x, y + 1] == 1
                if lower and lower_left and lower_right:
                    score += 1

        else:

            if y != 0:
                left = self.state[1][x, y - 1] == 1
                upper_left = self.state[0][x, y - 1] == 1
                lower_left = self.state[0][x + 1, y - 1] == 1
                if left and upper_left and lower_left:
                    score += 1

            if y != self.w - 1:
                right = self.state[1][x, y + 1] == 1
                upper_right = self.state[0][x, y] == 1
                lower_right = self.state[0][x + 1, y] == 1
                if right and lower_right and upper_right:
                    score += 1


    def finished(self):
        self.screen.blit(self.gameover if not self.didiwin else self.winningscreen, (0, 0))
        while 1:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit()
            pygame.display.flip()

bg = BoxesGame()  #_init is called right here
while 1:
    bg.update()
