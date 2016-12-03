# Flappy bird game
#
# Adapted from https://github.com/sourabhv/FlapPyBird

import Image
import itertools
import os
import numpy as np
import pygame
import random
import scipy.misc

class FlappyBirdGame:
  def __init__(self):
    self.SCREENWIDTH = 280
    self.SCREENHEIGHT = 510

    self.PIPEGAPSIZE = 100
    self.BASEY = self.SCREENHEIGHT * 0.79
    self.IMAGES = {}
    self.SOUNDS = {}
    self.HITMASKS = {}
    file_prefix = os.path.dirname(os.path.abspath(__file__))

    self.PLAYERS_LIST = (
      # red bird
      (
          file_prefix+'/FlapPyBird/assets/sprites/redbird-upflap.png',
          file_prefix+'/FlapPyBird/assets/sprites/redbird-midflap.png',
          file_prefix+'/FlapPyBird/assets/sprites/redbird-downflap.png',
      ),
      # blue bird
      #(
          # amount by which base can maximum shift to left
      #    file_prefix+'/FlapPyBird/assets/sprites/bluebird-upflap.png',
      #    file_prefix+'/FlapPyBird/assets/sprites/bluebird-midflap.png',
      #    file_prefix+'/FlapPyBird/assets/sprites/bluebird-downflap.png',      
      #),
      # yellow bird
      #(
      #    file_prefix+'/FlapPyBird/assets/sprites/yellowbird-upflap.png',      
      #    file_prefix+'/FlapPyBird/assets/sprites/yellowbird-midflap.png',     
      #    file_prefix+'/FlapPyBird/assets/sprites/yellowbird-downflap.png',    
      #),
    )

    self.BACKGROUNDS_LIST = (
      file_prefix+'/FlapPyBird/assets/sprites/background-day.png',
      file_prefix+'/FlapPyBird/assets/sprites/background-night.png',
    )

    self.PIPES_LIST = (
      file_prefix+'/FlapPyBird/assets/sprites/pipe-green.png',
      #file_prefix+'/FlapPyBird/assets/sprites/pipe-red.png',
    )

    self.SCREEN = pygame.display.set_mode((self.SCREENWIDTH, self.SCREENHEIGHT))

    self.IMAGES['numbers'] = (
        pygame.image.load(file_prefix+'/FlapPyBird/assets/sprites/0.png').convert_alpha(),
        pygame.image.load(file_prefix+'/FlapPyBird/assets/sprites/1.png').convert_alpha(),
        pygame.image.load(file_prefix+'/FlapPyBird/assets/sprites/2.png').convert_alpha(),
        pygame.image.load(file_prefix+'/FlapPyBird/assets/sprites/3.png').convert_alpha(),
        pygame.image.load(file_prefix+'/FlapPyBird/assets/sprites/4.png').convert_alpha(),
        pygame.image.load(file_prefix+'/FlapPyBird/assets/sprites/5.png').convert_alpha(),
        pygame.image.load(file_prefix+'/FlapPyBird/assets/sprites/6.png').convert_alpha(),
        pygame.image.load(file_prefix+'/FlapPyBird/assets/sprites/7.png').convert_alpha(),
        pygame.image.load(file_prefix+'/FlapPyBird/assets/sprites/8.png').convert_alpha(),
        pygame.image.load(file_prefix+'/FlapPyBird/assets/sprites/9.png').convert_alpha()
    )

    self.IMAGES['gameover'] = pygame.image.load(file_prefix+'/FlapPyBird/assets/sprites/gameover.png')\
      .convert_alpha()
    # message sprite for welcome screen
    self.IMAGES['message'] = pygame.image.load(file_prefix+'/FlapPyBird/assets/sprites/message.png')\
      .convert_alpha()
    # base (ground) sprite
    self.IMAGES['base'] = pygame.image.load(file_prefix+'/FlapPyBird/assets/sprites/base.png')\
      .convert_alpha()

    randBg = random.randint(0, len(self.BACKGROUNDS_LIST) - 1)
    self.IMAGES['background'] = pygame.image.load(self.BACKGROUNDS_LIST[randBg]).convert()

    randPlayer = random.randint(0, len(self.PLAYERS_LIST) - 1)
    self.IMAGES['player'] = (
        pygame.image.load(self.PLAYERS_LIST[randPlayer][0]).convert_alpha(),
        pygame.image.load(self.PLAYERS_LIST[randPlayer][1]).convert_alpha(),
        pygame.image.load(self.PLAYERS_LIST[randPlayer][2]).convert_alpha(),
    )

    # select random pipe sprites
    pipeindex = random.randint(0, len(self.PIPES_LIST) - 1)
    self.IMAGES['pipe'] = (
        pygame.transform.rotate(
            pygame.image.load(self.PIPES_LIST[pipeindex]).convert_alpha(), 180),
        pygame.image.load(self.PIPES_LIST[pipeindex]).convert_alpha(),
    )

    # hismask for pipes
    self.HITMASKS['pipe'] = (
        self.getHitmask(self.IMAGES['pipe'][0]),
        self.getHitmask(self.IMAGES['pipe'][1]),
    )

    # hitmask for player
    self.HITMASKS['player'] = (
        self.getHitmask(self.IMAGES['player'][0]),
        self.getHitmask(self.IMAGES['player'][1]),
        self.getHitmask(self.IMAGES['player'][2]),
    )

    self.score = self.playerIndex = self.loopIter = 0
    self.playerIndexGen = itertools.cycle([0, 1, 2, 1])
    self.playerx = int(self.SCREENWIDTH * 0.2)
    self.playery = int((self.SCREENHEIGHT - self.IMAGES['player'][0].get_height()) / 2)

    self.basex = 0
    self.baseShift = self.IMAGES['base'].get_width() - \
      self.IMAGES['background'].get_width()

    newPipe1 = self.getRandomPipe()
    newPipe2 = self.getRandomPipe()

    # list of upper pipes
    self.upperPipes = [
        {'x': self.SCREENWIDTH, 'y': newPipe1[0]['y']},
        {'x': self.SCREENWIDTH + (self.SCREENWIDTH / 2), 'y': newPipe2[0]['y']},
    ]

    # list of lowerpipe
    self.lowerPipes = [
        {'x': self.SCREENWIDTH, 'y': newPipe1[1]['y']},
        {'x': self.SCREENWIDTH + (self.SCREENWIDTH / 2), 'y': newPipe2[1]['y']},
    ]

    self.pipeVelX = -4

    # player velocity, max velocity, downward accleration, accleration on flap
    self.playerVelY    =  -9   # player's velocity along Y, default same as playerFlapped
    self.playerMaxVelY =  10   # max vel along Y, max descend speed
    self.playerMinVelY =  -8   # min vel along Y, max ascend speed
    self.playerAccY    =   1   # players downward accleration
    self.playerFlapAcc =  -9   # players speed on flapping
    self.playerFlapped = False # True when player flaps 

  def step(self,action, steps=6, display=False):
    frames = np.zeros(
      (steps, self.SCREENHEIGHT, self.SCREENWIDTH, 3), dtype=np.float32
    )
    for frame_idx in xrange(steps):
      reward = 0
      action = action if frame_idx == 0 else 0
      if action == 1:
        if self.playery > -2 * self.IMAGES['player'][0].get_height():
          self.playerVelY = self.playerFlapAcc
          self.playerFlapped = True

      crashTest = self.checkCrash(
        {'x': self.playerx, 'y': self.playery, 'index': self.playerIndex},
        self.upperPipes, self.lowerPipes
      )

      if crashTest[0]:
        #pygame.image.save(self.SCREEN, 'temp.bmp')
        imgstr = pygame.image.tostring(self.SCREEN, 'RGB')
        bmpfile = Image.frombytes('RGB', self.SCREEN.get_size(), imgstr);
        return scipy.misc.imresize(
          np.array(bmpfile, dtype=np.float32), 0.25
        ), reward, True, {}

      playerMidPos = self.playerx + self.IMAGES['player'][0].get_width() / 2
      for pipe in self.upperPipes:
        pipeMidPos = pipe['x'] + self.IMAGES['pipe'][0].get_width() / 2
        if pipeMidPos <= playerMidPos < pipeMidPos + 4:
          reward += 100
          self.score += 1

      # playerIndex basex change
      if (self.loopIter + 1) % 3 == 0:
        self.playerIndex = self.playerIndexGen.next()
      self.loopIter = (self.loopIter + 1) % 30
      self.basex = -((-self.basex + 100) % self.baseShift)

      # player's movement
      if self.playerVelY < self.playerMaxVelY and not self.playerFlapped:
        self.playerVelY += self.playerAccY
      if self.playerFlapped:
        self.playerFlapped = False
      self.playerHeight = self.IMAGES['player'][self.playerIndex].get_height()
      self.playery += min(self.playerVelY, self.BASEY - self.playery - self.playerHeight)

      # move pipes to left
      for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
        uPipe['x'] += self.pipeVelX
        lPipe['x'] += self.pipeVelX

          # add new pipe when first pipe is about to touch left of screen
      if 0 < self.upperPipes[0]['x'] < 5:
        newPipe = self.getRandomPipe()
        self.upperPipes.append(newPipe[0])
        self.lowerPipes.append(newPipe[1])

      # remove first pipe if its out of the screen
      if self.upperPipes[0]['x'] < -self.IMAGES['pipe'][0].get_width():
        self.upperPipes.pop(0)
        self.lowerPipes.pop(0)

      #self.SCREEN.blit(self.IMAGES['background'], (0,0))
      self.SCREEN.fill((0,0,0))

      for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
        self.SCREEN.blit(self.IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
        self.SCREEN.blit(self.IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

      self.SCREEN.blit(self.IMAGES['base'], (self.basex, self.BASEY))
      # print score so player overlaps the score
      self.showScore(self.score)
      self.SCREEN.blit(self.IMAGES['player'][self.playerIndex], (self.playerx, self.playery))

      #pygame.image.save(self.SCREEN, 'temp.bmp')
      #bmpfile = Image.open('temp.bmp');
      imgstr = pygame.image.tostring(self.SCREEN, 'RGB')
      bmpfile = Image.frombytes('RGB', self.SCREEN.get_size(), imgstr);
      frames[frame_idx, :, :, :] = np.array(bmpfile, dtype=np.float32)

      if display:
        self.render()

    return frames, reward, False, {}

  def render(self, mode='human', close=False):
    pygame.display.update()

  def reset(self):
    self.__init__()

  def getHitmask(self, image):
    """returns a hitmask using an image's alpha."""
    mask = []
    for x in range(image.get_width()):
        mask.append([])
        for y in range(image.get_height()):
            mask[x].append(bool(image.get_at((x,y))[3]))
    return mask

  def getRandomPipe(self):
    """returns a randomly generated pipe"""
    # y of gap between upper and lower pipe
    gapY = random.randrange(0, int(self.BASEY * 0.6 - self.PIPEGAPSIZE))
    gapY += int(self.BASEY * 0.2)
    pipeHeight = self.IMAGES['pipe'][0].get_height()
    pipeX = self.SCREENWIDTH + 10

    return [
        {'x': pipeX, 'y': gapY - pipeHeight},  # upper pipe
        {'x': pipeX, 'y': gapY + self.PIPEGAPSIZE}, # lower pipe
    ]

  def checkCrash(self, player, upperPipes, lowerPipes):
    """returns True if player collders with base or pipes."""
    pi = player['index']
    player['w'] = self.IMAGES['player'][0].get_width()
    player['h'] = self.IMAGES['player'][0].get_height()

    # if player crashes into ground
    if player['y'] + player['h'] >= self.BASEY - 1:
      return [True, True]
    else:

      playerRect = pygame.Rect(player['x'], player['y'],
                    player['w'], player['h'])
      pipeW = self.IMAGES['pipe'][0].get_width()
      pipeH = self.IMAGES['pipe'][0].get_height()

      for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
        # upper and lower pipe rects
        uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], pipeW, pipeH)
        lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], pipeW, pipeH)

        # player and upper/lower pipe hitmasks
        pHitMask = self.HITMASKS['player'][pi]
        uHitmask = self.HITMASKS['pipe'][0]
        lHitmask = self.HITMASKS['pipe'][1]

        # if bird collided with upipe or lpipe
        uCollide = self.pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
        lCollide = self.pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

        if uCollide or lCollide:
          return [True, False]

    return [False, False]

  def pixelCollision(self, rect1, rect2, hitmask1, hitmask2):
    """Checks if two objects collide and not just their rects"""
    rect = rect1.clip(rect2)

    if rect.width == 0 or rect.height == 0:
      return False

    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    for x in xrange(rect.width):
      for y in xrange(rect.height):
        if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
          return True
    return False

  def showScore(self, score):
    """displays score in center of screen"""
    scoreDigits = [int(x) for x in list(str(self.score))]
    totalWidth = 0 # total width of all numbers to be printed

    for digit in scoreDigits:
      totalWidth += self.IMAGES['numbers'][digit].get_width()

    Xoffset = (self.SCREENWIDTH - totalWidth) / 2

    for digit in scoreDigits:
      self.SCREEN.blit(self.IMAGES['numbers'][digit], (Xoffset, self.SCREENHEIGHT * 0.1))
      Xoffset += self.IMAGES['numbers'][digit].get_width()

if __name__ == '__main__':
  fb = FlappyBirdGame()
  import time

  while True:
    _, _, terminal, _ = fb.step(np.random.randint(2), display=True)
    if terminal:
      fb.reset()
    time.sleep(1./30)
