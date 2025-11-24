import numpy as np
import pygame
import random

from game_inputs import Direction, Game
from collections import namedtuple

INITIAL_SPEED = 100
SCAN_DISTANCE = 3
STATE_SIZE = 4

MAJOR_REWARD = 10
PAIN_OF_MOVEMENT = -0.01

BG_COLOR = (0, 0, 0)
SNAKE_COLOR = (101, 177, 238)
FOOD_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)


Point = namedtuple('Point', 'x, y')


class SnakeGame(Game):
    def __init__(self, bloc_size=20, width_bloc=20, height_blocs=20):
        pygame.init()

        self.font = pygame.font.Font('data/DS-DIGI.TTF', 25)

        self.w = width_bloc
        self.h = height_blocs
        self.bloc_size = bloc_size
        self.reset()

        self.clock = pygame.time.Clock()

    def reset(self):
        self.display = None

        self.direction = Direction.RIGHT
        self.head = Point(self.w // 2, self.h // 2)
        self.snake = [
            self.head,
            Point(self.head.x - 1, self.head.y),
            Point(self.head.x - 2, self.head.y),]

        self._score = 0
        self.game_over = False

        self.food = None
        self._place_food()

        return self._state(), self._infos()

    def step(self, direction):
        old_score = self._score

        self._update_position(direction)
        self.snake.insert(0, self.head)

        self.game_over = True
        if not self._has_collide(self.head):
            self.game_over = False
            if self.head == self.food:
                self._score += 1
                self._place_food()
            else:
                self.snake.pop()

        return self._state(), self._reward(old_score), self.game_over, self._infos()

    def render(self):
        if self.display is None:
            self.display = pygame.display.set_mode(
                (self.w * self.bloc_size, self.h * self.bloc_size))
            pygame.display.set_caption('Snake')

        self.display.fill(BG_COLOR)

        for pt in self.snake:
            pygame.draw.rect(self.display, SNAKE_COLOR, pygame.Rect(
                pt.x*self.bloc_size, pt.y*self.bloc_size, self.bloc_size, self.bloc_size))

        pygame.draw.rect(self.display, FOOD_COLOR, pygame.Rect(
            self.food.x*self.bloc_size, self.food.y*self.bloc_size, self.bloc_size, self.bloc_size))

        text = self.font.render(f'Score: {self._score}', True, TEXT_COLOR)
        self.display.blit(text, [10, 10])

        pygame.display.flip()

        speed = INITIAL_SPEED + (len(self.snake) / 3)
        self.clock.tick(speed)

    def must_quit(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
        return False

    def close(self):
        pygame.quit()
        quit()

    def snake_length(self) -> int:
        return len(self.snake)

    def _place_food(self):
        x = random.randint(0, self.w-1)
        y = random.randint(0, self.h-1)
        self.food = Point(x, y)
        while self.food in self.snake:
            self._place_food()

    def score(self):
        return self._score

    def state_size(self):
        return (SCAN_DISTANCE*2+1)**2 + STATE_SIZE

    def _state(self):
        dist_ = SCAN_DISTANCE*2+1
        scan = np.zeros((dist_, dist_))
        for i in range(dist_):
            for j in range(dist_):
                scan[i, j] = self._has_collide(
                    Point(self.head.x+(j-SCAN_DISTANCE), self.head.y+(i-SCAN_DISTANCE)))

        state = np.zeros(STATE_SIZE)

        state[0] = -1 if self.food.x < self.head.x else (
            0 if self.food.x == self.head.x else 1)
        state[1] = -1 if self.food.y < self.head.y else (
            0 if self.food.y == self.head.y else 1)

        state[2] = self._get_smallest_distance(self.head, self.food)

        state[3] = self.direction.value

        state = np.concatenate((scan.flatten(), state))
        return state.tolist()

    def _infos(self):
        return {
            'screen_dim': (self.w, self.h),
            'game_over': self.game_over,
            'score': self._score,
            'direction': self.direction,
            'head': self.head,
            'snake_length': len(self.snake),
            'food_pos': self.food,
        }

    def _reward(self, old_score):
        reward = PAIN_OF_MOVEMENT

        if self.game_over:  # bite his body?
            reward = -MAJOR_REWARD
        elif old_score < self._score:  # got the food?
            reward = MAJOR_REWARD
        else:
            for i in range(1, SCAN_DISTANCE):  # at a distance from danger?
                dir_ = self._get_direction_point(self.direction)
                dir_ = Point(dir_.x * i, dir_.y * i)

                pos = self._get_new_position(self.head, dir_)
                if self._has_collide(pos):
                    reward = -MAJOR_REWARD / (i+1)
                    break

            if reward == PAIN_OF_MOVEMENT:  # getting near or far from food?
                dist = self._get_smallest_distance(self.food, self.head)

                dir_ = self._get_direction_point(self.direction)
                dist_ = self._get_smallest_distance(
                    self.food, self._get_new_position(self.head, dir_, reverse=True))

                if dist_ < dist:
                    reward = -0.5
                elif dist_ > dist:
                    reward = -0.1

        return reward

    def _get_smallest_distance(self, pt1: Point, pt2: Point):
        pt11 = self._get_new_position(pt1, pt1, reverse=True)
        pt21 = self._get_new_position(pt2, pt1, reverse=True)
        dist1 = abs(pt11.x - pt21.x) + abs(pt11.y - pt21.y)

        pt12 = self._get_new_position(pt1, pt2, reverse=True)
        pt22 = self._get_new_position(pt2, pt2, reverse=True)
        dist2 = abs(pt12.x - pt22.x) + abs(pt12.y - pt22.y)

        return min(dist1, dist2)

    def _invalid_direction(self, old_dir, new_dir):
        return (old_dir == Direction.RIGHT and new_dir == Direction.LEFT) \
            or (old_dir == Direction.LEFT and new_dir == Direction.RIGHT) \
            or (old_dir == Direction.DOWN and new_dir == Direction.UP) \
            or (old_dir == Direction.UP and new_dir == Direction.DOWN)

    def _update_position(self, direction):
        if not self._invalid_direction(self.direction, direction):
            self.direction = direction

        dir_ = self._get_direction_point(self.direction)
        self.head = self._get_new_position(self.head, dir_)

    def _get_direction_point(self, direction: Direction):
        x, y = 0, 0

        if direction == Direction.RIGHT:
            x = 1
        elif direction == Direction.LEFT:
            x = -1
        elif direction == Direction.DOWN:
            y = 1
        elif direction == Direction.UP:
            y = -1

        return Point(x, y)

    def _get_new_position(self, pt: Point, direction: Point, reverse: bool = False):
        x, y = pt

        factor = -1 if reverse else 1

        x += direction.x * factor
        y += direction.y * factor

        if x < 0:
            x = self.w - x
        elif x >= self.w:
            x %= self.w

        if y < 0:
            y = self.h - y
        elif y >= self.h:
            y %= self.h

        return Point(x, y)

    def _has_collide(self, pt: Point):
        # no more collision on the sides
        # return pt.x < 0 \
        #     or pt.x >= self.w \
        #     or pt.y < 0 \
        #     or pt.y >= self.h \
        return pt in self.snake[1:]
