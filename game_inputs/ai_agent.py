import os
import random
import time

import torch
from torch import nn
from torch.nn import functional as F

from game_inputs import BaseAgent, Direction

SAVE_FOLDER = 'data/models'

REPLAY_MEMORY_SIZE = 100000
BATCH_SIZE = 1000

MIN_EXPLOITATION_RATE = 0.01
MAX_EXPLOITATION_RATE = 1.0
EXPLOITATION_DECAY_RATE = 0.01

MAX_EPISODES = 10000
STEPS_PER_BLOC = 200

DISCOUNT_RATE = 0.9
LEARNING_RATE = 0.001

torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')


class QModel(nn.Module):
    def __init__(self, ins, fc1s, fc2s, outs):
        super().__init__()

        self.fc1 = nn.Linear(ins, fc1s)
        self.fc2 = nn.Linear(fc1s, fc2s)
        self.fc3 = nn.Linear(fc2s, outs)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def save(self, filename):
        if not os.path.exists(SAVE_FOLDER):
            os.makedirs(SAVE_FOLDER)

        torch.save(self.state_dict(), f'{SAVE_FOLDER}/{filename}')

    def load(self, filename):
        filepath = f'{SAVE_FOLDER}/{filename}'
        if os.path.isfile(filepath):
            self.load_state_dict(torch.load(f'{SAVE_FOLDER}/{filename}'))
            return True

        return False


class QTrainer:
    def __init__(self, model):
        self.model = model
        self.criterion = torch.nn.MSELoss()
        self.optim = torch.optim.Adam(
            self.model.parameters(), lr=LEARNING_RATE)

        self.replay_memory = []

    def model(self):
        return self.model

    def memorize(self, state, action, reward, next_state, done):
        self.replay_memory.append((state, action, reward, next_state, done))
        if len(self.replay_memory) > REPLAY_MEMORY_SIZE:
            self.replay_memory.pop(0)

    def train(self, last_step_only=False):
        batch = []
        if not last_step_only:
            if len(self.replay_memory) < BATCH_SIZE:
                return

            batch = random.sample(self.replay_memory[:-1], BATCH_SIZE-1)

        batch.append(self.replay_memory[-1])
        batch_size = len(batch)

        states, actions, rewards, next_states, done = [
            torch.tensor(x, dtype=torch.float) for x in zip(*batch)]

        actions = actions.type(torch.int)
        idxs = list(range(batch_size))
        q_values = self.model.forward(states)[idxs, actions]

        q_target = rewards + DISCOUNT_RATE * done * \
            torch.max(self.model.forward(next_states), dim=1)[0]

        self.optim.zero_grad()
        loss = self.criterion(q_target, q_values)
        loss.backward()
        self.optim.step()


class AIAgent(BaseAgent):
    def __init__(self, game, model_filename: str = None, render: bool = None):
        super().__init__(game)
        self.model = QModel(game.state_size(), 512, 128, len(Direction))
        self.trainer = QTrainer(self.model)

        self.step = 0
        self.episode = 0
        self.high_score = 0

        self.exploration_rate = MAX_EXPLOITATION_RATE
        self.render = False

        if model_filename and self.model.load(model_filename):
            self.high_score = int(model_filename.split('.')[0].split('_')[-1])
            self.exploration_rate = MIN_EXPLOITATION_RATE
            self.render = True

        if render is not None:
            self.render = render

    def update(self):
        # stop if last episode
        game = self.game()
        if game.must_quit() or self.episode >= MAX_EPISODES:
            self.stop_game()
            return

        # reset game if start of episode
        if self.step == 0:
            state, _ = game.reset()
            self.current_state = state

        self.step += 1

        action, exploration = self._get_action(
            torch.tensor([self.current_state]))

        next_state, reward, done, _ = game.step(action)
        steps_limit = STEPS_PER_BLOC * game.snake_length()
        if (self.step >= steps_limit) and (reward > -5):
            reward = -5

        self.trainer.memorize(self.current_state, action.value,
                              reward, next_state, done)

        if reward in [-10, 10]:
            print(
                f'episode: {self.episode}, step: {self.step} => reward: {reward}, done: {done}, exploration: {exploration}, exploration_rate: {round(self.exploration_rate, 2)}')

        self.trainer.train(last_step_only=True)
        self.current_state = next_state

        # episode is over, start a new episode
        if done or self.step >= steps_limit:
            self.step = 0
            self.episode += 1

            if game.score() > self.high_score:
                self.high_score = game.score()
                filename = f'model_{int(time.time())}_{self.high_score}.pth'
                self.model.save(filename)

            self.trainer.train()

            self.exploration_rate -= EXPLOITATION_DECAY_RATE
            if self.exploration_rate < MIN_EXPLOITATION_RATE:
                self.exploration_rate = MIN_EXPLOITATION_RATE

        if self.render:
            game.render()

    def _get_action(self, state):
        exploration = False
        if random.random() > self.exploration_rate:
            with torch.no_grad():
                action = self.model.forward(state).argmax().item()
                action = Direction(action)
        else:
            action = random.choice(list(Direction))
            exploration = True

        return action, exploration
