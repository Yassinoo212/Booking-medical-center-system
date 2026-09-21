import torch
import torch.optim as optim
import numpy as np
import random
import pygame as pg
from collections import deque
from model import DuelingDQN

class QFrog:
    def __init__(self, screen_width, screen_height):
        self.num_columns = 4
        self.col_width = screen_width // self.num_columns
        self.size = 36
        self.column = 1                        # start in column 1
        self.y = screen_height - self.size - 50

        # DQN setup
        self.state_size = 4
        self.action_size = 3                   # 0=left, 1=stay, 2=right
        self.model = DuelingDQN(self.state_size, self.action_size)
        self.target_model = DuelingDQN(self.state_size, self.action_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005)

        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.9994
        self.epsilon_min = 0.02
        self.tau = 0.01
        self.episodes = 0

    def soft_update(self):
        for tp, lp in zip(self.target_model.parameters(), self.model.parameters()):
            tp.data.copy_(self.tau * lp.data + (1.0 - self.tau) * tp.data)

    def get_state(self, logs, screen_height):
        # Default state if no logs on screen
        if not logs:
            return np.array([self.column / 3, 0.0, 1.0, 0.5], dtype=np.float32)

       
        upcoming = sorted(
            [l for l in logs if l.y < self.y + self.size],
            key=lambda l: abs(self.y - l.y)
        )
        l1 = upcoming[0] if upcoming else None
        l2 = upcoming[1] if len(upcoming) > 1 else l1

        if not l1:
            return np.array([self.column / 3, 0.0, 1.0, 0.5], dtype=np.float32)

        return np.array([
            self.column / 3,
            l1.column / 3,
            (self.y - l1.y) / screen_height,
            l2.column / 3 if l2 else 0.0
        ], dtype=np.float32)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 2)
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            return torch.argmax(self.model(state_t)).item()

    def train_step(self):
        if len(self.memory) < 128:
            return
        batch = random.sample(self.memory, 128)
        s, a, r, ns, d = zip(*batch)
        s_t  = torch.FloatTensor(np.array(s))
        ns_t = torch.FloatTensor(np.array(ns))
        a_t  = torch.LongTensor(a)
        r_t  = torch.FloatTensor(r)
        d_t  = torch.FloatTensor(d)

        curr_q = self.model(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)
        next_q = self.target_model(ns_t).max(1)[0]
        targets = r_t + (1 - d_t) * self.gamma * next_q

        loss = torch.nn.MSELoss()(curr_q, targets.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def move(self, action):
        if action == 0 and self.column > 0:
            self.column -= 1
        elif action == 2 and self.column < self.num_columns - 1:
            self.column += 1

    def show(self, screen):
        x = self.column * self.col_width + (self.col_width - self.size) // 2
        y = self.y

        # Draw frog body
        body_color  = (50, 180, 50)
        dark_green  = (30, 120, 30)
        eye_color   = (255, 255, 255)
        pupil_color = (0, 0, 0)

        # Body
        pg.draw.ellipse(screen, body_color, (x, y + 6, self.size, self.size - 6))
        pg.draw.ellipse(screen, dark_green, (x, y + 6, self.size, self.size - 6), 2)

        # Head bump
        pg.draw.ellipse(screen, body_color, (x + 4, y, self.size - 8, 18))

        # Eyes (two white circles with pupils)
        pg.draw.circle(screen, eye_color,   (x + 7,          y + 5), 6)
        pg.draw.circle(screen, eye_color,   (x + self.size - 7, y + 5), 6)
        pg.draw.circle(screen, pupil_color, (x + 7,          y + 5), 3)
        pg.draw.circle(screen, pupil_color, (x + self.size - 7, y + 5), 3)

        # Little legs sticking out
        pg.draw.line(screen, dark_green, (x,              y + self.size - 4), (x - 8, y + self.size + 4), 3)
        pg.draw.line(screen, dark_green, (x + self.size,  y + self.size - 4), (x + self.size + 8, y + self.size + 4), 3)

    def get_hitbox(self):
        x = self.column * self.col_width + (self.col_width - self.size) // 2
        return pg.Rect(x + 4, self.y + 6, self.size - 8, self.size - 6)