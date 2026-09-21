import pygame as pg
import random
from log import Log
from qfrog import QFrog

class Game:
    def __init__(self, screen):
        self.screen = screen
        self.width, self.height = screen.get_size()
        self.frog = QFrog(self.width, self.height)
        self.logs = []
        self.spawn_time = 0
        self.score = 0
        self.high_score = 0
        self.font = pg.font.SysFont("Arial", 18, bold=True)

    def draw_background(self):
        # Draw river 
        colors = [(45, 130, 200), (50, 140, 210)]
        lane_w = self.width // 4
        for i in range(4):
            pg.draw.rect(self.screen, colors[i % 2], (i * lane_w, 0, lane_w, self.height))

        # Draw lily pad strips at top and bottom 
        pg.draw.rect(self.screen, (34, 120, 34), (0, 0, self.width, 40))
        pg.draw.rect(self.screen, (34, 120, 34), (0, self.height - 80, self.width, 80))

        # Draw small wave lines on the river
        wave_color = (60, 150, 220)
        for row in range(60, self.height - 90, 50):
            for col in range(10, self.width - 10, 40):
                pg.draw.arc(self.screen, wave_color,
                            (col, row, 25, 10), 0, 3.14, 2)

    def update(self):
        self.draw_background()

        state = self.frog.get_state(self.logs, self.height)
        action = self.frog.choose_action(state)
        self.frog.move(action)

        # Difficulty goes up as high score increases
        current_speed = 3.5 + (self.high_score / 15.0)
        spawn_delay   = max(450, 850 - (self.high_score * 4))

        # Spawn a new log 
        if pg.time.get_ticks() - self.spawn_time > spawn_delay:
            self.logs.append(Log(self.width, random.randint(0, 3), current_speed))
            self.spawn_time = pg.time.get_ticks()

        reward, done = 0.05, False
        frog_rect = self.frog.get_hitbox()

        for log in self.logs[:]:
            log.move()
            if log.get_hitbox().colliderect(frog_rect):
                # Frog got hit by a log
                reward, done = -30, True
            elif log.y > self.height:
                #frog dodged successfully
                self.logs.remove(log)
                self.score += 1
                reward = 10

        next_state = self.frog.get_state(self.logs, self.height)
        self.frog.memory.append((state, action, reward, next_state, done))
        self.frog.train_step()

        if done:
            # Frog died
            self.frog.episodes += 1
            self.high_score = max(self.score, self.high_score)
            self.score = 0
            self.logs = []
            self.frog.column = random.randint(0, 3)

        # Draw everything
        for log in self.logs:
            log.show(self.screen)
        self.frog.show(self.screen)
        self.draw_ui(current_speed)

    def draw_ui(self, speed):
        score_text = self.font.render(f"Score: {self.score}",         True, (255, 255, 255))
        best_text  = self.font.render(f"Best:  {self.high_score}",    True, (255, 230, 50))
        ep_text    = self.font.render(f"Episode: {self.frog.episodes}", True, (255, 255, 255))
        eps_text   = self.font.render(f"Epsilon: {self.frog.epsilon:.3f}", True, (200, 230, 255))

        self.screen.blit(score_text, (10, 8))
        self.screen.blit(best_text,  (200, 8))
        self.screen.blit(ep_text,    (10, self.height - 70))
        self.screen.blit(eps_text,   (10, self.height - 50))
