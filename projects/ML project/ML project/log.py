import pygame as pg

class Log:
    def __init__(self, screen_width, column, speed):
        self.num_columns = 4
        self.width = screen_width // self.num_columns
        self.height = 70
        self.column = column
        self.x = self.column * self.width
        self.y = -self.height
        self.speed = speed

    def move(self):
        self.y += self.speed

    def show(self, screen):
        #log body
        log_color = (101, 67, 33)
        bark_color = (139, 90, 43)
        pg.draw.rect(screen, log_color, (self.x + 4, self.y, self.width - 8, self.height), border_radius=12)

        # bark lines to make it look like a log
        for i in range(3):
            line_y = self.y + 15 + i * 18
            pg.draw.line(screen, bark_color, (self.x + 10, line_y), (self.x + self.width - 14, line_y), 2)

    def get_hitbox(self):
        return pg.Rect(self.x + 8, self.y + 5, self.width - 16, self.height - 10)
