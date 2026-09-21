import pygame as pg
from game import Game

def main():
    pg.init()
    screen = pg.display.set_mode((400, 600))
    pg.display.set_caption("Frog DQN AI - Dodge the Logs!")
    game = Game(screen)
    clock = pg.time.Clock()

    running = True
    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False

        game.update()
        pg.display.flip()
        clock.tick(60)

    pg.quit()

if __name__ == "__main__":
    main()
