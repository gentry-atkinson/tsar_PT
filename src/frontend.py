import pygame as pg

screen = None
running = True

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800


def setup():
    print('starting')
    pg.init()
    screen = pg.display.set_mode([SCREEN_WIDTH,SCREEN_HEIGHT])

def main_loop():
    for event in pg.event.get():
        if event.type == pg.QUIT:
            print('exit')
            running = False

if __name__ == '__main__':
    setup()
    while(running):
        main_loop()
    pg.quit()