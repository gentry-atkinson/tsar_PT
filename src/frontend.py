import pygame as pg

screen = None
running = True

SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 800


def setup():
    print('starting')
    pg.init()
    screen = pg.display.set_mode([SCREEN_WIDTH,SCREEN_HEIGHT])
    logo = pg.image.load('imgs/branding/tsar_logo.png')
    screen.blit(logo, (0,0))

def main_loop():
    global running
    pg.display.update()
    for event in pg.event.get():
        if event.type == pg.QUIT:
            print('exit')
            running = False

if __name__ == '__main__':
    setup()
    while(running):
        main_loop()
    pg.display.quit()
    pg.quit()