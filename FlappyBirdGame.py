import pygame
import os
import random
import neat

ia_playing = True
generation = 0

SCREEN_WIDTH = 500
SCREEN_HEIGHT = 650

IMAGE_PIPE = pygame.transform.scale2x(pygame.image.load(os.path.join('imgs', 'pipe.png')))
IMAGE_FLOOR = pygame.transform.scale2x(pygame.image.load(os.path.join('imgs', 'base.png')))
IMAGE_BACKGROUND = pygame.transform.scale2x(pygame.image.load(os.path.join('imgs', 'bg.png')))
IMAGE_BIRDS = [
    pygame.image.load(os.path.join('imgs', 'bird1.png')),
    pygame.image.load(os.path.join('imgs', 'bird2.png')),
    pygame.image.load(os.path.join('imgs', 'bird3.png')),
    '''
    pygame.transform.scale(pygame.image.load(os.path.join('imgs', 'bird1.png'))),
    pygame.transform.scale(pygame.image.load(os.path.join('imgs', 'bird2.png'))),
    pygame.transform.scale(pygame.image.load(os.path.join('imgs', 'bird3.png')))
    '''
]

pygame.font.init()
FONT_SCORE = pygame.font.SysFont('arial', 50)


class Bird:
    IMGS = IMAGE_BIRDS
    # rotation animations
    MAX_ROTATION = 25
    SPEED_ROTATION = 20
    TIME_ANIMATION = 5

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.angle = 0
        self.speed = 0
        self.height = self.y
        self.time = 0
        self.counter_images = 0
        self.image = self.IMGS[0]

    def jump(self):
        self.speed = -10.5
        self.time = 0
        self.height = self.y

    def move(self):
        # calculate the moviments
        self.time += 1
        displacement = 1.5 * (self.time ** 2) + self.speed * self.time

        # restrict the displacement
        if displacement > 16:
            displacement = 16
        elif displacement < 0:
            displacement -= 2

        self.y += displacement

        # bird angle
        if displacement < 0 or self.y < (self.height + 50):
            if self.angle < self.MAX_ROTATION:
                self.angle = self.MAX_ROTATION
        else:
            if self.angle > -90:
                self.angle -= self.SPEED_ROTATION

    def draw(self, screen):
        # define which image we gonna use
        self.counter_images += 1

        if self.counter_images < self.TIME_ANIMATION:
            self.image = self.IMGS[0]
        elif self.counter_images < self.TIME_ANIMATION*2:
            self.image = self.IMGS[1]
        elif self.counter_images < self.TIME_ANIMATION*3:
            self.image = self.IMGS[2]
        elif self.counter_images < self.TIME_ANIMATION*4:
            self.image = self.IMGS[1]
        elif self.counter_images < self.TIME_ANIMATION*4 + 1:
            self.image = self.IMGS[0]
            self.counter_images = 0

        # if bird is falling it won't flap wings
        if self.angle <= -80:
            self.image = self.IMGS[1]
            self.counter_images = self.TIME_ANIMATION*2

        # draw image
        rotate_image = pygame.transform.rotate(self.image, self.angle)
        pos_center_image = self.image.get_rect(topleft=(self.x, self.y)).center
        rectangle = rotate_image.get_rect(center=pos_center_image)
        screen.blit(rotate_image, rectangle.topleft)

    def get_mask(self):
        return pygame.mask.from_surface(self.image)


class Pipe:
    DISTANCE = 200
    SPEED = 5

    def __init__(self, x):
        self.x = x
        self.height = 0
        self.pos_top = 0
        self.pos_base = 0
        self.PIPE_TOP = pygame.transform.flip(IMAGE_PIPE, False, True)
        self.PIPE_BASE = IMAGE_PIPE
        self.passed = False
        self.define_height()

    def define_height(self):
        self.height = random.randint(50, 350)
        self.pos_top = self.height - self.PIPE_TOP.get_height()
        self.pos_base = self.height + random.randint(self.DISTANCE - 50, self.DISTANCE + 100)

    def move(self):
        self.x -= self.SPEED

    def draw(self, screen):
        screen.blit(self.PIPE_TOP, (self.x, self.pos_top))
        screen.blit(self.PIPE_BASE, (self.x, self.pos_base))

    def collision(self, bird):
        bird_mask = bird.get_mask()
        mask_top = pygame.mask.from_surface(self.PIPE_TOP)
        mask_base = pygame.mask.from_surface(self.PIPE_BASE)

        distance_top = (self.x - round(bird.x), self.pos_top - round(bird.y))
        distance_base = (self.x - round(bird.x), self.pos_base - round(bird.y))

        point_top = bird_mask.overlap(mask_top, distance_top)
        point_base = bird_mask.overlap(mask_base, distance_base)

        if point_base or point_top:
            return True
        else:
            return False


class Floor:
    SPEED = 5
    WIDTH = IMAGE_FLOOR.get_width()
    IMAGE = IMAGE_FLOOR

    def __init__(self, y):
        self.y = y
        self.x0 = 0
        self.x1 = self.WIDTH

    def move(self):
        self.x0 -= self.SPEED
        self.x1 -= self.SPEED

        if self.x0 + self.WIDTH < 0:
            self.x0 = self.x1 + self.WIDTH
        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x0 + self.WIDTH

    def draw(self, screen):
        screen.blit(self.IMAGE, (self.x0, self.y))
        screen.blit(self.IMAGE, (self.x1, self.y))


def draw_screen(screen, birds, pipes, floor, score):
    screen.blit(IMAGE_BACKGROUND, (0, 0))
    for bird in birds:
        bird.draw(screen)
    for pipe in pipes:
        pipe.draw(screen)

    txt = FONT_SCORE.render(f"Score: {score}", 1, (255, 255, 255))
    screen.blit(txt, (SCREEN_WIDTH - 10 - txt.get_width(), 10))

    if ia_playing:
        txt = FONT_SCORE.render(f"Generation: {generation}", 1, (255, 255, 255))
        screen.blit(txt, (10, 10))

    floor.draw(screen)
    pygame.display.update()


def main(genomes, config): # Fitness function
    global generation
    generation += 1

    if ia_playing:
        redes = []
        genome_list = []
        birds = []
        for _, genome in genomes:
            rede = neat.nn.FeedForwardNetwork.create(genome, config)
            redes.append(rede)
            genome.fitness = 0
            genome_list.append(genome)
            birds.append(Bird(230, 200))
    else:
        birds = [Bird(230, 200)]

    floor = Floor(600)
    pipes = [Pipe(500)]
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    score = 0
    clock = pygame.time.Clock()

    playing = True
    while playing:
        clock.tick(30)

        # Interaction user
        for events in pygame.event.get():
            if events.type == pygame.QUIT:
                playing = False
                pygame.quit()
                quit()
            if not ia_playing:
                if events.type == pygame.KEYDOWN:
                    if events.key == pygame.K_SPACE:
                        for bird in birds:
                            bird.jump()
        index_pipe = 0
        if len(birds) > 0: # Find out which pipe should look at
            if len(birds) > 1 and birds[0].x > (pipes[0].x + pipes[0].PIPE_TOP.get_width()):
                index_pipe = 1
        else:
            playing = False
            break

        # Move
        for i, bird in enumerate(birds):
            bird.move()
            # Increase bird fitness and jump
            genome_list[i].fitness += 0.1
            output = redes[i].activate((bird.y,
                                        abs(bird.y - pipes[index_pipe].height),
                                        abs(bird.y - pipes[index_pipe].pos_base)))
            if output[0] > 0.5:
                bird.jump()
        floor.move()

        add_pipe = False
        remove_pipes = []
        for pipe in pipes:
            for i, bird in enumerate(birds):
                if pipe.collision(bird):
                    birds.pop(i)
                    if ia_playing:
                        genome_list[i].fitness -= 1
                        genome_list.pop(i)
                        redes.pop(i)
                if not pipe.passed and bird.x > pipe.x:
                    pipe.passed = True
                    add_pipe = True
            pipe.move()
            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                remove_pipes.append(pipe)

        if add_pipe:
            score += 1
            pipes.append(Pipe(600))
            for genome in genome_list:
                genome.fitness += 5
        for pipe in remove_pipes:
            pipes.remove(pipe)

        for i, bird in enumerate(birds):
            if (bird.y + bird.image.get_height()) > floor.y or bird.y < 0:
                birds.pop(i)
                if ia_playing:
                    genome_list.pop(i)
                    redes.pop(i)

        draw_screen(screen, birds, pipes, floor, score)


def load(way_config):
    config = neat.config.Config(neat.DefaultGenome,
                                neat.DefaultReproduction,
                                neat.DefaultSpeciesSet,
                                neat.DefaultStagnation,
                                way_config)

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())

    if ia_playing:
        population.run(main, 50)
    else:
        main(None, None)


if __name__ == '__main__':
    way = os.path.dirname(__file__)
    way_config = os.path.join(way, 'config.txt')
    load(way_config)
