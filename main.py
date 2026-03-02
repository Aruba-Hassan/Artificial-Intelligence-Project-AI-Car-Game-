import pygame
import os
from voice_controller import VoiceController

pygame.init()

# ================= SETTINGS =================
WIDTH, HEIGHT = 900, 500
FPS = 60

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(BASE_DIR, "assets")

WELCOME_IMG = os.path.join(ASSETS, "backgrounds", "welcome.png")
DASHBOARD_IMG = os.path.join(ASSETS, "backgrounds", "dashboard.png")
CAR_IMG = os.path.join(ASSETS, "cars", "car.png")

MODEL_PATH = os.path.join(BASE_DIR, "model", "voice_lstm.pth")

# ================= GAME CLASS =================
class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("AI Voice Car Game")
        self.clock = pygame.time.Clock()

        # Load assets
        self.welcome_bg = pygame.transform.scale(
            pygame.image.load(WELCOME_IMG), (WIDTH, HEIGHT)
        )
        self.dashboard_bg = pygame.transform.scale(
            pygame.image.load(DASHBOARD_IMG), (WIDTH, HEIGHT)
        )
        self.car = pygame.image.load(CAR_IMG)

        self.car_x = WIDTH // 2
        self.car_y = HEIGHT - 120
        self.speed = 5

        self.voice = VoiceController(MODEL_PATH)
        self.state = "WELCOME"

    # -------- STATES --------
    def show_welcome(self):
        self.screen.blit(self.welcome_bg, (0,0))
        pygame.display.update()

        for e in pygame.event.get():
            if e.type == pygame.KEYDOWN:
                self.state = "DASHBOARD"
            if e.type == pygame.QUIT:
                pygame.quit()

    def show_dashboard(self):
        self.screen.blit(self.dashboard_bg, (0,0))
        pygame.display.update()

        for e in pygame.event.get():
            if e.type == pygame.KEYDOWN:
                self.state = "PLAY"
            if e.type == pygame.QUIT:
                pygame.quit()

    def play(self):
        self.screen.fill((60,60,60))
        self.screen.blit(self.car, (self.car_x, self.car_y))
        pygame.display.update()

        command = self.voice.listen()

        if command == "left":
            self.car_x -= self.speed
        elif command == "right":
            self.car_x += self.speed

        self.car_x = max(0, min(self.car_x, WIDTH - self.car.get_width()))

    # -------- MAIN LOOP --------
    def run(self):
        while True:
            self.clock.tick(FPS)

            if self.state == "WELCOME":
                self.show_welcome()
            elif self.state == "DASHBOARD":
                self.show_dashboard()
            elif self.state == "PLAY":
                self.play()

# ================= RUN =================
if __name__ == "__main__":
    Game().run()