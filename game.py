import pygame, random, os, json, threading
import torch
import torch.nn as nn
import numpy as np
import sounddevice as sd
import librosa
import noisereduce as nr

pygame.init()
pygame.mixer.init()

# ================= SETTINGS =================
BASE_DIR = r"C:\Users\Dell\CarGame"  # Update path
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 660
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("2D Car Racing Game")

FONT = pygame.font.SysFont("Arial", 26)
BIG_FONT = pygame.font.SysFont("Arial", 55)
MID_FONT = pygame.font.SysFont("Arial", 32)

# ================= COLORS =================
WHITE = (255,255,255)
BLACK = (0,0,0)
GREEN = (0,180,0)
YELLOW = (200,200,0)
RED = (200,0,0)
GRAY = (50,50,50)
ROAD = (50,50,50)
ROAD_LINE = (255,255,255)
BLUE = (0,120,255)

# ================= HIGH SCORE =================
HS_FILE = os.path.join(BASE_DIR, "highscores.json")
def load_highscores():
    if os.path.exists(HS_FILE):
        with open(HS_FILE,"r") as f:
            return json.load(f)
    return {}
def save_highscores(data):
    with open(HS_FILE,"w") as f:
        json.dump(data,f)

# ================= SAFE IMAGE LOADER =================
def load_img(path,size=None):
    try:
        img = pygame.image.load(os.path.join(BASE_DIR,path))
        if size: img = pygame.transform.scale(img,size)
        return img
    except:
        surf = pygame.Surface(size if size else (100,100))
        surf.fill(GRAY)
        return surf

# ================= ASSETS =================
WELCOME_BG = load_img("assets/backgrounds/welcome.png",(SCREEN_WIDTH,SCREEN_HEIGHT))
DASHBOARD_BG = load_img("assets/backgrounds/dashboard.png",(SCREEN_WIDTH,SCREEN_HEIGHT))
try:
    player_img = pygame.image.load(os.path.join(BASE_DIR,"assets/cars/player_car.png"))
    player_img = pygame.transform.scale(player_img,(90,180))
except:
    player_img = pygame.Surface((90,180))
    player_img.fill((255,0,0))

obs_imgs = {
    "car": load_img("assets/cars/otherCar.png",(90,180)),
    "tree": load_img("assets/obstacles/tree.png",(80,160)),
    "block": load_img("assets/obstacles/block.png",(100,60)),
}
try:
    CRASH_SOUND = pygame.mixer.Sound(os.path.join(BASE_DIR,"assets/carCrash.wav"))
except:
    CRASH_SOUND = None

clock = pygame.time.Clock()

# ================= BUTTON CLASS =================
class Button:
    def __init__(self,text,x,y,w,h,color):
        self.text=text
        self.rect=pygame.Rect(x,y,w,h)
        self.color=color
    def draw(self):
        pygame.draw.rect(SCREEN,self.color,self.rect,border_radius=12)
        pygame.draw.rect(SCREEN,WHITE,self.rect,3,border_radius=12)
        t=FONT.render(self.text,True,BLACK)
        SCREEN.blit(t,(self.rect.centerx-t.get_width()//2,self.rect.centery-t.get_height()//2))
    def clicked(self,pos):
        return self.rect.collidepoint(pos)

# ================= VOICE MODEL =================
SR = 16000
DURATION = 1.0
N_MFCC = 13
MAX_LEN = 110
COMMANDS = ["left","right","pause"]
CONFIDENCE_THRESHOLD = 0.50
VOLUME_THRESHOLD = 0.01

class VoiceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(N_MFCC,128,batch_first=True)
        self.fc = nn.Linear(128,len(COMMANDS))
    def forward(self,x):
        out,_ = self.lstm(x)
        return self.fc(out[:, -1, :])

device = torch.device("cpu")
voice_model = VoiceModel().to(device)
voice_model.load_state_dict(torch.load(os.path.join(BASE_DIR,"voice_model_best.pth"),map_location=device))
voice_model.eval()

# ================= AUDIO RECORDING =================
def record_audio(duration=DURATION,sr=SR):
    audio = sd.rec(int(duration*sr),samplerate=sr,channels=1)
    sd.wait()
    audio = audio.flatten()
    if np.max(np.abs(audio))<VOLUME_THRESHOLD:
        return None
    audio = audio - np.mean(audio)
    audio = audio / (np.max(np.abs(audio))+1e-6)
    audio = nr.reduce_noise(y=audio, sr=SR)
    return audio

def predict_command(audio):
    if audio is None: return None
    mfcc = librosa.feature.mfcc(y=audio,sr=SR,n_mfcc=N_MFCC,n_fft=512,hop_length=160).T
    if mfcc.std()<1e-6:
        mfcc = mfcc-mfcc.mean()
    else:
        mfcc = (mfcc-mfcc.mean())/(mfcc.std()+1e-6)
    if mfcc.shape[0]<MAX_LEN:
        mfcc = np.pad(mfcc,((0,MAX_LEN-mfcc.shape[0]),(0,0)))
    else:
        start=(mfcc.shape[0]-MAX_LEN)//2
        mfcc = mfcc[start:start+MAX_LEN]
    X = torch.tensor(mfcc,dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        out = voice_model(X)
        probs = torch.softmax(out,dim=1)
        confidence,pred_idx = torch.max(probs,dim=1)
        if confidence.item()<CONFIDENCE_THRESHOLD:
            return None
        return COMMANDS[pred_idx.item()]

# ================= VOICE LISTENER THREAD =================
def voice_listener(game_instance):
    last_cmd = None
    while True:
        audio = record_audio()
        cmd = predict_command(audio)
        if cmd in ["left","right","pause"]:
            last_cmd = cmd
            game_instance.voice_command = last_cmd
        # keep previous command until user speaks new one

# ================= GAME CLASS =================
class Game:
    def __init__(self):
        self.state="WELCOME"
        self.highscores=load_highscores()
        self.reset_game()
        self.voice_command=None
        self.command_timer=0  # cooldown to avoid jitter

    def reset_game(self):
        self.score=0
        self.speed=6
        self.road_w=SCREEN_WIDTH//1.6
        self.road_left=SCREEN_WIDTH//2-self.road_w//2
        self.road_right=SCREEN_WIDTH//2+self.road_w//2
        self.left_lane=self.road_left+self.road_w//4
        self.right_lane=self.road_right-self.road_w//4
        self.player_rect=player_img.get_rect(center=(self.left_lane,SCREEN_HEIGHT-140))
        self.obstacles=[]
        self.spawn_timer=0
        self.min_spawn_delay=80
        self.road_scroll=0
        self.paused=False
        self.mode=""
        self.sub_level=1

    def run(self):
        running=True
        while running:
            for e in pygame.event.get():
                if e.type==pygame.QUIT:
                    running=False
                if e.type==pygame.MOUSEBUTTONDOWN:
                    self.handle_mouse(e.pos)
                if e.type==pygame.KEYDOWN:
                    self.handle_keys(e.key)
            if self.state=="PLAY":
                self.update_game()
            self.draw()
            pygame.display.update()
            clock.tick(60)
        pygame.quit()

    def handle_keys(self,key):
        if self.state=="PLAY" and key==pygame.K_SPACE:
            self.paused=not self.paused
        if self.state=="GAME_OVER" and key==pygame.K_SPACE:
            self.state="DASHBOARD"

    def start_game(self,difficulty,sub):
        self.reset_game()
        self.mode=difficulty
        self.sub_level=sub
        if difficulty=="EASY":
            self.speed=6+sub*0.5
            self.min_spawn_delay=90
        elif difficulty=="MEDIUM":
            self.speed=8+sub*0.6
            self.min_spawn_delay=70
        else:
            self.speed=10+sub*0.8
            self.min_spawn_delay=50
        self.state="PLAY"

    def update_game(self):
        if self.paused: return
        self.score+=1
        self.road_scroll=(self.road_scroll+self.speed)%40

        # ===== VOICE COMMAND CONTROL =====
        if self.command_timer>0:
            self.command_timer-=1

        if self.voice_command and self.command_timer==0:
            if self.voice_command=="left":
                self.player_rect.centerx=self.left_lane
                self.command_timer=18
            elif self.voice_command=="right":
                self.player_rect.centerx=self.right_lane
                self.command_timer=18
            elif self.voice_command=="pause":
                self.paused=not self.paused
                self.command_timer=18

        # ===== OBSTACLES =====
        for o in self.obstacles: o[1].y+=self.speed
        self.obstacles=[o for o in self.obstacles if o[1].y<SCREEN_HEIGHT+200]

        self.spawn_timer+=1
        if self.spawn_timer>self.min_spawn_delay:
            lane=random.choice([self.left_lane,self.right_lane])
            choices=["car"]
            if self.mode=="HARD": choices+=["tree","block"]
            img=obs_imgs[random.choice(choices)]
            self.obstacles.append([img,img.get_rect(center=(lane,-200))])
            self.spawn_timer=0

        self.check_collision()

    def check_collision(self):
        for o in self.obstacles:
            if self.player_rect.inflate(-20,-20).colliderect(o[1]):
                key=f"{self.mode}_L{self.sub_level}"
                best=self.highscores.get(key,0)
                if self.score>best:
                    self.highscores[key]=self.score
                    save_highscores(self.highscores)
                if CRASH_SOUND: CRASH_SOUND.play()
                self.state="GAME_OVER"

    # ================= DRAW FUNCTIONS =================
    def draw(self):
        if self.state=="WELCOME": self.draw_welcome()
        elif self.state=="DASHBOARD": self.draw_dashboard()
        elif self.state=="HIGH_SCORES": self.draw_high_scores()
        elif self.state in ["EASY_MENU","MEDIUM_MENU","HARD_MENU"]: self.draw_sub_menu()
        elif self.state=="PLAY": self.draw_game()
        elif self.state=="GAME_OVER": self.draw_game_over()

    def draw_game(self):
        SCREEN.fill(GREEN)
        pygame.draw.rect(SCREEN,ROAD,(self.road_left,0,self.road_w,SCREEN_HEIGHT))
        for i in range(-1,SCREEN_HEIGHT//40+2):
            y=i*40+self.road_scroll
            pygame.draw.line(SCREEN,ROAD_LINE,(SCREEN_WIDTH//2,y),(SCREEN_WIDTH//2,y+20),5)
        for o in self.obstacles: SCREEN.blit(o[0],o[1])
        SCREEN.blit(player_img,self.player_rect)
        key=f"{self.mode}_L{self.sub_level}"
        best=self.highscores.get(key,0)
        self.draw_text(f"SCORE: {self.score}",FONT,WHITE,680,40)
        self.draw_text(f"BEST: {best}",FONT,YELLOW,680,80)
        if self.paused: self.draw_text("PAUSED",BIG_FONT,YELLOW,400,330)

    def draw_text(self,t,f,c,x,y):
        img=f.render(t,True,c)
        SCREEN.blit(img,(x-img.get_width()//2,y-img.get_height()//2))

    def draw_welcome(self):
        SCREEN.blit(WELCOME_BG,(0,0))
        self.draw_text("2D CAR RACING",BIG_FONT,WHITE,400,200)
        start_btn.draw()

    def draw_dashboard(self):
        SCREEN.blit(DASHBOARD_BG,(0,0))
        self.draw_text("SELECT MODE",BIG_FONT,WHITE,400,120)
        easy_btn.draw(); med_btn.draw(); hard_btn.draw(); highscore_btn.draw()

    def draw_sub_menu(self):
        SCREEN.blit(DASHBOARD_BG,(0,0))
        self.draw_text("SELECT LEVEL",BIG_FONT,WHITE,400,130)
        for b in sub_buttons: b.draw()
        back_btn.draw()

    def draw_high_scores(self):
        SCREEN.fill(BLACK)
        self.draw_text("HIGH SCORES",BIG_FONT,YELLOW,400,80)
        y=150
        for mode in ["EASY","MEDIUM","HARD"]:
            self.draw_text(mode,MID_FONT,BLUE,400,y)
            y+=30
            for i in range(5):
                key=f"{mode}_L{i+1}"
                score=self.highscores.get(key,0)
                self.draw_text(f"L{i+1}: {score}",FONT,WHITE,400,y)
                y+=25
            y+=20
        back_btn.draw()

    def draw_game_over(self):
        SCREEN.fill(BLACK)
        self.draw_text("GAME OVER",BIG_FONT,RED,400,260)
        self.draw_text(f"SCORE: {self.score}",FONT,WHITE,400,320)
        self.draw_text("PRESS SPACE TO RESTART",FONT,WHITE,400,400)

    def handle_mouse(self,pos):
        if self.state=="WELCOME":
            if start_btn.clicked(pos): self.state="DASHBOARD"
        elif self.state=="DASHBOARD":
            if easy_btn.clicked(pos): self.state="EASY_MENU"
            if med_btn.clicked(pos): self.state="MEDIUM_MENU"
            if hard_btn.clicked(pos): self.state="HARD_MENU"
            if highscore_btn.clicked(pos): self.state="HIGH_SCORES"
        elif self.state in ["EASY_MENU","MEDIUM_MENU","HARD_MENU"]:
            for i,b in enumerate(sub_buttons):
                if b.clicked(pos): self.start_game(self.state.replace("_MENU",""),i+1)
            if back_btn.clicked(pos): self.state="DASHBOARD"
        elif self.state=="HIGH_SCORES":
            if back_btn.clicked(pos): self.state="DASHBOARD"

# ================= BUTTONS =================
start_btn=Button("START",300,420,200,60,GREEN)
easy_btn=Button("EASY",300,230,200,60,GREEN)
med_btn=Button("MEDIUM",300,310,200,60,YELLOW)
hard_btn=Button("HARD",300,390,200,60,RED)
highscore_btn=Button("HIGH SCORES",300,470,200,60,BLUE)
back_btn=Button("BACK",20,20,120,45,GRAY)
sub_buttons=[Button(f"L{i+1}",100+i*130,320,110,55,(100,100,255)) for i in range(5)]

# ================= RUN =================
if __name__=="__main__":
    game_instance = Game()
    threading.Thread(target=voice_listener,args=(game_instance,),daemon=True).start()
    game_instance.run()
