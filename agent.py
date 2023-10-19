import numpy as np
import torch
import random
from collections import deque
from snake import SnakeGameAI, Point, Direction

MAX_MEMORY = 1000_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self) -> None:
        self.n_game = 0
        self.epsilon = 0 # random rate
        self.gama = 0 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.trainer = None
        self.model = None
    
    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_d = Point(head.x, head.y + 20)
        point_u = Point(head.x, head.y - 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_d = game.direction == Direction.DOWN
        dir_u = game.direction == Direction.UP
        
        state = [
            # straight danger
            dir_r and game.is_collision(point_r),
            dir_l and game.is_collision(point_l),
            dir_d and game.is_collision(point_d),
            dir_u and game.is_collision(point_u),
            
            # right danger
            dir_r and game.is_collision(point_d),
            dir_l and game.is_collision(point_u),
            dir_d and game.is_collision(point_l),
            dir_u and game.is_collision(point_r),
            
            # left danger
            dir_r and game.is_collision(point_u),
            dir_l and game.is_collision(point_d),
            dir_d and game.is_collision(point_r),
            dir_u and game.is_collision(point_l),
            
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            game.head.x > game.food.x, # food left
            game.head.x < game.food.x, # food right
            game.head.y > game.food.y, # food down
            game.head.y < game.food.y # food up
        ]
        
        return np.array(state, dtype=int)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def train_long_memory(self):
        if (len(self.memory) > BATCH_SIZE):
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones) 
    
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
    
    def get_action(self, state):
        self.epsilon = 80 - self.n_game
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            idx = random.randint(0, 2)
            final_move[idx] = 1
        else:
            state_tensor = torch.tensor(state, dtype=torch.float)
            pred = self.model.predict(state)
            idx = torch.argmax(pred).item()
            final_move[idx] = 1
        return final_move
        
        
def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        state_old = agent.get_state(game)
        
        final_move = agent.get_action(state_old)
        
        reward, done, score = game.play_step(final_move)
        
        state_new = agent.get_state(game)
        
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)
        
        if done:
            game._reset()
            agent.n_game += 0
            agent.train_long_memory()
            
            if score > record: 
                record = score
                
            print("Game: ", agent.n_game, 'Score: ', game.score, 'Record: ', record)