import numpy as np
import random

class ChessEnv:
    def __init__(self, board_size=8):
        self.board_size = board_size
        self.agent_pos = 0
        self.goal_pos = self.board_size - 1
        self.enemy_pos = self.board_size // 2

    def step(self, action):
        if action == "forward":
            self.agent_pos = min(self.agent_pos + 1, self.board_size - 1)
        elif action == "backward":
            self.agent_pos = max(self.agent_pos - 1, 0)

        reward = 0
        done = False

        if self.agent_pos == self.goal_pos:
            reward = 1
            done = True
            print("Агент достиг цели!")
        elif self.agent_pos == self.enemy_pos:
            reward = -1
            print("Агент столкнулся с противником!")

        return self.agent_pos, reward, done

    def reset(self):
        self.agent_pos = 0
        return self.agent_pos

    def render(self):
        board = ['-' for _ in range(self.board_size)]
        board[self.agent_pos] = 'A'
        board[self.enemy_pos] = 'E'
        board[self.goal_pos] = 'G'
        print(" ".join(board))

def random_search(env, num_iterations):
    best_reward = float("-inf")
    best_policy = None

    for _ in range(num_iterations):
        policy = {i: random.choice(["forward", "backward", "stay"]) for i in range(env.board_size)}
        total_reward = 0

        for _ in range(100):
            state = env.reset()
            done = False

            while not done:
                action = policy[state]
                state, reward, done = env.step(action)
                total_reward += reward

        average_reward = total_reward / 100

        if average_reward > best_reward:
            best_reward = average_reward
            best_policy = policy

    return best_policy

def play_game(policy):
    env = ChessEnv(board_size=8)
    state = env.reset()
    done = False

    while not done:
        print("\n-----------------------------")
        env.render()
        action = policy[state]
        state, reward, done = env.step(action)
        print(f"Агент перешел в позицию {state}, Награда: {reward}")
    print("\n-----------------------------")
    env.render()

# Пример использования случайного поиска гиперпараметров
best_policy = random_search(ChessEnv(board_size=8), num_iterations=1000)
print("Лучшая политика:", best_policy)
play_game(best_policy)
