"""
Simple Reinforcement Learning Example with Pygame
Q-Learning Agent that learns to catch falling objects

Controls:
- Run the script to watch the AI learn
- The agent will improve over episodes
- Close the window to exit
"""

import pygame
import random
import pickle
import os
import numpy as np
from collections import defaultdict

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 600, 400
PADDLE_WIDTH, PADDLE_HEIGHT = 80, 10
BALL_RADIUS = 10
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 100, 255)
RED = (255, 50, 50)
GREEN = (50, 255, 50)

# Q-Learning Parameters
ALPHA = 0.1  # Learning rate
GAMMA = 0.95  # Discount factor
EPSILON = 1.0  # Exploration rate
EPSILON_DECAY = 0.995  # Decay per episode
EPSILON_MIN = 0.01
LEARNING_EPISODES = 2000


class Paddle:
    def __init__(self):
        self.width = PADDLE_WIDTH
        self.height = PADDLE_HEIGHT
        self.x = WIDTH // 2 - self.width // 2
        self.y = HEIGHT - self.height - 5
        self.speed = 8
        self.color = BLUE

    def move(self, action):
        """
        Action: 0 = stay, 1 = move left, 2 = move right
        """
        if action == 1:  # Left
            self.x -= self.speed
        elif action == 2:  # Right
            self.x += self.speed

        # Keep paddle in bounds
        self.x = max(0, min(WIDTH - self.width, self.x))

    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.get_rect())


class Ball:
    def __init__(self):
        self.reset()
        self.color = RED

    def reset(self):
        self.x = random.randint(BALL_RADIUS, WIDTH - BALL_RADIUS)
        self.y = BALL_RADIUS
        self.speed_y = random.uniform(3, 6)
        self.speed_x = random.uniform(-2, 2)

    def update(self):
        self.x += self.speed_x
        self.y += self.speed_y

        # Bounce off walls
        if self.x <= BALL_RADIUS or self.x >= WIDTH - BALL_RADIUS:
            self.speed_x *= -1

    def get_rect(self):
        return pygame.Rect(
            self.x - BALL_RADIUS,
            self.y - BALL_RADIUS,
            BALL_RADIUS * 2,
            BALL_RADIUS * 2
        )

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), BALL_RADIUS)


class Game:
    def __init__(self, render=True):
        self.render_flag = render
        if self.render_flag:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("RL Catch Game - Q-Learning")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)

        self.paddle = Paddle()
        self.ball = Ball()
        self.score = 0
        self.done = False

    def reset(self):
        self.paddle = Paddle()
        self.ball = Ball()
        self.score = 0
        self.done = False
        return self.get_state()

    def get_state(self):
        """
        Discretize the state for Q-learning
        State: (paddle_position, ball_x_relative, ball_y, ball_direction_x)
        """
        # Paddle position (3 bins: left, center, right)
        paddle_pos = self.paddle.x / (WIDTH - self.paddle.width)
        if paddle_pos < 0.33:
            paddle_bin = 0
        elif paddle_pos < 0.66:
            paddle_bin = 1
        else:
            paddle_bin = 2

        # Ball x relative to paddle (3 bins: left, center, right)
        ball_rel_x = (self.ball.x - (self.paddle.x + self.paddle.width / 2)) / WIDTH
        if ball_rel_x < -0.2:
            ball_x_bin = 0
        elif ball_rel_x > 0.2:
            ball_x_bin = 2
        else:
            ball_x_bin = 1

        # Ball y position (3 bins: top, middle, bottom)
        ball_y_bin = int(self.ball.y / HEIGHT * 3)
        ball_y_bin = min(2, ball_y_bin)

        # Ball horizontal direction (2 bins: left, right)
        ball_dir_bin = 0 if self.ball.speed_x < 0 else 1

        return (paddle_bin, ball_x_bin, ball_y_bin, ball_dir_bin)

    def step(self, action):
        """
        Take an action and return next_state, reward, done
        Actions: 0 = stay, 1 = move left, 2 = move right
        """
        reward = 0

        # Move paddle
        self.paddle.move(action)

        # Update ball
        self.ball.update()

        # Check collision with paddle
        if self.ball.get_rect().colliderect(self.paddle.get_rect()):
            reward = 10
            self.score += 1
            self.ball.reset()
            self.ball.speed_y = abs(self.ball.speed_y)  # Ensure going down

        # Check if ball falls below paddle
        if self.ball.y > HEIGHT:
            reward = -10
            self.done = True

        # Small penalty for each step to encourage catching quickly
        reward -= 0.01

        next_state = self.get_state()

        # Render if enabled
        if self.render_flag:
            self._render()

        return next_state, reward, self.done

    def _render(self):
        self.screen.fill(BLACK)
        self.paddle.draw(self.screen)
        self.ball.draw(self.screen)

        # Draw score
        score_text = self.font.render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(score_text, (10, 10))

        # Draw epsilon
        eps_text = self.font.render(f"Epsilon: {epsilon:.3f}", True, GREEN)
        self.screen.blit(eps_text, (10, 50))

        pygame.display.flip()
        self.clock.tick(FPS)

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True


class QLearningAgent:
    def __init__(self):
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0])  # 3 actions
        self.actions = [0, 1, 2]  # stay, left, right

    def get_action(self, state, epsilon):
        """Epsilon-greedy action selection"""
        if random.random() < epsilon:
            return random.choice(self.actions)
        else:
            q_values = self.q_table[state]
            max_q = max(q_values)
            # Choose randomly among actions with max Q-value
            best_actions = [a for a in self.actions if q_values[a] == max_q]
            return random.choice(best_actions)

    def update(self, state, action, reward, next_state, done):
        """Update Q-value using Q-learning formula"""
        current_q = self.q_table[state][action]

        if done:
            max_next_q = 0
        else:
            max_next_q = max(self.q_table[next_state])

        # Q-learning update formula
        new_q = current_q + ALPHA * (reward + GAMMA * max_next_q - current_q)
        self.q_table[state][action] = new_q

    def save(self, filename="q_table.pkl"):
        with open(filename, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
        print(f"Q-table saved to {filename}")

    def load(self, filename="q_table.pkl"):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0], pickle.load(f))
            print(f"Q-table loaded from {filename}")
            return True
        return False


def train(episodes=LEARNING_EPISODES):
    """Train the Q-learning agent"""
    global epsilon

    agent = QLearningAgent()

    # Try to load existing Q-table
    if agent.load():
        print("Continuing training with existing Q-table")
    else:
        print("Starting fresh training")

    scores = []

    for episode in range(episodes):
        game = Game(render=(episode % 50 == 0))  # Render every 50th episode
        state = game.reset()
        total_reward = 0
        running = True

        while not game.done and running:
            action = agent.get_action(state, epsilon)
            next_state, reward, done = game.step(action)
            agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            if not game.render_flag:
                # Check for quit in non-rendering mode
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        break

        scores.append(game.score)

        # Decay epsilon
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

        if episode % 100 == 0:
            avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
            print(f"Episode {episode}, Epsilon: {epsilon:.3f}, Avg Score: {avg_score:.2f}")

        # Save progress every 500 episodes
        if episode % 500 == 0 and episode > 0:
            agent.save()

    agent.save()
    print(f"\nTraining complete! Average score: {np.mean(scores[-100:]):.2f}")
    return agent


def watch_trained_agent():
    """Watch the trained agent play"""
    agent = QLearningAgent()

    if not agent.load():
        print("No trained model found. Please train first.")
        return

    print("\nWatching trained agent... Close window to exit.")

    game = Game(render=True)
    state = game.reset()
    running = True

    while running:
        # Always choose best action (no exploration)
        action = agent.get_action(state, epsilon=0)
        next_state, reward, done = game.step(action)
        state = next_state

        if done:
            state = game.reset()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    pygame.quit()


if __name__ == "__main__":
    global epsilon
    epsilon = EPSILON

    print("=" * 50)
    print("Reinforcement Learning with Pygame - Q-Learning")
    print("=" * 50)
    print("\nThis example demonstrates:")
    print("- Q-Learning algorithm")
    print("- State discretization")
    print("- Epsilon-greedy exploration")
    print("- Reward shaping")
    print("\nTraining the agent...")
    print("(Rendering every 50th episode for speed)")
    print()

    try:
        # Train the agent
        agent = train()

        # Ask if user wants to watch
        print("\nWould you like to watch the trained agent? (y/n)")
        # For automated testing, we'll just watch
        watch_trained_agent()

    except Exception as e:
        print(f"Error: {e}")
        pygame.quit()
