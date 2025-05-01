import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random


class City(gym.Env):
    def __init__(self, size=5):
        super().__init__()
        self._size = size
        self.EMPTY = 0
        self.STREET = 1
        self.HOUSE = 2
        self._streets_build = 0
        self._houses_build = 0

        self._agent_location = np.array(
            [random.randint(0, self._size - 1), random.randint(0, self._size - 1)],
            dtype=np.int32
        )

        self.observation_space = gym.spaces.Box(
            low=0, high=max(self.BUILDINGS.values()), shape=(size, size), dtype=np.int32
        )

        self.action_space = gym.spaces.Discrete(2)
        self._city = np.full((self._size, self._size), self.EMPTY, dtype=np.int32)

    def _is_in_bounds(self, x, y):
        return 0 <= x < self._size and 0 <= y < self._size

    def _get_neighbor(self, x, y):
        valid_positions = []
        for i in range(self._size):
            for j in range(self._size):
                neighbors = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
                for ni, nj in neighbors:
                    if self._is_in_bounds(ni, nj) and (self._city[ni, nj] == self.STREET) and self._city[i, j] == self.EMPTY:
                        valid_positions.append((i, j))
                        break
        return random.choice(valid_positions) if valid_positions else None

    def _get_obs(self):
        directions = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1),           (0, 1),
                      (1, -1),  (1, 0),  (1, 1)]
        neighbors_state = []
        for dx, dy in directions:
            x, y = self._agent_location
            x += dx
            y += dy
            neighbors_state.append(self._city[x, y] if self._is_in_bounds(x, y) else -1)
        x, y = self._agent_location
        return np.array([x, y] + neighbors_state, dtype=np.float32)

    def _get_neighbors(self, x, y):
        directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]
        neighbors = [(x + dx, y + dy) for dx, dy in directions]
        valid_neighbors = [pos for pos in neighbors if self._is_in_bounds(*pos)]
        return valid_neighbors

    def step(self, action):
        if action not in [0, 1]:
            raise ValueError(f"Action must be 0 or 1, got {action}")
        x, y = self._agent_location
        if action == 1:
            self._city[x, y] = self.HOUSE
            self._houses_build += 1
        else:
            self._city[x, y] = self.STREET
            self._streets_build += 1

        neighbor = self._get_neighbor(x, y)
        if neighbor is not None:
            self._agent_location = neighbor
            done = False
        else:
            done = True

        base_reward = 0.09 if action == 1 else 0
        is_fully_filled = (self._houses_build + self._streets_build == self._size * self._size)
        reward = base_reward
        if done and not is_fully_filled:
            number_of_empty_cells = np.sum(self._city == self.EMPTY)
            reward -= 10 * number_of_empty_cells
        return self._get_obs(), reward, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._city.fill(self.EMPTY)
        self._streets_build = 0
        street_x = random.randint(0, self._size - 1)
        street_y = random.randint(0, self._size - 1)
        self._city[street_x, street_y] = self.STREET
        self._streets_build = 1
        neighbors = [(street_x - 1, street_y), (street_x + 1, street_y),
                     (street_x, street_y - 1), (street_x, street_y + 1)]
        valid_neighbors = [pos for pos in neighbors if self._is_in_bounds(*pos) and self._city[pos] == self.EMPTY]
        self._agent_location = random.choice(valid_neighbors) if valid_neighbors else (street_x, street_y)
        return self._get_obs(), {}

    def render(self):
        plt.clf()
        plt.imshow(self._city, cmap='viridis')
        plt.colorbar()
        plt.draw()
        plt.pause(0.001)
        print(self._city)

    def plot(self):
        plt.imshow(self._city, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.show()
