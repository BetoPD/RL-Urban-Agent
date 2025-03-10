import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random

class City(gym.Env):
    def __init__(self, size=10):
        super().__init__()
        self._size = size
        self.EMPTY = 0  
        self.STREET = 1
        self.BUILDINGS = {"A": 2, "B": 3, "E": 4, "G": 5, "H": 6, "K": 7, "M": 8, "F": 9}

        self._streets_build = 0
        self._buildings_build = {key: 0 for key in self.BUILDINGS.keys()}

        # Compatibility matrix
        self.COMPATIBILITY_MATRIX = {
            "A": {"A": 5, "B": 3, "E": 4, "G": 4, "H": 1, "K": 4, "M": 4, "L": 5, "F": 2},
            "B": {"A": 3, "B": 5, "E": 3, "G": 3, "H": 4, "K": 4, "M": 1, "L": 5, "F": 1},
            "E": {"A": 4, "B": 3, "E": 5, "G": 2, "H": 2, "K": 1, "M": 1, "L": 5, "F": 3},
            "G": {"A": 4, "B": 3, "E": 2, "G": 5, "H": 3, "K": 4, "M": 1, "L": 5, "F": 5},
            "H": {"A": 1, "B": 4, "E": 2, "G": 3, "H": 5, "K": 4, "M": 3, "L": 5, "F": 2},
            "K": {"A": 4, "B": 4, "E": 1, "G": 4, "H": 4, "K": 5, "M": 1, "L": 5, "F": 2},
            "M": {"A": 4, "B": 1, "E": 1, "G": 1, "H": 3, "K": 1, "M": 5, "L": 5, "F": 1},
            "L": {"A": 5, "B": 5, "E": 5, "G": 5, "H": 5, "K": 5, "M": 5, "L": 5, "F": 5},
            "F": {"A": 2, "B": 1, "E": 3, "G": 5, "H": 2, "K": 2, "M": 1, "L": 5, "F": 5}
        }

        # Observation and action spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=max(self.BUILDINGS.values()), shape=(size, size), dtype=np.int32
        )
        self.action_space = gym.spaces.Discrete(9)  # 0 = road, 1-8 = buildings

        self._city = np.full((self._size, self._size), self.EMPTY, dtype=np.int32)
        self._agent_location = (random.randint(0, self._size - 1), random.randint(0, self._size - 1))

    def _distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _get_reward(self, x, y, building_type):
        reward = 0
        grid_size = self._size

        for (bx, by), value in np.ndenumerate(self._city):
            if value in self.BUILDINGS.values():
                building_label = [k for k, v in self.BUILDINGS.items() if v == value][0]
                distance = self._distance((x, y), (bx, by))
                reward += self.COMPATIBILITY_MATRIX[building_type][building_label] * (grid_size - distance)

        return reward

    def step(self, action):
        x, y = self._agent_location

        if action == 0:
            self._city[x, y] = self.STREET
            self._streets_build += 1
        else:
            building_label = list(self.BUILDINGS.keys())[action - 1]
            self._city[x, y] = self.BUILDINGS[building_label]
            self._buildings_build[building_label] += 1

        reward = self._get_reward(x, y, building_label if action != 0 else "L")

        neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        valid_neighbors = [pos for pos in neighbors if 0 <= pos[0] < self._size and 0 <= pos[1] < self._size]

        if valid_neighbors:
            self._agent_location = random.choice(valid_neighbors)
            done = False
        else:
            done = True

        return self._city.copy(), reward, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._city.fill(self.EMPTY)
        self._streets_build = 0
        self._buildings_build = {key: 0 for key in self.BUILDINGS.keys()}
        self._agent_location = (random.randint(0, self._size - 1), random.randint(0, self._size - 1))
        return self._city.copy(), {}

    def render(self):
        plt.imshow(self._city, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.show()
