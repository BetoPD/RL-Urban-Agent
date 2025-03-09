import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

class City(gym.Env):
    def __init__(self, size = 5):
        super().__init__()
        # size of the grid
        self._size = size
        self.EMPTY = 0  
        self.STREET = 1
        self.HOUSE = 2  
        self._streets_build = 0

        # Gymnasium libarary

        # Agent Initial Location
        self._agent_location = np.array([size // 2, size // 2], dtype=np.int32)

        # Observations
        # self.observation_space = gym.spaces.Dict({
        #     "position": gym.spaces.Box(low=0, high=self._size-1, shape=(2,), dtype=np.int32),
        #     "neighbors": gym.spaces.Box(low=-1, high=self.HOUSE, shape=(8,), dtype=np.int32)
        # })
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=2.0,            # or any valid bound covering your data
            shape=(10,),
            dtype=np.float32
        )

        
        # Actions
        self.action_space = gym.spaces.Discrete(2)

        # city layout represented in a 2D-matrix
        self._city = np.full((self._size, self._size), self.EMPTY, dtype=np.int32)

    def _is_in_bounds(self, x, y):
        return 0 <= x < self._size and 0 <= y < self._size
    
    def _get_neighbor(self, x, y):
        # directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]
        # neighbors = [(x + dx, y + dy) for dx, dy in directions]
        # valid_neighbors = [pos for pos in neighbors if self._is_in_bounds(*pos) and self._city[pos] == self.EMPTY]
        # if valid_neighbors:
        #     return random.choice(valid_neighbors)
        # else:
        #     return None
        # get random neighbor where the grid is empty

        empty_positions = np.where(self._city == self.EMPTY)
        if empty_positions[0].size > 0:
            idx = np.random.choice(empty_positions[0].size)
            return (empty_positions[0][idx], empty_positions[1][idx])
        else:
            return None

    def _get_obs(self):
        directions = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1),           (0, 1),
                      (1, -1),  (1, 0),  (1, 1)]

        neighbors_state = []

        for dx, dy in directions:
            x, y = self._agent_location 
            x += dx
            y += dy

            if (self._is_in_bounds(x, y)):
                neighbors_state.append(self._city[x, y])
            else:
                neighbors_state.append(-1)
        
        x, y = self._agent_location

        # return {
        #     "position": self._agent_location,
        #     "neighbors": np.array(neighbors_state)
        # }
        return np.array([x, y] + neighbors_state, dtype=np.float32)
    
    def _evaluate_house(self, x, y):
        directions = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1),           (0, 1),
                      (1, -1),  (1, 0),  (1, 1)]
        
        neighbors = [(x, y)]

        for dx, dy in directions:
            temp_x, temp_y = x + dx, y + dy

            if self._is_in_bounds(temp_x, temp_y) and self._city[temp_x, temp_y] == self.HOUSE:
                neighbors.append((temp_x, temp_y))

        if len(neighbors) == 1:
            return 0
        
        has_street_access = False

        for neighbor in neighbors:
            for direction in directions:
                new_neighbor = (neighbor[0] + direction[0], neighbor[1] + direction[1])
                if not self._is_in_bounds(*new_neighbor):
                    continue

                if self._city[new_neighbor] == self.STREET:
                    has_street_access = True
                    break

        return 1 if has_street_access else 0

    def _get_neighbors(self, x, y):
        directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]
        neighbors = [(x + dx, y + dy) for dx, dy in directions]
        valid_neighbors = [pos for pos in neighbors if self._is_in_bounds(*pos)]
        return valid_neighbors
    
    def _check_street_connectivity(self) -> int:
        x, y = self._agent_location

        visited = set()

        stack = [(x, y)]

        while len(stack) > 0:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                neighbors = self._get_neighbors(*current)
                for neighbor in neighbors:
                    # only visit streets
                    if self._city[neighbor] != self.STREET:
                        continue
                    if neighbor not in visited:
                        stack.append(neighbor)
        
        return len(visited) / self._streets_build if self._streets_build > 0 else 0

    def step(self, action):
        
        if action not in [0, 1]:
            raise ValueError(f"Action can be either 0 or 1 not {action}")
        
        done = False
        reward = 0
        
        if action == 1:
            x, y = self._agent_location
            self._city[x, y] = self.HOUSE
            reward += self._evaluate_house(x, y)
        else:
            x, y = self._agent_location
            self._city[x, y] = self.STREET
            self._streets_build += 1
            reward += self._check_street_connectivity()

        neighbor = self._get_neighbor(x, y)

        if neighbor is not None:
            self._agent_location = neighbor
        else:
            done = True
        
        return self._get_obs(), reward, done, {}
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._city = np.full((self._size, self._size), self.EMPTY, dtype=np.int32)
        self._agent_location = np.array([self._size // 2, self._size // 2], dtype=np.int32)
        self._streets_build = 0
        return self._get_obs()
    
    def render(self):
        plt.clf()
        plt.imshow(self._city, cmap='viridis')
        # what color is wich building
        plt.colorbar()
        plt.draw()
        plt.pause(0.001)

        print(self._city)

        


