import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random
class City(gym.Env):
    def __init__(self, size = 5):
        super().__init__()
        # size of the grid
        self._size = size
        self.EMPTY = 0  
        self.STREET = 1
        self.HOUSE = 2  
        self._streets_build = 0
        self._houses_build = 0

        # Gymnasium libarary

        # Agent Initial Location
        # self._agent_location = np.array([size // 2, size // 2], dtype=np.int32)
        # random agent location
        self._agent_location = np.array([random.randint(0, self._size - 1), random.randint(0, self._size - 1)], dtype=np.int32)

        # Observations
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0] + [-1]*8, dtype=np.float32),
            high=np.array([self._size - 1, self._size - 1] + [2]*8, dtype=np.float32),
            dtype=np.float32
        )
        # self.observation_space = gym.spaces.Box(
        #     low=np.array([-1]*8, dtype=np.float32),
        #     high=np.array([2]*8, dtype=np.float32),
        #     dtype=np.float32
        # )

        # Actions
        self.action_space = gym.spaces.Discrete(2)

        # city layout represented in a 2D-matrix
        self._city = np.full((self._size, self._size), self.EMPTY, dtype=np.int32)

    def _is_in_bounds(self, x, y):
        return 0 <= x < self._size and 0 <= y < self._size
    
    def _get_neighbor(self, x, y):
        valid_positions = []
        # Iteramos sobre todas las posiciones de la cuadrícula
        for i in range(self._size):
            for j in range(self._size):
                # Vecinos Von Neumann: arriba, abajo, izquierda y derecha
                neighbors = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
                # Verificamos si alguno de estos vecinos es una calle
                for ni, nj in neighbors:
                    if self._is_in_bounds(ni, nj) and (self._city[ni, nj] == self.STREET) and self._city[i, j] == self.EMPTY:
                        valid_positions.append((i, j))
                        break  # Se encontró al menos un vecino calle, no es necesario seguir revisando
        # Si se encontraron posiciones válidas, se elige una al azar
        if valid_positions:
            return random.choice(valid_positions)
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

        # return np.array(neighbors_state, dtype=np.float32)
        return np.array([x, y] + neighbors_state, dtype=np.float32)
    
    def step(self, action):
        # Validate the action
        if action not in [0, 1]:
            raise ValueError(f"Action must be 0 or 1, got {action}")

        # Get current agent location
        x, y = self._agent_location

        # Apply the action: build a house or street
        if action == 1:
            self._city[x, y] = self.HOUSE
            self._houses_build += 1
        else:
            self._city[x, y] = self.STREET
            self._streets_build += 1

        # Attempt to move to a new position
        neighbor = self._get_neighbor(x, y)
        if neighbor is not None:
            self._agent_location = neighbor
            done = False
        else:
            done = True

        # Compute the reward
        base_reward = 0.09 if action == 1 else 0
        is_fully_filled = (self._houses_build + self._streets_build == self._size * self._size)
        reward = base_reward
        if done and not is_fully_filled:
            number_of_empty_cells = np.sum(self._city == self.EMPTY)
            reward -= (10 * number_of_empty_cells)

        # Return the standard Gymnasium tuple: observation, reward, terminated, truncated, info
        return self._get_obs(), reward, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._city = np.full((self._size, self._size), self.EMPTY, dtype=np.int32)
        self._houses_build = 0
        self._streets_build = 0
        
        # Place an initial street
        street_x = random.randint(0, self._size - 1)
        street_y = random.randint(0, self._size - 1)
        self._city[street_x, street_y] = self.STREET
        self._streets_build = 1
        
        # Start agent at an adjacent empty cell
        neighbors = [(street_x - 1, street_y), (street_x + 1, street_y), 
                     (street_x, street_y - 1), (street_x, street_y + 1)]
        valid_neighbors = [pos for pos in neighbors if self._is_in_bounds(*pos) and self._city[pos] == self.EMPTY]
        self._agent_location = random.choice(valid_neighbors) if valid_neighbors else (street_x, street_y)
        
        return self._get_obs(), {}
    
    def render(self):
        plt.clf()
        plt.imshow(self._city, cmap='viridis')
        # what color is wich building
        plt.colorbar()
        plt.draw()
        plt.pause(0.01)

        print(self._city)

    def _street_connectiviy(self):
        number_of_streets = np.sum(self._city == self.STREET)
        return number_of_streets / self._streets_build

    def plot(self):
        plt.imshow(self._city, cmap='viridis', interpolation='nearest')
        plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Street'),
                            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=10, label='House'),
                            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=10, label='Empty')],
                    loc='upper left')
        
        plt.title("City Layout")
        # show connectivity
        connectivity = self._street_connectiviy()
        plt.text(0, -1, f"Street Connectivity: {connectivity:.2f}")
        # How many houses and streets were built
        plt.text(0, -2, f"Houses Built: {self._houses_build}")
        plt.text(0, -3, f"Streets Built: {self._streets_build}")

        plt.show()
    



