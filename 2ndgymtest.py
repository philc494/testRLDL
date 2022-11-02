from collections import deque
import numpy as np
import gym
from icecream import ic
from math import e


def seq_translation(seq):
    win_list = []
    win_dic = {}
    for letter in seq:
        if letter == "A":
            win_list.append(0)
        elif letter == "B":
            win_list.append(4)
        elif letter == "C":
            win_list.append(20)
        elif letter == "D":
            win_list.append(24)
        elif letter == "M":
            win_list.append(12)
        else:
            return "Error"
    for index, target in enumerate(win_list):
        win_dic[index] = target
    return win_dic


def get_exp(n):
    return e ** n


class TwoDGridWorld(gym.Env):
    # metadata = {'render.modes': ['console']}

    up = 0
    left = 1
    down = 2
    right = 3

    def __init__(self, size):
        super(TwoDGridWorld, self).__init__()
        self.size = size  # size of the grid world
        self.end_state = train_pattern[game]
        self.agent_position = 12
        self.memory = deque([], 1000000)
        self.gamma = .90

        def relu(mat):
            return np.multiply(mat, (mat > 0))


        self.hidden_size = 12
        self.input_size = 2
        self.output_size = 4
        self.num_hidden_layers = 3
        self.epsilon = 1.0
        self.layers = [NNLayer(self.input_size + 1, self.hidden_size, activation=relu)]
        for i in range(self.num_hidden_layers - 1):
            self.layers.append(NNLayer(self.hidden_size + 1, self.hidden_size, activation=relu))
        self.layers.append(NNLayer(self.hidden_size + 1, self.output_size))

    def step(self, action):
        base_reward = 20
        exp_decay = -.25

        row = self.agent_position // self.size
        col = self.agent_position % self.size
        if action == self.up:
            if row != 0:
                self.agent_position -= self.size
            else:
                reward = 0
        elif action == self.left:
            if col != 0:
                self.agent_position -= 1
            else:
                reward = 0
        elif action == self.down:
            if row != self.size - 1:
                self.agent_position += self.size
            else:
                reward = 0
        elif action == self.right:
            if col != self.size - 1:
                self.agent_position += 1
            else:
                reward = 0
        else:
            raise ValueError("Received invalid action={} which is not part of the action space".format(action))

        done = bool(self.agent_position == self.end_state)

        # reward agent when it is in the terminal cell, else reward = 0
        reward = base_reward * (get_exp(exp_decay * timesteps)) if done else 0

        return np.array([[self.agent_position], [self.end_state]]).astype(np.uint8), reward, done

    def render(self, mode='console'):
        '''
            render the state
        '''
        if mode != 'console':
            raise NotImplementedError()

        row = self.agent_position // self.size
        col = self.agent_position % self.size

        for r in range(self.size):
            for c in range(self.size):
                if r == row and c == col:
                    print("X", end='')
                else:
                    print('.', end='')
            print('')

    def reset(self):
        # -1 to ensure agent inital position will not be at the end state
        self.agent_position = 12
        self.end_state = train_pattern[game]
        return np.array([self.agent_position, self.end_state]).astype(np.uint8)

    def close(self):
        pass

    def forward_agent(self, observation, remember_for_backprop=True):
        vals = np.copy(observation)
        index = 0
        for layer in self.layers:
            # print(layer.output_size)
            vals = layer.forward_NN(vals, remember_for_backprop)
            index = index + 1
        # print(vals)
        return vals

    def select_action(self, observation):
        values = self.forward_agent(observation)
        if (np.random.random() > self.epsilon):
            print(values)
            print('intentional action being chosen')
            return np.argmax(values)
        else:
            print(values)
            print('random action being chosen')
            return np.random.randint(0, 4)

    def remember(self, done, action, observation, prev_obs):
        self.memory.append([done, action, observation, prev_obs])

    def backward_agent(self, calculated_values, experimental_values):
        ic(calculated_values)
        ic(experimental_values)
        delta = (calculated_values - experimental_values)
        for layer in reversed(self.layers):
            delta = layer.backward_NN(delta)

    def experience_replay(self, update_size=20):
        if (len(self.memory) < update_size):
            return
        else:
            batch_indices = np.random.choice(len(self.memory), update_size)
            for index in batch_indices:
                done, action_selected, new_obs, prev_obs = self.memory[index]
                action_values = self.forward_agent(prev_obs, remember_for_backprop=True)
                next_action_values = self.forward_agent(new_obs, remember_for_backprop=False)
                experimental_values = np.copy(action_values)
                if done:
                    experimental_values[action_selected] = -1
                else:
                    experimental_values[action_selected] = 1 + self.gamma * np.max(next_action_values)
                self.backward_agent(action_values, experimental_values)
                self.epsilon = self.epsilon if self.epsilon < 0.01 else self.epsilon * 0.995
            for layer in self.layers:
                layer.lr = layer.lr if layer.lr < 0.0001 else layer.lr * 0.995


def relu_derivative(mat):
    return (mat > 0) * 1


class NNLayer:
    # class representing a neural net layer
    def __init__(self, input_size, output_size, activation=None, lr=0.001):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.uniform(low=-0.5, high=0.5, size=(input_size, output_size))
        self.activation_function = activation
        self.lr = lr

    def forward_NN(self, inputs, remember_for_backprop=True):
        input_with_bias = np.append(np.ones(1,), inputs)
        # print(f'shape of inputs is {np.shape(inputs)}')
        # print(f'shape of input with bias is {np.shape(input_with_bias)}')
        # print(f'shape of weights is {np.shape(self.weights)}')
        # print(input_with_bias)
        unactivated = np.dot(input_with_bias, self.weights)
        output = unactivated
        # print(f'shape of output is {np.shape(output)}')
        if self.activation_function is not None:
            output = self.activation_function(output)
        if remember_for_backprop:
            # store variables for backward pass
            self.backward_store_in = input_with_bias
            self.backward_store_out = np.copy(unactivated)
        return output

    def update_weights(self, gradient):
        self.weights = self.weights - self.lr * gradient

    def backward_NN(self, gradient_from_above):
        adjusted_mul = gradient_from_above
        if self.activation_function is not None:
            print('relu derivative function below:')
            ic(self.backward_store_out)
            ic(gradient_from_above)
            adjusted_mul = np.multiply(relu_derivative(self.backward_store_out), gradient_from_above)
        D_i = np.dot(np.transpose(np.reshape(self.backward_store_in, (1, len(self.backward_store_in)))),
             np.reshape(adjusted_mul, (1, len(adjusted_mul))))
        delta_i = np.dot(adjusted_mul, np.transpose(self.weights))[:-1]
        self.update_weights(D_i)
        return delta_i


action_trans = {0: 'Up', 1: 'Left', 2: 'Down', 3: 'Right'}

# Initialize training pattern
cust1 = 'ABACAD' * 8
test1 = 'AB' * 10

train_pattern = seq_translation(test1)

# Set global variables
game = 0
games = len(train_pattern)
max_timesteps = 20

# Make environment
env = TwoDGridWorld(5)
env.reset()

# Run main program loop
for episode in range(games):
    observation = env.reset()
    timesteps = 1
    print(f'Environment reset, game number: {game}, new target: {env.end_state}')
    for t in range(max_timesteps):
        env.render()
        action = env.select_action(observation)
        prev_obs = observation
        observation, reward, done = env.step(action)
        env.remember(done, action, observation, prev_obs)
        env.experience_replay(20)
        env.epsilon = env.epsilon if env.epsilon < 0.01 else env.epsilon * 0.995
        timesteps += 1
        print(f'Action selected for timestep {timesteps} in scenario {env.end_state}: {action_trans[action]}')
        print(f'reward being applied is: {reward}')
        if done:
            game += 1
            print('Game done - reached target')
            break
        if t == max_timesteps - 1:
            game += 1
            print('Ended without reaching target')
