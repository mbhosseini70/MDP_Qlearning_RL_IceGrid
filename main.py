import random
import matplotlib.pyplot as plt
import numpy as np

from ice import *

EPISODES = 100000
EPSILON = 0.1
GAMMA = 0.9
LEARNING_RATE = 0.1

def argmax(l):
    """ Return the index of the maximum element of a list
    """
    return max(enumerate(l), key=lambda x:x[1])[0]

def main():
    env = Ice()
    average_cumulative_reward = 0.0
    episode_rewards = []
    rewards_buffer = [] 
    

    # Q-table, 4x4 states, 4 actions per state
    qtable = [[0., 0., 0., 0.] for state in range(4*4)]

    # Loop over episodes
    for i in range(1,EPISODES):
        state = env.reset()
        terminate = False
        cumulative_reward = 0.0

        # Loop over time-steps
        while not terminate:
            # Epsilon-greedy action selection
            if random.random() < EPSILON:
                action = random.randrange(4)
            else:
                action = argmax(qtable[state])

            # Perform the action
            next_state, reward, terminate = env.step(action)

            # Update the Q-Table
            best_next_action = argmax(qtable[next_state])
            qtable[state][action] += LEARNING_RATE * (reward + GAMMA * qtable[next_state][best_next_action] - qtable[state][action])


            # Update statistics
            cumulative_reward += reward
            state = next_state


        # # Per-episode statistics
        average_cumulative_reward *= 0.95
        average_cumulative_reward += 0.05 * cumulative_reward

        if i % 1000 == 0 or i == 100000:
            print(f'Episode: {i}, Cumulative Reward: {cumulative_reward}, Average Cumulative Reward: {average_cumulative_reward}')



        # Append cumulative_reward to episode_rewards list here
        episode_rewards.append(cumulative_reward)
        rewards_buffer.append(cumulative_reward)

    # Print the Q-table every 1000 episodes
        if i % 1000 == 0 and i !=0 :
            print_q_table(qtable)


    # print_q_table(qtable)
    # # Print the value table
    # for y in range(4):
    #     for x in range(4):
    #         print('%03.3f ' % max(qtable[y*4 + x]), end='')

    #     print()


    plot_rewards(episode_rewards)





# -------- Plotting function--------
# def plot_rewards(episode_rewards):
#     plt.figure(figsize=(20, 8))
#     plt.plot(episode_rewards, label='Cumulative Reward per Episode')
#     plt.xlabel('Episode')
#     plt.ylabel('Cumulative Reward')
#     plt.title('Learning Progress over Episodes')
#     plt.legend()
#     plt.show()

def plot_rewards(episode_rewards):
    plt.figure(figsize=(20, 8))
    episodes = np.arange(len(episode_rewards))
    num_points = len(episode_rewards) // 100  # Calculate number of complete 100-episode batches
    
    rewards_array = np.array(episode_rewards)
    split_rewards = np.array_split(rewards_array, num_points) if len(episode_rewards) % 100 != 0 else np.split(rewards_array, num_points)

    # Calculate mean and standard error for each segment
    means = [np.mean(seg) for seg in split_rewards]
    stderrs = [np.std(seg) / np.sqrt(len(seg)) for seg in split_rewards]

    # Plotting
    plt.errorbar(episodes[::100][:num_points], means, yerr=stderrs, fmt='-o', label='Cumulative Reward per Episode', errorevery=1, capsize=5)
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('Learning Progress over Episodes')
    plt.ylim(bottom=0)
    plt.legend()
    plt.show()



def print_q_table(qtable):
    """
    Print the Q-table with a box around each state's maximum Q-value.
    Each cell in the grid will be visually separated.
    """
    
    print("Formatted Q-Table:")
    # Top border of the table
    print("+" + "---+" * 9)
    for y in range(4):
        row = "|"
        for x in range(4):
            # Fetch the maximum Q-value for the current state and format it
            max_q = max(qtable[y*4 + x])
            row += f' {max_q:.3f} |'
        print(row)
        print("+" + "---+" * 9)
    print()
    print()
if __name__ == '__main__':
    main()
