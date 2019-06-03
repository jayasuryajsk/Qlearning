import numpy as np 
import gym
import time
import random
from IPython.display import clear_output 

env = gym.make('FrozenLake-v0')

action_space_size = env.action_space.n
state_space_size = env.observation_space.n
q_table = np.zeros([state_space_size,action_space_size])


max_no_episodes = 10000
max_no_steps = 100

learning_rate = 0.1
disocunt_rate = 0.99

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001

rewards_of_all_episodes = []

for episode in range(max_no_episodes):
	state = env.reset()
	done = False
	total_rewards_in_episode = 0

	for step in range(max_no_steps):
		exploration_rate_threshold = random.uniform(0,1)

		if exploration_rate_threshold > exploration_rate :
			action = np.argmax(q_table[state:,])
		else:
			action = env.action_space.sample()

		new_state,reward,done,info = env.step(action)
		

		q_table[state,action] = q_table[state,action] * (1 -learning_rate) + learning_rate*(reward + disocunt_rate* np.max(q_table[state:,]))
		state = new_state
		total_rewards_in_episode += reward

		if done == True:
			break


	#change the exploration rate
	exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)
	rewards_of_all_episodes.append(total_rewards_in_episode)



	


rewards_per_thosand_episodes = np.array_split(np.array(rewards_of_all_episodes),max_no_episodes/1000)
count = 1000
print(rewards_of_all_episodes)

print("********Average reward per thousand episodes********\n")
for r in rewards_per_thosand_episodes:
    print(count, ": ", str(sum(r/1000)))
    count += 1000


print(q_table)