import json
import matplotlib.pyplot as plt

file_path = f"./results/AttitudeSatellite/training_status.json"

with open(file_path, 'r') as file:
    data = json.load(file)

# Extract the checkpoints
checkpoints = data["AttitudeSatellite"]["checkpoints"]

# Extract the steps and reward values
steps = [checkpoint["steps"] for checkpoint in checkpoints]
rewards = [checkpoint["reward"] for checkpoint in checkpoints]

print(len(steps))

# Plot the reward
plt.plot(steps, rewards, linestyle='-')
plt.title('Reward')
plt.xlabel('Step')
plt.ylabel('Mean cumulative reward')
plt.savefig('result_train_reward.png')
#plt.grid(True)
plt.show()
