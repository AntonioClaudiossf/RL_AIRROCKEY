from mlagents.trainers.torch.encoders import VectorInput
from mlagents.trainers.torch.layers import LinearEncoder
from mlagents.trainers.torch.action_model import ActionModel
from mlagents_envs.base_env import ActionSpec
from unity_env_gym import make_unity_env
import torch
import numpy as np
import glob
import os
import re
import matplotlib.pyplot as plt
from matplotlib import rcParams
import argparse
from mlagents.torch_utils import default_device
import math
import subprocess

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--num_test_episode", type=int, default=1, help="number of episode of tests per checkpoints")
parser.add_argument("-f", "--freq_test_models", type=int, default=1, help="skip checkpoints every freq_test_models")
parser.add_argument("-s", "--satellite", type=int, default=1, help="Satellite 1, 2 or 3")
parser.add_argument("-ts", "--time_scale", type=int, default=1,help="Time scale of the simulation (1-20)")
args = parser.parse_args()

num_test_episode = args.num_test_episode
freq_test_models = args.freq_test_models
freq_chkpt = 0.05 #1e6 steps
linspace = freq_test_models * freq_chkpt
time_scale = args.time_scale
maximum_torque = 2.0/1000.0

#Function to test the agent
@torch.no_grad()
def test_agent_rl(nb_test_episodes):
    rewards_episode = []
    reward_history = []
    action_history = []
    obs_history = []
    euler_angles = []
    for episode in range(nb_test_episodes):
        obs = env.reset()
        done = False 
        while not done: 
            env.render()
            with torch.no_grad(): 
                norm_obs = vector_input(torch.FloatTensor(obs).to(default_device()))
                encoding = body(norm_obs)
                action,_,_ = action_head.forward(encoding,torch.full((AGENT_HIDDEN_DIM,), False, dtype=bool))
                action = (torch.clamp(action.continuous_tensor, -3, 3)/3).squeeze(0)
            next_obs, rew, done, _ = env.step(action.cpu())
            rewards_episode.append(rew)
            obs_history.append(obs)
            euler_angles.append(quat2euler(obs[0:4]))
            action_history.append(action.cpu().numpy()*maximum_torque)
            obs = next_obs 
			
        reward_history.append(np.sum(rewards_episode))
        rewards_episode = []
    return np.mean(reward_history), np.std(reward_history),np.vstack(obs_history),np.vstack(action_history),np.vstack(euler_angles)

#Function to sort the steps
def bubbleSort(arr):
    n = len(arr)
    swapped = False
    for i in range(n-1):
        for j in range(0, n-i-1):
            if arr[j] > arr[j + 1]:
                swapped = True
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
         
        if not swapped:
            return
    return arr

def PD(q_des, q_obs, Wc):
    Kp = 0.02
    Kd = 0.01
    Tcx= 0
    Tcy= 0
    Tcz= 0
    q_des = -1*np.array(q_des)
    prod_w = q_des[3] * q_obs[3] - q_des[0] * q_obs[0] - q_des[1] * q_obs[1] - q_des[2] * q_obs[2]
    prod_x = q_des[3] * q_obs[0] + q_des[0] * q_obs[3] + q_des[1] * q_obs[2] - q_des[2] * q_obs[1]
    prod_y = q_des[3] * q_obs[1] + q_des[1] * q_obs[3] + q_des[2] * q_obs[0] - q_des[0] * q_obs[2]
    prod_z = q_des[3] * q_obs[2] + q_des[2] * q_obs[3] + q_des[0] * q_obs[1] - q_des[1] * q_obs[0]

    q_error = [prod_x,prod_y,prod_z,prod_w]

    Tcx = -Kp * q_error[0]*q_error[3] - Kd * Wc[0]
    Tcy = -Kp * q_error[1]*q_error[3] - Kd * Wc[1]
    Tcz = -Kp * q_error[2]*q_error[3] - Kd * Wc[2]
    return np.clip(Tcx/maximum_torque,-1,1),np.clip(Tcy/maximum_torque,-1,1),np.clip(Tcz/maximum_torque,-1,1)


def quat2euler(q):
    x = q[0]
    y = q[1]
    z = q[2]
    w = q[3]
    ysqr = y * y

    t0 =  -2*(y*z - x*w)
    t1 =  w*w - x*x - y*y + z*z 
    roll = math.degrees(math.atan2(t0, t1))

    t2 = 2*(x*z + y*w)
    picht = math.degrees(math.asin(t2))

    t3 = -2*(x*y - z*w)
    t4 = w*w + x*x - y*y - z*z 
    yaw = math.degrees(math.atan2(t3, t4))

    return np.array([roll, picht, yaw])

#Neural Network config
action_size = ActionSpec.create_continuous(3) #Action = 3
nn_obs_dim = 15                               #Observation = 15
AGENT_HIDDEN_DIM = 225                        #Hidden unity = 255

#Test performance on the environment
print()
print("-*-*-*-*-*-*-*-*-*--*-*-*-*-*-*-*-*-*-*")
print("-*-*-*-*-* TEST PERFORMANCE -*-*-*-*-*-")
print("-*-*-*-*-*-*-*-*-*--*-*-*-*-*-*-*-*-*-*")
print()
ENV_NAME = f"./envs/CENARIO_{args.satellite}.x86_64"

print(f"========== TESTING ATTITUDE CONTROL RL SATELLITE {args.satellite} ==========")

RES_DIR_PATH = f"./results/AttitudeSatellite/AttitudeSatellite/"

# Retrieve every models from the checkpoint folder for tests
checkpoint_name = f"AttitudeSatellite-*.pt"
files_name = glob.glob(RES_DIR_PATH+checkpoint_name)

if files_name == []:
    print(f"/!\ /!\ {RES_DIR_PATH} not found /!\ /!\ ")

steps = bubbleSort([int(re.search('(?<=AttitudeSatellite-).*?(?=.pt)',step).group(0)) for step in files_name])

#===============================
mean_reward_episodes = []
std_reward_episodes  = []
erro_episodes = []
MSE = []

#===============================

for step in steps:

    #step = steps[-1]
    print(f"-- Model step {step} --")

    # ======= LOAD MODEL =======
    vector_input = VectorInput(input_size = nn_obs_dim, normalize = True)
    vector_input.load_state_dict(torch.load(RES_DIR_PATH+f"vector_input-{step}.pth", map_location=torch.device(default_device())),strict = True)

    body = LinearEncoder(
        input_size=nn_obs_dim,
        num_layers=3,
        hidden_size=AGENT_HIDDEN_DIM,
    )
    body.load_state_dict(torch.load(RES_DIR_PATH+f"body_endoder-{step}.pth", map_location=torch.device(default_device())),strict = True)
    action_head = ActionModel(AGENT_HIDDEN_DIM,action_size,tanh_squash=False,deterministic=True)
    action_head.load_state_dict(torch.load(RES_DIR_PATH+f"action_model-{step}.pth", map_location=torch.device(default_device())),strict = True)
    # ===========================

    # Load unity environment
    env = make_unity_env(ENV_NAME, worker_id = 30, no_graphics = False, time_scale = time_scale)
    #subprocess.run(['tput', 'cuu', '30'])
    #subprocess.run(['tput', 'ed'])

    # Test for num_test_episode
    step_perf, step_std,step_obs,step_euler_angles = test_agent_rl(num_test_episode)

    #Close the environment
    env.close()
    mean_reward_episodes.append(step_perf)
    std_reward_episodes.append(step_std)
    roll_mse = np.mean(np.square(np.zeros(400) - step_euler_angles[:,0]))
    pich_mse = np.mean(np.square(np.zeros(400) - step_euler_angles[:,1]))
    yaw__mse = np.mean(np.square(np.zeros(400) - step_euler_angles[:,2]))
    MSE.append([roll_mse,pich_mse,yaw__mse])

'''' for ate aqui'''

#Data to plot figure
obs_quaternion = step_obs[:,:4]
des_quaternion = step_obs[:,4:8]
err_quaternion = step_obs[:,8:12]
angular_vel = step_obs[:,12:]
#Torque = step_actions
x_perfs = np.arange(0,len(steps))

#Dir to save plots
path_images = f"./ressources/images/Cenario_4/"
path_pdfs = f"./ressources/pdfs/Cenario_4/"

rcParams['figure.figsize'] = 7, 4
plt.plot(steps,MSE)
plt.savefig(path_images+f"Mean_Reward_Satelite_{args.satellite}_RL.png", bbox_inches='tight', dpi=200)
plt.show()


"""
#Plot and save angular velocity
rcParams['figure.figsize'] = 7, 4
colors_wc = ['r','g','b']
wc_indice = ['x','y','z']
for i in range(0,3):
    plt.plot(x_perfs,angular_vel[:,i],color = colors_wc[i], label = f"$\omega_{wc_indice[i]}$")

plt.ylabel("Velocidade $(rad/s)$")
plt.xlabel("Tempo (s)")
plt.title("Velocidades angulares do satélite")
#plt.ylim([-0.45,0.45])
plt.legend(loc = "upper right")
plt.grid()
plt.savefig(path_images+f"Veloc_Ang_Cenario_{args.scenario}_{args.controller}.png", bbox_inches='tight', dpi=200)
plt.savefig(path_pdfs+f"Veloc_Ang_Cenario_{args.scenario}_{args.controller}.pdf", format='pdf', bbox_inches='tight')
plt.clf()

#Plot and save control torque
rcParams['figure.figsize'] = 7, 4
colors_T = ['r','g','b']
T_indice = ['x','y','z']
for i in range(0,3):
    plt.plot(x_perfs,Torque[:,i]*1000,color = colors_T[i], label = f"$T_{T_indice[i]}$")

plt.ylabel("Torque $(mNm)$")
plt.xlabel("Tempo (s)")
plt.title("Torque de controle")
#plt.ylim([-2.5/1000,2.5/1000])
plt.legend(loc = "upper right")
plt.grid()
plt.savefig(path_images+f"Torque_Cenario_{args.scenario}_{args.controller}.png", bbox_inches='tight', dpi=200)
plt.savefig(path_pdfs+f"Torque_Cenario_{args.scenario}_{args.controller}.pdf", format='pdf', bbox_inches='tight')

#Plot and ave Quaternions
rcParams['figure.figsize'] = 5, 7.5
colors_q = ['blue', 'orange', 'green', 'red']
q_indice = ['x', 'y', 'z', 'w']
fig, axs = plt.subplots(4, 1, layout='constrained')
for i in range(0,4):
	axs[i].plot(x_perfs, obs_quaternion[:,i], '--',color = colors_q[i],  label = f"$q_{q_indice[i]}$"+"$_{-obs}$")
	axs[i].plot(x_perfs, des_quaternion[:,i],      color = colors_q[i],  label = f"$q_{q_indice[i]}$"+"$_{-des}$")
	axs[i].set_xlabel('Tempo (s)')
	axs[i].set_ylabel('Quatérnion value')
	axs[i].legend()

fig.suptitle("Quatérnions", fontsize=18)
plt.savefig(path_images+f"Quaternion_Cenario_{args.scenario}_{args.controller}.png", bbox_inches='tight', dpi=200)
plt.savefig(path_pdfs+f"Quaternion_Cenario_{args.scenario}_{args.controller}.pdf", format='pdf', bbox_inches='tight')
plt.show()
"""
