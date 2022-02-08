import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

matplotlib.rcParams.update({'font.size': 40,'lines.linewidth':4})

def plot_returns():

    path = "./simulation_results/raw_data/"
    envs = os.listdir(path)
    sim_data, labels= [], []

    '''Retrieve simulation results from subdirectories'''
    for i,env in enumerate(envs):
        sim_data.append([])
        labels.append([])
        path2 = path+env+"/"
        print(path2)
        agents = os.listdir(path2)
        for j,agent in enumerate(agents):
            path3 = path2+agent+"/"
            print(path3)
            sim_data[i].append([])
            labels[i].append(agent)
            random_seeds = os.listdir(path3)
            for rs in random_seeds:
                path4 = path3+rs+"/"
                print(path4)
                sim_data[i][j].append(pd.read_pickle(path4+"sim_data.pkl"))

    '''Compute mean over simulations with different random seeds and plot all agents in a single figure'''
    print("n environments")
    print(len(sim_data))
    for i,sim_data_env in enumerate(sim_data):
        fig,ax = plt.subplots(1,1,figsize=(20,10))
        for j,sim_data_agent in enumerate(sim_data_env):
            lbl = labels[i][j]
            sim_data_mean = pd.concat(sim_data_agent).groupby(level=0).mean()
            sd_rollling_mean = sim_data_mean.rolling(window=50).mean()
            t = np.arange(sim_data_mean.shape[0])
            ax.plot(t,sd_rollling_mean["True_team_returns"],label=lbl)
            #ax.set_ylim([-1.6,1.6])
            ax.set_xlim([0,sim_data_mean.shape[0]])
            ax.legend()
        plt.savefig('./simulation_results/figures/'+envs[i]+'.png',dpi=fig.dpi)
        #plt.show()

if __name__ == '__main__':
    plot_returns()
