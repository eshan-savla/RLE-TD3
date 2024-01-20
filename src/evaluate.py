import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
from functions import flatten
import datetime
import numpy as np
import datetime
from scipy import stats
import os


#Set the path for the different evaluate functions
data_path_csv = './benchmarks_td3_test_hp.csv'              #path to the csv file to evaluate the benchmark results
training_data_path_csv = 'models/ddpg_gt/ddpg_results.csv'  #path to the csv file to evaluate the training results

def evaluate_enjoy(data_path_csv:str = data_path_csv, plot_type: str = 'bar', plot_avgs:bool = False, plot_timeseries:bool = False):
    """_summary_
    This function allows you to evaluate the enjoy phase of a trained model based on the benchmark results of the different agents.

    Args:
        data_path_csv (str, optional): _description_. Specify the path to the csv file
        plot_type (str, optional): _description_. Specify the plot_type for the graph. Default = 'bar'.
        only_avgs (bool, optional): _description_. Default = False.
    """

    # Read data from CSV file
    data = pd.read_csv(data_path_csv)
    
    # Convert the string representation of lists into actual lists of floats
    data['returns'] = data['returns'].apply(ast.literal_eval)
    data['stddevs'] = data['stddevs'].apply(ast.literal_eval)
    data['episode_no'] = data['episode_no'].apply(ast.literal_eval)
        
    if isinstance(data['episode_no'], str):
        episode_no = ast.literal_eval(data['episode_no'])

    data['stddevs'] = data['stddevs'].apply(flatten)

    # Calculate the mean of the lists in 'stddevs' column
    data['stddevs'] = data['stddevs'].apply(np.mean)

    # Group the data and calculate the mean standard deviation per episode over all experiments
    mean_stddev_per_episode = data.groupby(['time_stamp', 'config_name', 'user_name', 'agent_type'])['stddevs'].mean()

    # Update y1 and y2 values
    data['y1'] = data.apply(lambda row: np.mean(row['returns']) + mean_stddev_per_episode[(row['time_stamp'], row['config_name'], row['user_name'], row['agent_type'])], axis=1)
    data['y2'] = data.apply(lambda row: np.mean(row['returns']) - mean_stddev_per_episode[(row['time_stamp'], row['config_name'], row['user_name'], row['agent_type'])], axis=1)

    # Create a linegraph with x-axis = episode, y-axis=returns per episode
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    if plot_type.lower() == 'line':
        fig, ax = plt.subplots()
        for i, row in data.iterrows():
            episode_no = row['episode_no']
            returns = row['returns']
            stddevs = row['stddevs']
            
            # Convert the list into a string representation
            episode_no_str = str(episode_no)
            # Convert the string representation of episode numbers into actual list of floats
            episode_no = ast.literal_eval(episode_no_str)
            
            # Plot the returns per episode
            label = f' {row["agent_type"]}-{row["config_name"]}'
            label_means = f'{row["agent_type"]}-{row["config_name"]}-mean'
            
            if plot_timeseries:
                ax.plot(episode_no, returns, label=label)
            if plot_avgs:
                ax.plot(episode_no, [row["avg_return"] for i in range(len(returns))], label=label_means)
            
            # Add a subplot using fill_between
            if plot_timeseries:
                ax.fill_between(episode_no, np.array(returns) + np.array(stddevs), np.array(returns) - np.array(stddevs), alpha=0.3)
            if plot_avgs:
                ax.fill_between(episode_no, np.array([row["avg_return"] for i in range(len(returns))]) + np.array([row["avg_return_stddev"] for i in range(len(returns))]), np.array([row["avg_return"] for i in range(len(returns))]) - np.array([row["avg_return_stddev"] for i in range(len(returns))]), alpha=0.3)

        # Set the labels and title and legend
        ax.set_xlabel('Episode')
        ax.set_ylabel('Returns per Episode')
        ax.set_title('Returns per Episode for different configurations')
        ax.legend()
                
        # Save the plot with the unique timestamp in the file name
        fig.savefig(f'{timestamp}_returns_per_episode.png')

    if plot_type.lower() == 'bar':
        # Create a barchart with x-axis = configuration, y-axis=average return
        color=['green', 'blue', 'blue', 'gold', 'gold', 'darkred',  'darkred', 'darkgreen', 'darkorchid', 'darkorchid', 'cadetblue']
        X = data['config_name']
        Y = data['avg_return']
        t = type(Y)
        yerr = data['avg_return_stddev']
        
        fig, ax = plt.subplots() #create figure and axes
        #set the size of the figure
        fig.set_figheight(10)
        fig.set_figwidth(25)
        ax.bar(X, Y, yerr=yerr, align='center', color=color[0:len(X)], ecolor='black', capsize=10)
        #set the labels (including ticks and it's parameter) and title
        ax.set_title('Average Return per Configuration', fontsize=25)
        ax.set_ylabel('Average Return', fontsize=25)
        ax.set_xlabel('Configuration', fontsize=25)
        ax.set_xticklabels(X, rotation=45, fontsize=30)
        ax.tick_params(axis='y', labelsize=30)
        fig.tight_layout()
        fig.savefig(f'{timestamp}_avg_return_per_config.png') #save the figure


def evaluate_training(training_data_path_csv):
    """
    Evaluate the training by plotting the actor and critic losses over time.

    Parameters:
        - training_data_path_csv (str): The file path to the training data in CSV format.

    Returns:
        - None
    """
    # Load the training data from the CSV file
    training_data = pd.read_csv(training_data_path_csv)

    plt.figure(figsize=(15,10))     #specify figure size

    # Plot the actor loss over time
    X = np.array(list(range(1000, 1000001, int(1000000/len(training_data['actor_losses']))))[1:])
    plt.plot(X, training_data['actor_losses'], color = "blue",label='Actor Loss')
    
    # Set labels and title
    plt.xlabel('Timesteps x 1e5', fontsize=30)
    plt.ylabel('Actor Loss', fontsize=30)
    #plt.title('Actor Loss over Time', fontsize=30)
    
    # Visualize a trend line
    x = training_data['Unnamed: 0']
    y = training_data['actor_losses']
    z = np.polyfit(X, y, 1)
    p = np.poly1d(z)
    plt.plot(X, p(X), "r--", label='Trend Line', linewidth=6)
    plt.xticks(range(0,1000000, 100000), fontsize=30) #set ticks for x-axis labels
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0)) #format the labels in scientific notation
    plt.tick_params(axis='y', labelsize=30) #set ticks for y-axis labels and labelsize

    # Add legend
    plt.legend(fontsize=30)
    plt.title('Actor Loss over Time', fontsize=30)
    plt.tight_layout()

    # Save plot
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M") #create unigque timestamp
    filename = f"{training_data_path_csv}_{timestamp}_actor_losses.png"
    plt.savefig(filename)

    # Plot of critic loss
    plt.figure(figsize=(15,10)) #define figure size

    # Plot the critics loss of critic 1 over time
    x1 = training_data['Unnamed: 0']
    y1 = training_data['critic1_losses']
    plt.plot(X, y1, label='Critic loss', color='red')
    plt.xticks(range(0,1000000, 100000), fontsize=30) #set ticks for x-axis labels
    plt.tick_params(axis='y', labelsize=30) #set ticks for y-axis labels and labelsize
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0)) #format the labels in scientific notation

    # Plot the critics loss of critic 2 over time
    x2 = training_data['Unnamed: 0']
    y2 = training_data['critic2_losses']
    plt.plot(X, y2, label='Critic loss', color='#1f77ba')
    plt.xticks(range(0,1000000, 100000), fontsize=30)
    plt.tick_params(axis='y', labelsize=30)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    # Calculate and plot trend line for critics losses
    # Critic 2
    slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(X, y2)
    plt.plot(X, intercept2 + slope2*X, 'r--', label='Trend line 2', linewidth=6)
    plt.xticks(range(0,1000000, 100000), fontsize=30)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.tick_params(axis='y', labelsize=30)

    # Critic 1
    slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(X, y1)
    plt.plot(X, intercept1 + X*slope1, 'r--', label='Trend line 1', linewidth=6)
    plt.xticks(range(0,1000000, 100000), fontsize=30)
    plt.tick_params(axis='y', labelsize=30)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    # Set labels and title
    plt.xlabel('Timesteps x 1e5', fontsize=30)
    plt.ylabel('Loss', fontsize=30)
    plt.title('Critic Loss over Time', fontsize=30)

    # Add a legend
    plt.legend(fontsize=30)
    plt.tight_layout()

    # Save the plot
    plt.savefig(f"{training_data_path_csv}_{timestamp}_critic_losses.png")


#specify if you want to evaluate the enjoy phase or the training phase by toggling comments
if __name__ == "__main__":
    evaluate_enjoy(data_path_csv="")
    evaluate_training(training_data_path_csv="")