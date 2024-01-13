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


#Set the path to the csv file to evaluate the benchmark results
data_path_csv = './benchmarks_td3_test_hp.csv'

# Set the path to the csv file to evaluate the training results
training_data_path_csv = './models/td3_gt_(config_0)/td3_results.csv'

def evaluate_enjoy(data_path_csv:str = 'benchmarks_test.csv', plot_type: str = 'bar', only_avgs:bool = False):

    # Read the data from the CSV file
    data = pd.read_csv(data_path_csv) #'benchmarks_test.csv'
    avg_return = data['avg_return']
    # Convert the string representation of lists into actual lists of floats
    data['returns'] = data['returns'].apply(ast.literal_eval)
    data['stddevs'] = data['stddevs'].apply(ast.literal_eval)
    data['episode_no'] = data['episode_no'].apply(ast.literal_eval)
        
    if isinstance(data['episode_no'], str):
        episode_no = ast.literal_eval(data['episode_no'])

    data['stddevs'] = data['stddevs'].apply(flatten)

    # Calculate the mean of the lists in 'stddevs' column
    data['stddevs'] = data['stddevs'].apply(np.mean)

    # Now group the data and calculate the mean standard deviation per episode over all experiments
    mean_stddev_per_episode = data.groupby(['time_stamp', 'config_name', 'user_name', 'agent_type'])['stddevs'].mean()

    # Update y1 and y2 values
    data['y1'] = data.apply(lambda row: np.mean(row['returns']) + mean_stddev_per_episode[(row['time_stamp'], row['config_name'], row['user_name'], row['agent_type'])], axis=1)
    data['y2'] = data.apply(lambda row: np.mean(row['returns']) - mean_stddev_per_episode[(row['time_stamp'], row['config_name'], row['user_name'], row['agent_type'])], axis=1)

    # Create a Linegraph with x-axis = episode, y-axis=returns per episode
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
            #label = f'{row["time_stamp"]} - {row["config_name"]} - {row["user_name"]} - {row["agent_type"]}'
            label = f' {row["agent_type"]}-{row["config_name"]}'
            label_means = f'{row["agent_type"]}-{row["config_name"]}-mean'
            
            if not only_avgs:
                ax.plot(episode_no, returns, label=label)
            ax.plot(episode_no, [row["avg_return"] for i in range(len(returns))], label=label_means)
            
            # Add a subplot using fill_between
            if not only_avgs:
                ax.fill_between(episode_no, np.array(returns) + np.array(stddevs), np.array(returns) - np.array(stddevs), alpha=0.3)
            ax.fill_between(episode_no, np.array([row["avg_return"] for i in range(len(returns))]) + np.array([row["avg_return_stddev"] for i in range(len(returns))]), np.array([row["avg_return"] for i in range(len(returns))]) - np.array([row["avg_return_stddev"] for i in range(len(returns))]), alpha=0.3)

        # Set the labels and title and legend
        ax.set_xlabel('Episode')
        ax.set_ylabel('Returns per Episode')
        ax.set_title('Returns per Episode for different configurations')
        ax.legend()
        
        # Show the plot
        #plt.show()
        
        # Save the plot with the timestamp in the file name
        fig.savefig(f'{timestamp}_returns_per_episode.png')

    if plot_type.lower() == 'bar':
        color=['green', 'blue', 'blue', 'gold', 'gold', 'darkred',  'darkred', 'darkgreen', 'darkorchid', 'darkorchid', 'cadetblue']
        X = data['config_name']
        Y = data['avg_return']
        t = type(Y)
        yerr = data['avg_return_stddev']
        fig, ax = plt.subplots()
        fig.set_figheight(10)
        fig.set_figwidth(25)
        ax.bar(X, Y, yerr=yerr, align='center', color=color[0:len(X)], ecolor='black', capsize=10)
        ax.set_title('Average Return per Configuration', fontsize=25)
        ax.set_ylabel('Average Return', fontsize=25)
        ax.set_xlabel('Configuration', fontsize=25)
        ax.set_xticklabels(X, rotation=45, fontsize=20)
        ax.tick_params(axis='y', labelsize=20)
        fig.tight_layout()
        fig.savefig(f'{timestamp}_avg_return_per_config.png')


def evaluate_training(training_data_path_csv = training_data_path_csv):
    # Load the training data from the CSV file
    training_data = pd.read_csv(training_data_path_csv)
    # Plot actor loss over time
    plt.figure()
    plt.plot(training_data['Unnamed: 0'], training_data['actor_losses'], color = "blue",label='Actor Loss')
    plt.xlabel('Steps')
    plt.ylabel('Actor Loss')
    plt.title('Actor Loss over Time')
    # Visualize a trend line
    x = training_data['Unnamed: 0']
    y = training_data['actor_losses']
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), "r--", label='Trend Line') 
    plt.xticks(range(0,10, 100000))

    # Add legend
    plt.legend()
    plt.title('Actor Loss over Time')
    # Save plot
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    filename = f"{training_data_path_csv}_{timestamp}_actor_losses.png"
    plt.savefig(filename)

    # Plot critics loss over time
    plt.figure()

    x1 = training_data['Unnamed: 0']
    y1 = training_data['critic1_losses']
    plt.plot(x1, y1, label='Critic1 loss', color='red')
    plt.xticks(range(0,10, 100000))
    plt.set_xlabel(range(0,100000, 100000))

    x2 = training_data['Unnamed: 0']
    y2 = training_data['critic2_losses']
    plt.plot(x2, y2, label='Critic2 loss')
    plt.xticks(range(0,10, 100000))

    # Calculate and plot trend line for critics loss
    slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(x2, y2)
    plt.plot(x2, intercept2 + slope2*x2, 'g--', label='Trend line 2')
    plt.xticks(range(0,10, 100000))


    slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(x1, y1)
    plt.plot(x1, intercept1 + slope1*x1, 'r--', label='Trend line 1')
    plt.xticks(range(0,10, 100000))
    
    # Set labels and title
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Critic Losses over Time')

    # Add a legend
    plt.legend()

    # Save the plot
    plt.savefig(f"{training_data_path_csv}_{timestamp}_critic_losses.png")




if __name__ == "__main__":
    evaluate_enjoy(data_path_csv=data_path_csv)
    #evaluate_training(training_data_path_csv=training_data_path_csv)

  

 


