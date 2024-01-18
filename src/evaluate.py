import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
from functions import flatten
import datetime
import numpy as np
import datetime
from scipy import stats

def evaluate_enjoy(data_path_csv:str = 'benchmarks_test.csv', plot_type: str = 'bar', only_avgs:bool = False):
    """
    Evaluate and visualize the performance of different configurations based on the data in a CSV benachmark file.

    Parameters:
        - data_path_csv (str): The path to the CSV file containing the data. Default is 'benchmarks_test.csv'.
        - plot_type (str): The type of plot to generate. Options are 'bar' and 'line'. Default is 'bar'.
        - only_avgs (bool): Whether to only plot the average returns or also the individual returns. Default is False.
    Returns:
        - None
    """
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
        ax.set_xticklabels(X, rotation=45, fontsize=30)
        ax.tick_params(axis='y', labelsize=30)
        fig.tight_layout()
        fig.savefig(f'{timestamp}_avg_return_per_config.png')


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
    # Plot actor loss over time
    plt.figure(figsize=(15,10))
    X = np.array(list(range(1000, 1000001, int(1000000/len(training_data['actor_losses']))))[1:])
    plt.plot(X, training_data['actor_losses'], color = "blue",label='Actor Loss')
    plt.xlabel('Timesteps x 1e5', fontsize=30)
    plt.ylabel('Actor Loss', fontsize=30)
    # plt.title('Actor Loss over Time', fontsize=30)
    # Visualize a trend line
    x = training_data['Unnamed: 0']
    y = training_data['actor_losses']
    z = np.polyfit(X, y, 1)
    p = np.poly1d(z)
    plt.plot(X, p(X), "r--", label='Trend Line', linewidth=6)
    plt.xticks(range(0,1000000, 100000), fontsize=30)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.tick_params(axis='y', labelsize=30)

    # Add legend
    plt.legend(fontsize=30)
    plt.title('Actor Loss over Time', fontsize=30)
    plt.tight_layout()
    # Save plot
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    filename = f"{training_data_path_csv}_{timestamp}_actor_losses.png"
    plt.savefig(filename)

    # Plot critics loss over time
    plt.figure(figsize=(15,10))

    x1 = training_data['Unnamed: 0']
    y1 = training_data['critic1_losses']
    plt.plot(X, y1, label='Critic loss', color='red')
    plt.xticks(range(0,1000000, 100000), fontsize=30)
    plt.tick_params(axis='y', labelsize=30)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    x2 = training_data['Unnamed: 0']
    y2 = training_data['critic2_losses']
    plt.plot(X, y2, label='Critic loss', color='#1f77ba')
    plt.xticks(range(0,1000000, 100000), fontsize=30)
    plt.tick_params(axis='y', labelsize=30)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    # # Calculate and plot trend line for critics loss
    slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(X, y2)
    plt.plot(X, intercept2 + slope2*X, 'r--', label='Trend line 2', linewidth=6)
    plt.xticks(range(0,1000000, 100000), fontsize=30)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.tick_params(axis='y', labelsize=30)


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




if __name__ == "__main__":
    evaluate_enjoy(data_path_csv="")
    evaluate_training(training_data_path_csv="")