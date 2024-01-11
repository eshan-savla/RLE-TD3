import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast

from functions import flatten
import datetime
import numpy as np
import datetime
from scipy import stats


#Set the path to the csv file to evaluate the benchmark results
data_path_csv = './benchmarks_test.csv'

# Set the path to the csv file to evaluate the training results
training_data_path_csv = 'models/td3_best/td3_results.csv'

def evaluate_enjoy(data_path_csv:str = 'benchmarks_test.csv'):

    # Read the data from the CSV file
    data = pd.read_csv(data_path_csv) #'benchmarks_test.csv'

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
        label = f'{row["time_stamp"]} - {row["config_name"]} - {row["user_name"]} - {row["agent_type"]}'
        ax.plot(episode_no, returns, label=label)
        
        # Add a subplot using fill_between
        ax.fill_between(episode_no, np.array(returns) + np.array(stddevs), np.array(returns) - np.array(stddevs), alpha=0.3)

    # Set the labels and title and legend
    ax.set_xlabel('Episode')
    ax.set_ylabel('Returns per Episode')
    ax.set_title('Returns per Episode for different configurations')
    ax.legend()
    
    # Show the plot
    #plt.show()
    
    # Save the plot with the timestamp in the file name
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    fig.savefig(f'{timestamp}_returns_per_episode.png')




def evaluate_training(training_data_path_csv = training_data_path_csv):
    # Load the training data from the CSV file
    training_data = pd.read_csv(training_data_path_csv)
    # Plot actor loss over time
    plt.figure()
    plt.plot(training_data['Unnamed: 0'], training_data['actor_losses'], label='Actor Loss')
    plt.xlabel('Steps')
    plt.ylabel('Actor Loss')
    plt.title('Actor Loss over Time')
    # Visualize a trend line
    x = training_data['Unnamed: 0']
    y = training_data['actor_losses']
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), "r--", label='Trend Line')
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
    plt.plot(x1, y1, label='Critic1 Loss', color='red')

    x2 = training_data['Unnamed: 0']
    y2 = training_data['critic2_losses']
    plt.plot(x2, y2, label='Critic2 Loss', color='blue')

    # Calculate and plot trend line for critics loss
    slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(x2, y2)
    plt.plot(x2, intercept2 + slope2*x2, 'g--', label='Trend Line 2')

    slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(x1, y1)
    plt.plot(x1, intercept1 + slope1*x1, 'r--', label='Trend Line 1')
    
    # Set labels and title
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Critic Losses over Time')

    # Add a legend
    plt.legend()

    # Save the figure
    plt.savefig(f"{training_data_path_csv}_{timestamp}_critic_losses.png")




if __name__ == "__main__":
    evaluate_enjoy(data_path_csv=data_path_csv)
    evaluate_training(training_data_path_csv=training_data_path_csv)

  

 


