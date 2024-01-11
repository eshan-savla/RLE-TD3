import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast

from functions import flatten
import timeit


#Set the path to the csv file to evaluate the benchmark results
data_path_csv = './benchmarks_test.csv'


def evaluate(data_path_csv:str = 'benchmarks_test.csv'):

    # Read the data from the CSV file
    data = pd.read_csv(data_path_csv) #'benchmarks_test.csv'

    # Show the head of the DataFrame
    print(data.head())

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

    # Set the labels and title
    ax.set_xlabel('Episode')
    ax.set_ylabel('Returns per Episode')
    ax.set_title('Returns per Episode vs. Episode')

    # Show the legend
    ax.legend()

    # Show the plot
    plt.show()
    
    # Save the plot
    fig.savefig('returns_per_episode.png')

if __name__ == "__main__":
    elapsed_time = timeit.timeit(evaluate(data_path_csv=data_path_csv), number=1)
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"The main function ran for {int(minutes)} minutes and {seconds:.2f} seconds.")


