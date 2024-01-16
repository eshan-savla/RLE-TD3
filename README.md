# RLE-TD3
Implemementation of the TD3 RL Network by RLE

## I. Configuration of the environment 

-----> TODO: Explain the basic setup for a foreign repo user.



## II. Test multiple configurations for the algorithm

The configuration management is done via hydra. 
You can find the different config files within the folder /configs.
The config.yaml file is the ground truth file of configurations. 


![Alt text](./documentation/images/Workflow_structure_overview.png)

Workflow:

1. Create your own config file within the /config folder. 
2. Switch to the td3_config.py file
3. Change the config_name to the name of your desired config file name (without the file extension)
4. Switch to the main_td3.py file 
5. Make your that your virtual environment is set-up properly (see I) 
6. Run the code.
    - The necessary data for an evaluation will be stored automatically in a Pandas DF within the folder /evals/result
    - The visualisation graphic for the returns over time will be stored automatically within the folder /evals/returns
    - The visualisation graphic for the losses over time will be stored automatically within the folder /evals/losses

