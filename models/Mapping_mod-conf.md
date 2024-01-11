# Mapping of models and configuration

**---> ToDo: Fill the Mapping with tested models and corresponding configs**



| Model_path | Config | Description | Comment       |
|---         |---     |---          |---          |
|'../models/ddpg_gt/2024-01-09_23-17/' | config.yaml | Ground truth | |
|'../models/td3_best/2024-01-08_18-27/'| config_higher_lr.yaml  | Best td3 model  | Parameters -> Same as ground truth except learning rate -> 0.01, policy noise std dev -> 0.5 and negative exponential reduction of noise over timesteps |
| '../models/td3_tau/2024-01-06_13-02/'  | config_1.yaml  | Change to ground truth: <br> - Tau = 0.005 -> 0.001  | Training continued at 499.000 steps due to hardware computing issues. <br> ReplayBuffer was collected newly. <br> => td3_results.csv to be checked regarding training steps coverage!|
| '../models/td3_tau/2024-01-06_20-46/'  | config_2.yaml  | Change to ground truth: <br> - Tau = 0.005 -> 0.01  | Training continued at 560.000 steps due to hardware computing issues. <br> ReplayBuffer was collected newly. <br> => td3_results.csv to be checked regarding training steps coverage! |
../models/td3_policy/2024-01-10_08-58 | config_5.yaml | Change to ground truth: policy noise std dev 0.2 -> 0.5 | |
| ../model/td3_policy/2024-01-10_18-42 | config_6.yaml | Change to ground truth: policy noise std dev 0.2 -> 0.1 | |
|