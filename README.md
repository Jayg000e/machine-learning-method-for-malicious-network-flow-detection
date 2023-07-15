# Classification of Encrypted Malicious Traffic



## View Results
If you only want to see the experimental results, please directly view the result folder

## Before performing the following operations, first enter the src folder

## Download Dataset
1.Download the required dataset from the address specified in the TechReport, then change the value of the data_dir variable in tree_bot.py and logistic.py to the corresponding address of the CTU-13 folder you downloaded.

2. Change the value of the data_dir variable in tree_brute_force.py, tree_DoS.py, tree_DDoS.py, and tree_brute_force.py to the corresponding addresses of the 02-14-2018.csv, 02-15-2018.csv, and 02-21-2018.csv files you downloaded.

## Environment
Install conda and run the following command   
    
    conda env create -f env.yaml

If you are prompted that there are missing packages during subsequent runs, use pip to install them

## Run
1. Please run the code for each model separately as follows. If you need to perform attack classification, set the attack_classification variable in the corresponding code to True. If you only need to perform identification, set it to False” in English. This means that you can run the code for each model separately and adjust the attack_classification variable depending on whether you want to perform attack classification or not.

2. In addition, setting GradientBoost=True in tree_bot.py allows you to use gradient boosting trees. At this time, it is recommended to set the attack_classification variable to True, otherwise it will take a long time to train, and the report does not recommend using gradient boosting trees for multi-classification
    
    python logistic.py

    python tree_bot.py

    python tree_brute_force.py

    python tree_DoS.py
    
    python tree_DDoS.py



