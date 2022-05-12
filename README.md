#加密恶意流量分类

##查看结果
如果你只想看实验结果，请直接查看result文件夹

##数据集下载
按文章中所述的地址下载所需要的数据集，然后把tree_bot.py、logistic.py中data_dir变量的
赋值改为你下载CTU-13文件夹对应的地址；把tree_brute_force.py、tree_DoS.py、tree_DDoS.py、
tree_brute_force.py中data_dir变量的赋值改为所下02-14-2018.csv、02-15-2018.csv、02-21-2018.csv文件的
对应地址。

##环境配置
安装conda软件，并运行如下命令    
    
    conda env create -f env.yaml
如果后续运行中提示有缺包，利用pip安装

##运行代码
请按如下方式分别运行每个模型的代码，如果需要进行攻击分类，请把对应代码中
attack_classification变量设置为True，如果只需要进行鉴别则设置为False
另外在tree_bot.py中设置GradientBoost=True可以使用梯度提升树，此时建议将
attack_classification变量设置为True否则会训练很久，而且报告中并不建议进行利用梯度
提升树进行多分类。
    
    python logistic.py

    python tree_bot.py

    python tree_brute_force.py

    python tree_DoS.py
    
    python tree_DDoS.py



