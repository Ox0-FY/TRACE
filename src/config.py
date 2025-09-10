# 数据路径
PATH_REAL_MAIN = '../data/Enzyme_regression.csv'
PATH_SIMULATED = '../data/Enzyme_regression Augmentation-30.csv'
PATH_REAL_TEST = '../data/HRP_regression.csv'

# 训练超参数
LR = 0.0005
N_ITERATIONS = 50000
K_SHOT, Q_QUERY = 10, 15
HIDDEN_SIZE = 128
TASK_EMBEDDING_DIM = 64

# 课程学习
STAGE1_ITERATIONS = 25000
STAGE1_POOL_RATIO = 0.1
