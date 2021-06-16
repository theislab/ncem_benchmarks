class HyperparameterContainer:

    def __init__(self):
        self.n_eval_nodes = {
            "1": 1,
            "5": 5,
            "10": 10,
            "100": 100,
            "200": 200
        }
        self.learning_rate = {
            "0": 0.5,
            "1": 0.05,
            "2": 0.005,
            "3": 0.0005,
            "4": 0.00005
        }
        self.depth = {
            "0": 0,
            "1": 1,
            "2": 2,
            "3": 3,
            "4": 5
        }
        self.dropout = {
            "1": 0.05,
            "2": 0.1,
            "3": 0.2,
            "N": None
        }
        self.l1_coef = {
            "1": 0.,
            "2": 1e-6,
            "3": 1e-3,
            "4": 1e0,
            "5": 1e2,
            "6": 1e-1,
            "7": 1e-2,
            "8": 1e1,
            "N": None
        }
        self.l2_coef = {
            "1": 0.,
            "2": 1e-6,
            "3": 1e-3,
            "4": 1e0,
            "5": 1e2,
            "6": 1e-1,
            "7": 1e-2,
            "8": 1e1,
            "N": None
        }
        self.conditional_depth = {
            "0": 0,
            "1": 1,
            "2": 2
        }
        self.conditional_dimension = {
            "0": 8,
            "1": 16,
            "2": 64,
            "3": 128,
            "4": 256,
            "5": 512,
            "6": 1024
        }
        self.conditional_dropout = {
            "1": 0.,
            "2": 0.2,
            "3": 0.5
        }
        self.conditional_l2_dict = {
            "1": 0.,
            "2": 1e-6,
            "3": 1e-3,
            "4": 1e0
        }
        # general
        self.batchsize = {
            "0": 2,
            "1": 4,
            "2": 16,
            "3": 32,
            "4": 64,
            "5": 128,
            "S": None
        }