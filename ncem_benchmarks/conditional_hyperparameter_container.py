class ConditionalHyperparameterContainer:

    def __init__(self):
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