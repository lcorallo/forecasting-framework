class ModelsIperParameters():
    FEATURE_LENGTH = "feature_length"
    C = "C"
    EPSILON = "epsilon"
    GAMMA = "gamma"
    FIT_INTERCEPT = "fit_intercept"
    LINEAR_LAYERS = "linear_layers"
    NEURONS = "neurons"
    DROPOUT = "dropout"
    RECURRENT_LAYERS = "recurrent_layers"
    HIDDEN_STATE = "hidden_state"

    __parameters = None
    def __init__(self, FEATURE_LENGTH=None, C=None, EPSILON=None, GAMMA=None, FIT_INTERCEPT=None, LINEAR_LAYERS=None, NEURONS = None, DROPOUT = None, RECURRENT_LAYERS = None, HIDDEN_STATE = None):
        temp_feature_length = [] if FEATURE_LENGTH is None else FEATURE_LENGTH
        temp_c = [] if C is None else C
        temp_epsilon = [] if EPSILON is None else EPSILON
        temp_gamma = [] if GAMMA is None else GAMMA
        temp_fit_intercept = [] if FIT_INTERCEPT is None else FIT_INTERCEPT
        temp_linear_layers = [] if LINEAR_LAYERS is None else LINEAR_LAYERS
        temp_neurons = [] if NEURONS is None else NEURONS
        temp_dropout = [] if DROPOUT is None else DROPOUT
        temp_recurrent_layers = [] if RECURRENT_LAYERS is None else RECURRENT_LAYERS
        temp_hidden_state = [] if HIDDEN_STATE is None else HIDDEN_STATE

        self.__parameters = {
            ModelsIperParameters.FEATURE_LENGTH: temp_feature_length,
            ModelsIperParameters.C: temp_c,
            ModelsIperParameters.EPSILON: temp_epsilon,
            ModelsIperParameters.GAMMA: temp_gamma,
            ModelsIperParameters.FIT_INTERCEPT: temp_fit_intercept,
            ModelsIperParameters.LINEAR_LAYERS: temp_linear_layers,
            ModelsIperParameters.NEURONS: temp_neurons,
            ModelsIperParameters.DROPOUT: temp_dropout,
            ModelsIperParameters.RECURRENT_LAYERS: temp_recurrent_layers,
            ModelsIperParameters.HIDDEN_STATE: temp_hidden_state
        }
    
    def get(self, key):
        return self.__parameters[key]

class NeuralNetworkTrainingParameters:
    def __init__(self, EPOCHS = 900, EARLY_STOP = True, EARLY_STOP_DIFF = 0.0005, LEARNING_RATE = .001):
        self.EPOCHS = EPOCHS
        self.EARLY_STOP = EARLY_STOP
        self.EARLY_STOP_DIFF = EARLY_STOP_DIFF
        self.LEARNING_RATE = LEARNING_RATE