from sklearn.model_selection import KFold
import numpy as np

class KFoldCrossValidation():
    _RANDOM_SEED = 1
    _cross_validation_results = []

    def get_all_params(self):
        cross_validation_results = np.array(self._cross_validation_results)
        print(cross_validation_results)
        return cross_validation_results[:, 2:4]

    def get_best_params(self):
        cross_validation_results = np.array(self._cross_validation_results)
        min = cross_validation_results.T[0].min()
        index_min = np.where(cross_validation_results.T[0] == min)
        return cross_validation_results[index_min][0][0], cross_validation_results[index_min][0][1], cross_validation_results[index_min][0][2], cross_validation_results[index_min][0][3]

    def _generate_train_validation_sets(self, X_train, Y_train): 
        kfold = KFold(n_splits=self.kfolds, shuffle=True, random_state=self._RANDOM_SEED)
        new_X_train, new_X_validation, new_Y_train, new_Y_validation = [], [], [], []

        for train_index, test_index in kfold.split(X_train):
            new_X_train.append(X_train[train_index])
            new_Y_train.append(Y_train[train_index])
            new_X_validation.append(X_train[test_index])
            new_Y_validation.append(Y_train[test_index])
        
        return np.array(new_X_train, dtype=object), np.array(new_X_validation, dtype=object), np.array(new_Y_train, dtype=object), np.array(new_Y_validation, dtype=object)