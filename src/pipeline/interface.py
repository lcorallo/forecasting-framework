from src.util.goal import ForecastingGoal


class IPipeline():
    def __execute__(self, series, goal: ForecastingGoal):
        raise ValueError("Method not implemented")
    
    def __test_execute__(self):
        raise ValueError("Method not implemented")