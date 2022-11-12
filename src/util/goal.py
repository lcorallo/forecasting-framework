class ForecastingGoal():
    _GOAL = None
    _VIEW = None
    _OFFSET = None

    def __init__(self, goal, view, offset):
        self._GOAL = goal
        self._VIEW = view
        self._OFFSET = offset

    def get(self):
        return self._GOAL

    def options(self):
        return self._VIEW, self._OFFSET

class OneStepForecastingGoal(ForecastingGoal):
    def __init__(self, offset=1):
        super().__init__("ONE_STEP", view=1, offset=offset)

    def get(self):
        return super().get()
    
    def options(self):
        return super().options()

class OneShotForecastingGoal(ForecastingGoal):
    def __init__(self, view=2, offset=1):
        super().__init__("ONE_SHOT", view=view, offset=offset)
    
    def get(self):
        return super().get()
    
    def options(self):
        return super().options()
