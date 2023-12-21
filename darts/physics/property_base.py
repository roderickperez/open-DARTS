import numpy as np


class PropertyBase:
    output_props: {}

    def get_state(self, state):
        pass

    def evaluate(self, state):
        pass

    def evaluate_thermal(self, state):
        pass

    def evaluate_at_cond(self, state):
        pass

    def set_output_props(self, props: dict):
        """
        :param props: Dictionary of lambdas with output properties to be evaluated
        :type props: dict[str, lambda]
        """
        self.output_props = props
        return
