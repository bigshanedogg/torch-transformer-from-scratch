import numpy as np

def get_length_penalty(length, alpha=1.2, min_length=5):
    # multiply output to cumulative_prob
    output = ((min_length + 1) / (min_length + length)) ** alpha
    return output