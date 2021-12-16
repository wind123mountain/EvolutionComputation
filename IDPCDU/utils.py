import numpy as np
import matplotlib.pyplot as plt
from GraphDomain import EncodePriority
import os

def Load_data(sc = 'NULL'):
    sc = "Datasets/IDPCDU_Edges/set1/"
    Tasks = os.listdir(sc)
    MAX_DIM = 0
    NUM_TASK = len(Tasks)

    for i in range(len(Tasks)):
        path = sc + Tasks[i] 
        name = Tasks[i].split('.')[0]
        Tasks[i] = EncodePriority(path, name=name)
        MAX_DIM = max(MAX_DIM, Tasks[i].NUM_DOMAIN)

    return Tasks, NUM_TASK, MAX_DIM