
import numpy as np

def aggregate_fedavg(local_weights, sample_counts):
    # same as FedAvg aggregation
    nks = np.array(sample_counts, dtype=float)
    nks = np.maximum(nks, 1.0)
    wk = nks / nks.sum()
    new_weights = []
    for i in range(len(local_weights[0])):
        stacked = np.stack([lw[i] for lw in local_weights], axis=0)
        w_exp = wk.reshape((-1,) + (1,)*(stacked.ndim-1))
        new_weights.append(np.sum(w_exp * stacked, axis=0))
    return new_weights

def prox_update(model, global_weights, mu):
    # pullback towards global weights (L2 proximal)
    new_w = []
    for gw, lw in zip(global_weights, model.get_weights()):
        new_w.append(lw - mu*(lw - gw))
    model.set_weights(new_w)
