import numpy as np

def compute_similarity(P_old, P_new, y, delta=15/20):
    B, H, W, C = y.shape
    P_all = np.concatenate([P_old, P_new], axis=0)
    M_1 = y.dot(P_all).reshape(B*H*W, C)
    S = M_1.dot(P_old.T)
    S = np.exp(S) / np.sum(np.exp(S), axis=-1)[:, np.newaxis]
    S[S < delta] = 0
    return S

def weighted_distillation(O_0, P_old, P_new, y):
    V = P_old.shape[0]
    B, H, W, C = O_0.shape
    S = compute_similarity(P_old, P_new, y)
    O_0_tilde = O_0[:,:,:,:V].reshape(B*H*W,V) * S
    O_0_tilde = O_0_tilde.reshape(B, H, W, V)
    return O_0_tilde 

def extract_new_prototypes(X, y):
    C = y.shape[-1]
    Z = X.reshape(-1, C)
    batch_y = y.reshape(-1, C)

    proto_new = batch_y.T.dot(Z) # sum
    mean_proto = proto_new/np.sum(batch_y,axis=0)[:, np.newaxis] # mean

    mask_zero = proto_new == 0
    mean_proto[mask_zero] = 0 # remove nans
    P_1 = mean_proto / np.linalg.norm(mean_proto, axis=-1)[:,np.newaxis]
    P_1[np.isnan(np.sum(P_1, axis=-1))] = 0
    return P_1