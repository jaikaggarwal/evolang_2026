import numpy as np
import pandas as pd
import scipy
from tqdm import tqdm
from information_theory import mi



def compute_vanilla_ib_scores(data_loader, langs_of_interest, prior, pu_m):
    """
    Note that the prior must be a 1-dimensional vector with shape (n, ), 
    where n is the number of meanings.
    """

    scores = []
    encoders = data_loader.language_to_encoder

    for language in tqdm(langs_of_interest):
        encoder = encoders[language]
        informativity, complexity = information_plane(prior, pu_m, encoder)
        scores.append([language, complexity, informativity])
    
    scores = pd.DataFrame(scores, columns=['language', 'complexity', 'informativity'])
    scores = scores.set_index("language")
    scores['ground_truth'] = scores.index.isin(data_loader.attested_languages)

    return scores



def information_plane(p_x, p_y_x, p_z_x):
    """ Given p(x), p(y|x), and p(z|x), calculate I[Y:Z] and I[X:Z] """
    p_xz = p_x[:, None] * p_z_x # Joint p(x,y), shape X x Y
    p_xyz = p_x[:, None, None] * p_y_x[:, :, None] * p_z_x[:, None, :] # Joint p(x,y,z), shape X x Y x Z

    p_yz = p_xyz.sum(axis=0) # Joint p(y,z), shape Y x Z
    p_z = p_xz.sum(axis=-2, keepdims=True)

    informativity = mi(p_yz)
    complexity = scipy.special.xlogy(p_xz, p_xz).sum() - scipy.special.xlogy(p_x, p_x).sum() - scipy.special.xlogy(p_z, p_z).sum()
    
    return informativity, complexity




def ib_helper(p_x, p_y_x, beta, init, num_iter=10, ctol=6):
    """ Find encoder q(Z|X) to minimize J = I[X:Z] - beta * I[Y:Z].
    
    Input:
    p_x : Distribution on X, of shape X.
    p_y_x : Conditional distribution on Y given X, of shape X x Y.
    beta : A non-negative scalar value.

    Output: 
    Conditional distribution on Z given X, of shape X x Z.

    """

    # Randomly initialize the conditional distribution q(z|x)
    q_z_x = init #scipy.special.softmax(np.random.randn(X, Z), -1) # shape X x Z
    p_y_x = p_y_x[:, None, :] # shape X x 1 x Y
    p_x = p_x[:, None] # shape X x 1

    # Blahut-Arimoto iteration to find the minimizing q(z|x)
    for _ in range(num_iter):

        q_z_x = np.round(q_z_x, ctol)
        q_z_x = q_z_x / q_z_x.sum(axis=1, keepdims=True)
        q_xz = p_x * q_z_x # Joint distribution q(x,z), shape X x Z
        
        q_z = q_xz.sum(axis=0) # Marginal distribution q(z), shape 1 x Z
        non_zero_words = np.nonzero(q_z)[0]
        q_z_x = q_z_x[:, non_zero_words]
        q_z_x = q_z_x / q_z_x.sum(axis=1, keepdims=True)

        q_xz = p_x * q_z_x # Joint distribution q(x,z), shape X x Z
        q_z = q_xz.sum(axis=0, keepdims=True) # Marginal distribution q(z), shape 1 x Z

        
        # q_z_x = q_z_x[:, non_zero_words]
        # # Add Laplace Smoothing
        # q_z = q_z + 1e-16
        # q_z = q_z / q_z.sum()

        q_y_z = ((q_xz / q_z)[:, :, None] * p_y_x).sum(axis=0, keepdims=True) # Conditional decoder distribution q(y|z), shape 1 x Z x Y
        d = ( 
            scipy.special.xlogy(p_y_x, p_y_x)
            - scipy.special.xlogy(p_y_x, q_y_z) # negative KL divergence -D[p(y|x) || q(y|z)]
        ).sum(axis=-1) # expected distortion over Y; shape X x Z
        
        q_z_x = scipy.special.softmax((np.log(q_z) - beta*d), axis=-1) # Conditional encoder distribution q(z|x) = 1/Z q(z) e^{-beta*d}

        # word_probs = q_z_x.sum(axis=0)



    return q_z_x


def discrete_ib_helper(p_x, p_y_x, beta, init, num_iter=20, ctol=6):
    """ Find encoder q(Z|X) to minimize J = H(Z) - gamma * I[Y:Z].
    
    Input:
    p_x : Distribution on X, of shape X.
    p_y_x : Conditional distribution on Y given X, of shape X x Y.
    beta : A non-negative scalar value.

    Output: 
    Conditional distribution on Z given X, of shape X x Z.

    """
    # Randomly initialize the conditional distribution q(z|x)
    q_z_x = init #scipy.special.softmax(np.random.randn(X, Z), -1) # shape X x Z
    p_y_x = p_y_x[:, None, :] # shape X x 1 x Y
    p_x = p_x[:, None] # shape X x 1

    # Blahut-Arimoto iteration to find the minimizing q(z|x)
    for _ in range(num_iter):
        q_xz = p_x * q_z_x # Joint distribution q(x,z), shape X x Z
        q_z = q_xz.sum(axis=0, keepdims=True) # Marginal distribution q(z), shape 1 x Z

        # Add Laplace Smoothing
        q_z = q_z + 1e-16
        q_z = q_z / q_z.sum()


        q_y_z = ((q_xz / q_z)[:, :, None] * p_y_x).sum(axis=0, keepdims=True) # Conditional decoder distribution q(y|z), shape 1 x Z x Y
        
        

        d = ( 
            scipy.special.xlogy(p_y_x, p_y_x)
            - scipy.special.xlogy(p_y_x, q_y_z) # negative KL divergence -D[p(y|x) || q(y|z)]
        ).sum(axis=-1) # expected distortion over Y; shape X x Z
        
        # Make this discrete
        scores = (np.log(q_z) - beta*d)
        best_indices = np.argmax(scores, axis=1)
        q_z_x = np.zeros(q_z_x.shape)
        q_z_x[np.arange(len(q_z_x)), best_indices] = 1

        # # # Remove unused words
        # word_probs = q_z_x.sum(axis=0)
        # non_zero_words = np.nonzero(word_probs)[0]
        # q_z_x = q_z_x[:, non_zero_words]

    return q_z_x



def make_curve(num_words, betas, p_m, pu_m, discrete=False, num_iter=20, ctol=6):
    init = np.identity(num_words)

    qW_M = []
    informativity = []
    complexity = []

    for beta in tqdm(betas):

        if discrete:
            q_w_m  = discrete_ib_helper(p_m, pu_m, beta, init, num_iter, ctol)
        else:
            q_w_m  = ib_helper(p_m, pu_m, beta, init, num_iter, ctol)

        informativity_temp, complexity_temp = information_plane(p_m, pu_m, q_w_m)

        qW_M.append(q_w_m)
        informativity.append(informativity_temp)
        complexity.append(complexity_temp)
        init = q_w_m
        
    curve = pd.DataFrame(data = {'beta': betas,
                    'informativity' : informativity,
                    'complexity' : complexity,
                    'j' : complexity - betas*informativity})
    return curve, qW_M


