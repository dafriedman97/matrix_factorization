import numpy as np

class MatrixFactorization:
    """
    A class to run matrix factorization on a ratings matrix R for a specified number of factors F

    ...

    Attributes
    ----------
    F : int
        number of latent factors

    Methods
    _______
    fit(stochastic=True)
        user-called function to fit matrix factorization using stochastic or batch gradient descent
    _fit_stochastic(R, U, V, seed)
        fits the matrix factorization model with stochastic gradient descent and stores the U, V, and Rhat estimates
    _fit_batch(R, U, V, seed)
        fits the matrix factorization model with batch gradient descent and stores the U, V, and Rhat estimates        
    _estimate_ratings(U, V)
        forms the estimate of R, Rhat, and fills in 0 where corresponding entries in R are missing
    """
    def __init__(self, F):
        """
        Parameters
        __________
        F : int
            number of latent factors
        """
        self.F = F
        self.U = None
        self.V = None

    def fit(self, R, n_iter, lr, stochastic=True, seed=None):
        """
        fits the matrix factorization using stochastic or batch gradient descent and stores the fitted U and V matrices

        Parameters
        __________
        R : np.array
            ratings matrix of dimension (N, M) representing N users' ratings of M items (with missing values)        
        n_iter : int
            number of iterations of gradient descent
        lr : float
            learning rate
        stochastic : bool
            whether to fit the model with stochastic (versus batch) gradient descent
        seed : Optional[int]
            random seed
        """
        # Handle missing values
        self.R_missing = np.isnan(R) # a boolean matrix of the same shape as R, True where R's entries are missing and False elsewhere
        self.R = np.nan_to_num(R) # replace R's missing values with 0s

        # Instantiate U and V
        np.random.seed(seed) # set random seed
        self.N, self.M = self.R.shape # N = number of users, M = number of items
        U = np.random.randn(self.N, self.F) # randomly instantiate user-factor matrix (N x F)
        V = np.random.randn(self.M, self.F) # randomly instantiate item-factor matrix (M x F)

        # Fit
        self.n_iter, self.lr = n_iter, lr
        if stochastic:
            self._fit_stochastic(R=self.R, U=U, V=V, seed=seed) # estimate U, V with stochastic gradient descent
        else:
            self._fit_batch(R=self.R, U=U, V=V, seed=seed) # estimate U, V with batch gradient descent

    def _fit_stochastic(self, R, U, V, seed):
        """
        fits the matrix factorization model with stochastic gradient descent and stores the U, V, and Rhat estimates

        Parameters
        __________
        R : np.array
            ratings matrix 
        U : np.array
            user-factor matrix of dimension (N, F). nth row represents nth user's F latent factors
        V : np.array
            item-factor matrix of dimension (M, F). mth row represents mth item's F latent factors
        seed : Optional[int]
            random seed
        """
        Rhat = self._estimate_ratings(U=U, V=V) # Get Rhat, the estimate of R
        np.random.seed(seed) # set random seed
        for _ in range(self.n_iter): # until we've reached n_iter...
            n = np.random.choice(self.N) # randomly choose a user
            m = np.random.choice(self.M) # randomly choose an item
            un_grad = -(R[n,m] - Rhat[n,m])*V[m] # gradient with respect to u_n vector
            vm_grad = -(R[n,m] - Rhat[n,m])*U[n] # gradient with respect to v_m vector
            U[n] -= self.lr*un_grad # update 
            V[m] -= self.lr*vm_grad # update
            Rhat = self._estimate_ratings(U=U, V=V) # form new estimates
        self.Rhat, self.U, self.V = Rhat, U, V

    def _fit_batch(self, R, U, V, seed):
        """
        fits the matrix factorization model with batch gradient descent and stores the U, V, and Rhat estimates

        Parameters
        __________
        R : np.array
            ratings matrix 
        U : np.array
            user-factor matrix of dimension (N, F). nth row represents nth user's F latent factors
        V : np.array
            item-factor matrix of dimension (M, F). mth row represents mth item's F latent factors
        seed : Optional[int]
            random seed
        """
        Rhat = self._estimate_ratings(U=U, V=V) # Get Rhat, the estimate of R
        np.random.seed(seed) # set random seed
        for _ in range(self.n_iter): # until we've reached n_iter...
            E = Rhat - R # Get error matrix
            U_grad = np.matmul(E, V) # Get U gradient
            V_grad = np.matmul(E.transpose(), U) # Get V gradient
            U -= self.lr*U_grad # update
            V -= self.lr*V_grad # update
            Rhat = self._estimate_ratings(U=U, V=V) # form new estimates
        self.Rhat, self.U, self.V = Rhat, U, V

    def _estimate_ratings(self, U, V):
        """forms the estimate of R, Rhat, and fills in 0 where corresponding entries in R are missing"""
        Rhat = np.matmul(U, V.transpose()) # estimate R with UV^T
        Rhat = np.where(self.R_missing, 0, Rhat) # fill in missing values of R with 0s
        return Rhat        