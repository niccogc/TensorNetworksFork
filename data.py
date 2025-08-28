import numpy as np

from sklearn.preprocessing import PolynomialFeatures
import math

class RandomPolynomial:
    """
    Random multivariate polynomial of total degree ≤ D using sklearn PolynomialFeatures.

    Coefficients are sampled with per-degree scaling:
        sigma_k = sigma0 / ((k+1) * sqrt(C(d+k-1, k))) * r^{-k}

    - d: number of variables
    - degree: total degree D
    - sigma0: base scale for coefficients
    - r: typical input magnitude (if |x_j| ~ r, higher-degree terms are tamed)
    - include_bias: include the constant term
    - interaction_only: if True, monomials never have a power >1 on any variable
    - random_state: seed for reproducibility
    """
    def __init__(
        self,
        d: int,
        degree: int,
        sigma0: float = 0.2,
        r: float = 1.0,
        mask: float = 0.1,
        include_bias: bool = True,
        interaction_only: bool = False,
        random_state = None,
    ):
        self.d = int(d)
        self.degree = int(degree)
        self.sigma0 = float(sigma0)
        self.r = float(r)
        self.mask = float(mask)
        self.include_bias = bool(include_bias)
        self.interaction_only = bool(interaction_only)

        # RNG
        if random_state is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(random_state)

        # Build PolynomialFeatures and "fit" on dummy data to populate .powers_
        self.poly = PolynomialFeatures(
            degree=self.degree,
            include_bias=self.include_bias,
            interaction_only=self.interaction_only
        )
        self.poly.fit(np.zeros((1, self.d)))  # just to get powers_

        # powers_: (n_features, d), each row is a multi-index alpha
        self._powers = self.poly.powers_  # ndarray of ints
        self._degrees = self._powers.sum(axis=1)  # (n_features,)

        # Precompute per-degree std
        self._deg_std = self._compute_degree_stds(self.d, self.degree, self.sigma0, self.r)

        # Sample coefficients
        self.coeffs_ = self._sample_coeffs()

    @staticmethod
    def _compute_degree_stds(d, D, sigma0, r):
        # sigma_k = sigma0 / ((k+1)*sqrt(C(d+k-1, k))) * r^{-k}
        deg_std = {}
        for k in range(D + 1):
            n_terms_k = math.comb(d + k - 1, k)  # number of monomials with total degree k
            sigma_k = sigma0 / ((k + 1) * math.sqrt(n_terms_k))
            if r != 0.0:
                sigma_k *= (r ** (-k))
            deg_std[k] = sigma_k
        return deg_std

    def _sample_coeffs(self):
        sigmas = np.array([self._deg_std[int(k)] for k in self._degrees], dtype=float)
        scale = self.rng.uniform(-10, 10, size=sigmas.shape)
        return np.exp(scale) * self.rng.normal(0, sigmas) * (1-self.rng.binomial(1, self.mask, size=sigmas.shape))

    def design_matrix(self, x: np.ndarray):
        """
        Return the monomial feature matrix Phi for x.

        x: (B, d) array
        Phi: (B, n_features) where n_features = C(D+d, d) (if include_bias=True; minus 1 otherwise)
        """
        x = np.asarray(x)
        if x.ndim != 2 or x.shape[1] != self.d:
            raise ValueError(f"x must be shape (B, {self.d})")
        return self.poly.transform(x)

    def evaluate(self, x: np.ndarray):
        """
        Evaluate the polynomial at x.

        x: (B, d)
        returns y: (B,)
        """
        Phi = self.design_matrix(x)         # (B, n_features)
        return Phi.dot(self.coeffs_)        # (B,)
    
class RandomPolynomialRange:
    def __init__(self, d, degree, input_range=(-1, 1), mask=0.0, random_state=None):
        self.d = d
        self.degree = degree
        self.range_start, self.range_end = input_range

        if random_state is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(random_state)

        self.C = self.rng.normal(
            loc=0.0,
            scale=1.0,
            size=(self.degree, self.d)
        )
        self.C = np.exp(self.C - np.max(self.C, axis=1, keepdims=True))
        self.C /= (np.sum(self.C, axis=1, keepdims=True) + 1e-12)
        self.roots = self.rng.uniform(
            low=self.range_start,
            high=self.range_end,
            size=(self.degree,)
        )

    def evaluate(self, x, add_noise=0.0):
        """
        Evaluate the polynomial at x.

        x: (B, d) array
        returns y: (B,) array
        """
        x = np.asarray(x)
        if x.ndim != 2 or x.shape[1] != self.d:
            raise ValueError(f"x must be shape (B, {self.d})")
        
        t = np.dot(x, self.C.T)  # (B, degree) weighted sum of inputs

        # Output is a polynomial of t with roots at self.roots
        # Calculate distance to each root
        dist = (t - self.roots[None, :])  # (B, degree)
        # Multiply by (t - root) for each root
        y = np.prod(dist, axis=1)  # (B,) product over roots
        
        return y + add_noise * self.rng.normal(size=y.shape)

class RandomIndependentPolynomial:
    def __init__(
        self,
        d: int,
        degree: int,
        coeff_sigma: float = 0.2,
        r: float = 1.0,
        mask: float = 0.1,
        include_bias: bool = True,
        interaction_only: bool = False,
        random_state = None,
    ):
        self.d = int(d)
        self.degree = int(degree)
        self.r = float(r)
        self.mask = float(mask)
        self.include_bias = bool(include_bias)
        self.interaction_only = bool(interaction_only)

        # RNG
        if random_state is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(random_state)

        # Build PolynomialFeatures and "fit" on dummy data to populate .powers_
        self.poly = PolynomialFeatures(
            degree=self.degree,
            include_bias=self.include_bias,
            interaction_only=self.interaction_only
        )
        self.poly.fit(np.zeros((1, self.d)))  # just to get powers_

        # Sample coefficients with full independence (comb(degree + dim, dim) total coefficients)
        self.coeffs_ = self.rng.normal(
            loc=0.0,
            scale=coeff_sigma,
            size=(self.poly.powers_.shape[0], 1)  # (n_features, 1)
        )

    def design_matrix(self, x: np.ndarray):
        """
        Return the monomial feature matrix Phi for x.

        x: (B, d) array
        Phi: (B, n_features) where n_features = C(D+d, d) (if include_bias=True; minus 1 otherwise)
        """
        x = np.asarray(x)
        if x.ndim != 2 or x.shape[1] != self.d:
            raise ValueError(f"x must be shape (B, {self.d})")
        return self.poly.transform(x)

    def evaluate(self, x: np.ndarray):
        """
        Evaluate the polynomial at x.

        x: (B, d)
        returns y: (B,)
        """
        Phi = self.design_matrix(x)         # (B, n_features)
        return Phi.dot(self.coeffs_)        # (B,)