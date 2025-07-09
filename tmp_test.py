import numpy as np
import scipy.sparse as sp
import line_profiler

from sbm.io import SBMFit

from sbm.noisy_fit import create_sbm_noise

EPS, DELTA, ALPHA = 1.0, 1e-6, 0.999
RNG = np.random.default_rng(0)

@line_profiler.profile
def _make_sbm(block_sizes, P):
    """
    Create an SBMFit with integer edge counts according to prob-matrix P.
    P must be square len(block_sizes) × len(block_sizes), symmetric.
    """
    k_vec = np.array(block_sizes, int)
    B = len(k_vec)
    data, rows, cols = [], [], []

    for r in range(B):
        for s in range(r, B):
            N = k_vec[r] * k_vec[s] if r != s else k_vec[r] * (k_vec[r] - 1) // 2
            m = int(round(P[r, s] * N))
            if m > 0:
                rows.append(r); cols.append(s); data.append(m)

    M = sp.csr_array((data, (rows, cols)), shape=(B, B))
    M = M + M.T - sp.diags(M.diagonal())

    return SBMFit(
        block_sizes=list(block_sizes),
        block_conn=M,
        directed_graph=False,
        neg_loglike=-1.0,
        metadata={},
    )


def test_sample_integrity():
    P = np.array([[0.8, 0.3],
                  [0.3, 0.05]])
    sbm   = _make_sbm([3, 4], P)
    print('check')
    noise = create_sbm_noise(sbm, EPS, DELTA, ALPHA,
                             noise_type="heterogeneous_gaussian")
    print('check 2')
    sbm_noisy = noise.sample_sbm_fit(RNG)
    print('check 3')

def test_big_blocks_memory():
    B = 100
    k = 3
    sizes = [k] * B
    P = np.full((B, B), 0.1)
    np.fill_diagonal(P, 0.2)
    sbm = _make_sbm(sizes, P)          # builds sparse counts

    nz = create_sbm_noise(sbm, 1.0, 1e-6, 0.999,
                          noise_type="heterogeneous_gaussian")

if __name__ == "__main__":
    test_sample_integrity()



