import torch
from torch import nn
from torch.utils.data import DataLoader

# Experiment configuration.
A, B = 0.0, 1.0           # knot range; eigenvalues are sampled inside it
TAU = 0.1                 # barrier weight of phi(X) = -tau logdet(X)
N_DIM = 8                 # M in S^n
LAM_MIN, LAM_MAX = 0.05, 1.0
K = 8                     # cells, hence K-1 interior knots / matrix hinges
NS_DEGREE = 2             # quintic member p_2 of the canonical family
NS_ITERS = 15             # burn-in grows like log(1/LAM_MIN) / log(lambda_d)
N_TRAIN, N_TEST = 2048, 512
BATCH_SIZE = 64
EPOCHS = 30
LR_W, LR_XI = 5e-3, 2e-2  # weights and knots live on different scales
ETA_INIT, C0_INIT = 0.0, 1.0  # c0 warm-started at the asymptotic slope g'(inf) = 1
W_INIT = 1e-2
MARGIN = 1e-3             # keeps knots strictly inside (a, b)
N_GRID = 1001
EPS = 1e-12
SEED = 0

# Coefficients a_j of p_d(x) = sum_j a_j x^(2j+1), Cor. low-degree.
NS_COEFFS = {1: (1.5, -0.5), 2: (1.875, -1.25, 0.375)}[NS_DEGREE]

torch.manual_seed(SEED)

def g(x):
    """Scalar prox of -tau log(u): the root of u^2 - x u - tau = 0. Smooth, not CPWL."""
    return 0.5 * (x + torch.sqrt(x * x + 4.0 * TAU))

def sym(X):
    """Asymmetry compounds through the polynomial iteration; project it out each step."""
    return 0.5 * (X + X.transpose(-2, -1))

def random_orth(n_samples, n, generator):
    Q, Rm = torch.linalg.qr(torch.randn(n_samples, n, n, generator=generator))
    d = torch.sign(torch.diagonal(Rm, dim1=-2, dim2=-1))
    d[d == 0] = 1.0  # fixes the QR sign ambiguity, making the frames Haar-distributed
    return Q * d.unsqueeze(-2)

def random_matrices(n_samples, seed):
    """Build M = Q diag(lam) Q^T in S_++ and its target Q diag(g(lam)) Q^T."""
    gen = torch.Generator().manual_seed(seed)
    Q = random_orth(n_samples, N_DIM, gen)
    lam = torch.rand(n_samples, N_DIM, generator=gen) * (LAM_MAX - LAM_MIN) + LAM_MIN
    M = (Q * lam.unsqueeze(-2)) @ Q.transpose(-2, -1)
    Y = (Q * g(lam).unsqueeze(-2)) @ Q.transpose(-2, -1)
    return sym(M), sym(Y)

# Generate training data.
M_train, Y_train = random_matrices(N_TRAIN, SEED)
training_data = torch.utils.data.TensorDataset(M_train, Y_train)

# Generate test data with a disjoint draw.
M_test, Y_test = random_matrices(N_TEST, SEED + 1)
test_data = torch.utils.data.TensorDataset(M_test, Y_test)

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)

for X, y in test_dataloader:
    print(f"Shape of X [N, n, n]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

I_n = torch.eye(N_DIM, device=device)

def iter_sym(X, a):
    """Odd polynomial sum_j a_j X^(2j+1): eq. odd-poly with X X^T = X^2 for X symmetric."""
    G = sym(X @ X)
    out = a[0] * X
    P = X
    for j in range(1, len(a)):
        P = sym(G @ P)  # X^(2j+1), one product per degree
        out = out + a[j] * P
    return out

def sgn_ns(M, a=NS_COEFFS, n_iters=NS_ITERS):
    """Sgn(M) by Newton-Schulz; Sgn(M/beta) = Sgn(M), so prescaling is free (Lem. prescaling)."""
    beta = M.flatten(-2).norm(dim=-1).clamp_min(EPS).reshape(*M.shape[:-2], 1, 1)
    X = M / beta  # ||.||_F >= ||.||_2 puts the whole spectrum inside [-1,1]
    for _ in range(n_iters):
        X = iter_sym(X, a)
    return X

def mhinge(M, mu, S):
    """Matrix hinge, eq. net-mrelu: acts as ReLU(lambda - mu) with no decomposition."""
    M_mu = mu * S - M
    # The inexact Sgn(M_mu) enters multiplied by M_mu, whose i-th eigenvalue is
    # |lambda_i - mu|, so an eigenvalue sitting on a knot cannot spoil the term.
    return 0.5 * (M_mu @ sgn_ns(M_mu).transpose(-2, -1) @ S - M_mu)

# Define model
class SymSplineNet(nn.Module):
    def __init__(self, K):
        super().__init__()
        # eta is kept: g(0) = sqrt(tau) != 0, so eq. net-normalization does not apply.
        self.eta = nn.Parameter(torch.full((1,), ETA_INIT))
        self.c0 = nn.Parameter(torch.full((1,), C0_INIT))
        # Small random jumps: exact zeros leave every hidden unit gradient-symmetric.
        self.w = nn.Parameter(W_INIT * torch.randn(K - 1))
        self.xi = nn.Parameter(torch.linspace(A, B, K + 1)[1:-1].clone())

    def forward(self, M):
        S = sgn_ns(M)                      # one Sgn(M), shared by every hidden unit
        out = self.eta * S + self.c0 * M   # eq. net-spline, the two affine terms
        for i in range(self.w.numel()):    # one activation per interior knot
            out = out + self.w[i] * mhinge(M, self.xi[i], S)
        return sym(out)

model = SymSplineNet(K).to(device)
print(model)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam([
    {"params": [model.eta, model.c0, model.w], "lr": LR_W},
    {"params": [model.xi], "lr": LR_XI},
])

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # A knot leaving [a, b] sees no eigenvalue and its gradient dies.
        with torch.no_grad():
            model.xi.clamp_(A + MARGIN, B - MARGIN)

        if batch % 10 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, rel_err, resid = 0, 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            num = (pred - y).flatten(-2).norm(dim=-1)
            den = y.flatten(-2).norm(dim=-1).clamp_min(EPS)
            rel_err += (num / den).sum().item()
            # Optimality of the prox: U^2 - M U - tau I = 0, no reference solution needed.
            opt = pred @ pred - X @ pred - TAU * I_n
            resid += opt.flatten(-2).norm(dim=-1).sum().item()
    test_loss /= num_batches
    rel_err /= size
    resid /= size
    print(f"Test Error: \n Rel. Frobenius: {(100*rel_err):>0.3f}%, "
          f"Prox residual: {resid:>8e}, Avg loss: {test_loss:>8e} \n")

for t in range(EPOCHS):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

torch.save(model.state_dict(), "logdet_model.pth")
print("Saved PyTorch Model State to logdet_model.pth")

model = SymSplineNet(K).to(device)
model.load_state_dict(torch.load("logdet_model.pth", weights_only=True))

# Read the trained parameters back as a linear spline and check the identities of Sec. net.
model.eval()
with torch.no_grad():
    M = M_test[:BATCH_SIZE].to(device)
    lam, Q = torch.linalg.eigh(M)  # reference only, never used in the forward pass

    def from_eig(vals):
        return (Q * vals.unsqueeze(-2)) @ Q.transpose(-2, -1)

    def rel_fro(P, Ref):
        num = (P - Ref).flatten(-2).norm(dim=-1)
        den = Ref.flatten(-2).norm(dim=-1).clamp_min(EPS)
        return (num / den).max().item()

    # Sorting is needed only to read the partition off: the hinge sum is order-free.
    xi, order = torch.sort(model.xi.detach())
    w = model.w.detach()[order]
    c0 = model.c0.detach().squeeze()
    eta = model.eta.detach().squeeze()
    c = c0 + torch.cat([torch.zeros(1, device=device), torch.cumsum(w, 0)])  # eq. telescoped-jumps

    def s_bar(x):
        return eta + c0 * x + torch.relu(x.unsqueeze(-1) - xi) @ w           # eq. relu-rep

    U_hat = model(M)
    S_ns = sgn_ns(M)

    # Sanity: on S_++ the exact matrix sign is the identity.
    e_sgn = (S_ns - I_n).flatten(-2).norm(dim=-1).max().item()

    # E1: the matrix hinge against ReLU on the eigenvalues, at a probe knot.
    mu = torch.tensor(0.5 * (A + B), device=device)
    e_hinge = rel_fro(mhinge(M, mu, S_ns), from_eig(torch.relu(lam - mu)))

    # E2: the whole network against eq. net-entries.
    e_net = rel_fro(U_hat, from_eig(s_bar(lam)))

    # E3: fit against prox_{tau phi} = (M + (M^2 + 4 tau I)^{1/2}) / 2.
    U_ref = from_eig(g(lam))
    e_fit = rel_fro(U_hat, U_ref)

    # E4: the scalar profile the matrix network has learned.
    grid = torch.linspace(A, B, N_GRID, device=device)
    e_prof = (s_bar(grid) - g(grid)).abs().max().item()

    # E5: equivariance under orthogonal conjugation, the property of an eigenvalue operator.
    gen = torch.Generator().manual_seed(SEED + 2)
    P = random_orth(1, N_DIM, gen).to(device)
    e_equi = rel_fro(model(sym(P @ M @ P.transpose(-2, -1))),
                     sym(P @ U_hat @ P.transpose(-2, -1)))

    # E6: optimality, symmetry and definiteness of the learned prox.
    e_opt = (U_hat @ U_hat - M @ U_hat - TAU * I_n).flatten(-2).norm(dim=-1).max().item()
    e_sym = (U_hat - U_hat.transpose(-2, -1)).flatten(-2).norm(dim=-1).max().item()
    lam_min = torch.linalg.eigvalsh(U_hat).min().item()

print("\nNewton-Schulz forward (no decomposition)")
print(f" degree 2d+1          : {2*NS_DEGREE+1}, {NS_ITERS} iterations, {K-1} hinges")
print(f" ||Sgn(M) - I||_F     : {e_sgn:>12.4e}   (exact value on S_++)")
print(f" E1 eq. net-mrelu     : {e_hinge:>12.4e}")
print(f" E2 eq. net-entries   : {e_net:>12.4e}")
print(f" E5 equivariance      : {e_equi:>12.4e}")
print("Learned prox of -tau logdet")
print(f" E3 rel. Frobenius    : {e_fit:>12.4e}")
print(f" E4 sup |s_bar - g|   : {e_prof:>12.4e}")
print(f" E6 ||U^2 - MU - tI|| : {e_opt:>12.4e}")
print(f"    asymmetry         : {e_sym:>12.4e}")
print(f"    min eig of U      : {lam_min:>12.6f}   (sqrt(tau) = {TAU**0.5:.6f} at lambda = 0)")
print(f" eta, c0              : {eta.item():.6f}, {c0.item():.6f}")
print(f" cell slopes in       : [{c.min().item():.6f}, {c.max().item():.6f}]")
print("\nLearned interior knots")
print(" " + "  ".join(f"{v:.4f}" for v in xi))