import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Experiment configuration.
A, B = 0.0, 4.0          # domain [a, b]
GAMMA = 1.0              # shrinkage level of the target
K = 5               # cells, hence K-1 interior knots / hidden units
N_TRAIN, N_TEST = 8192, 2048
NOISE_STD = 0.0
BATCH_SIZE = 64
EPOCHS = 60
LR_W, LR_XI = 5e-3, 5e-3  # weights and knots live on different scales
W_INIT = 1e-2
MARGIN = 1e-3            # keeps knots strictly inside (a, b)
N_GRID = 2001
SEED = 0

torch.manual_seed(SEED)

# #logdet prox
# def f(x):
#     return 0.5 * (x + torch.sqrt(x**2 + 4 * GAMMA))
#svt
def f(x):
    return torch.clamp(x - GAMMA, min=0.0)



# Generate training data by sampling the target on [a, b].
x_train = torch.rand(N_TRAIN, 1) * (B - A) + A
y_train = f(x_train) + NOISE_STD * torch.randn(N_TRAIN, 1)
training_data = torch.utils.data.TensorDataset(x_train, y_train)

# Generate test data the same way, with a disjoint draw.
x_test = torch.rand(N_TEST, 1) * (B - A) + A
y_test = f(x_test) + NOISE_STD * torch.randn(N_TEST, 1)
test_data = torch.utils.data.TensorDataset(x_test, y_test)

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)

for X, y in test_dataloader:
    print(f"Shape of X [N, 1]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# Dense grid for the sup error, held on the device.
grid = torch.linspace(A, B, N_GRID, device=device).unsqueeze(-1)
f_grid = f(grid)

# Define model
class SplineNet(nn.Module):
    def __init__(self, K):
        super().__init__()
        self.eta = nn.Parameter(torch.zeros(1))
        self.c0 = nn.Parameter(torch.zeros(1))
        # Small random jumps: exact zeros leave every hidden unit gradient-symmetric.
        self.w = nn.Parameter(W_INIT * torch.randn(K - 1))
        self.xi = nn.Parameter(torch.linspace(A, B, K + 1)[1:-1].clone())

    def forward(self, x):
        hinges = torch.relu(x - self.xi)          # (N, K-1), one hinge per interior knot
        return self.eta + self.c0 * x + hinges @ self.w.unsqueeze(-1)

model = SplineNet(K).to(device)
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

        # A knot leaving [a, b] on the right sees no data and its gradient dies.
        with torch.no_grad():
            model.xi.clamp_(A + MARGIN, B - MARGIN)

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
        sup_err = (model(grid) - f_grid).abs().max().item()
    test_loss /= num_batches
    print(f"Test Error: \n Sup error: {sup_err:>8e}, Avg loss: {test_loss:>8f} \n")

for t in range(EPOCHS):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

model = SplineNet(K).to(device)
model.load_state_dict(torch.load("model.pth", weights_only=True))

# Read the trained parameters back as a linear spline (Sec. pwl).
model.eval()
with torch.no_grad():
    # Sorting is needed only to read the partition off: the hinge sum is order-free.
    xi, order = torch.sort(model.xi.detach().cpu().double())
    w = model.w.detach().cpu().double()[order]
    c0 = model.c0.detach().cpu().double().squeeze()
    eta = model.eta.detach().cpu().double().squeeze()
    c = c0 + torch.cat([torch.zeros(1, dtype=torch.float64), torch.cumsum(w, 0)])  # eq. telescoped-jumps
    theta = 0.5 * (c[0] + c[-1])
    kappa = eta - 0.5 * (w * xi).sum()                                             # eq. rep-consts

    x = grid.cpu().double().squeeze(-1)
    s_relu = eta + c0 * x + torch.relu(x[:, None] - xi) @ w                        # eq. relu-rep
    s_abs = kappa + theta * x + 0.5 * (torch.abs(x[:, None] - xi) @ w)             # eq. abs-rep
    slope = c0 + (x[:, None] > xi).double() @ w                                    # Lem. ext-slopes

    # Uniform linear interpolant of f at the same K, as a baseline (Def. interp).
    knots = torch.linspace(A, B, K + 1, dtype=torch.float64)
    vals = f(knots)
    c_int = torch.diff(vals) / torch.diff(knots)
    w_int = torch.diff(c_int)
    eta_int = vals[0] - c_int[0] * knots[0]
    c0_int = c_int[0]
    s_int = eta_int + c0_int * x + torch.relu(x[:, None] - knots[1:-1]) @ w_int

    y_exact = f(x)
    h = (B - A) / K

print("\nFree-knot spline")
print(f" sup error            : {(s_relu - y_exact).abs().max().item():>12.6e}")
print("Uniform interpolant at equal K")
print(f" sup error            : {(s_int - y_exact).abs().max().item():>12.6e}")
# print("Identities")
# print(f" Lem. ext-forms  |relu - abs|     : {(s_relu - s_abs).abs().max().item():>12.3e}")
print("\nLearned interior knots")
print(" " + "  ".join(f"{v:.4f}" for v in xi))
print(f" eta: {eta.item():.4f}  c0: {c0.item():.4f}")
print(" w: " + "  ".join(f"{v:.4f}" for v in w))
print("Uniform interior knots")
print(" " + "  ".join(f"{v:.4f}" for v in knots[1:-1]))
print(f" eta: {eta_int.item():.4f}  c0: {c0_int.item():.4f}")
print(" w: " + "  ".join(f"{v:.4f}" for v in w_int))

# Plot: real target vs. free-knot spline vs. uniform interpolant.
x_np = x.numpy()
fig, ax = plt.subplots(figsize=(8, 5), facecolor="#fcfcfb")
ax.set_facecolor("#fcfcfb")
ax.plot(x_np, y_exact.numpy(), label="f (real)", color="#0b0b0b", linewidth=2, linestyle="--")
ax.plot(x_np, s_relu.numpy(), label="Free-knot spline", color="#2a78d6", linewidth=2)
ax.plot(x_np, s_int.numpy(), label="Uniform interpolant", color="#eb6834", linewidth=2)
ax.scatter(xi.numpy(), torch.zeros_like(xi).numpy(), color="#2a78d6", s=20, zorder=3, label="Learned knots")
ax.scatter(knots[1:-1].numpy(), torch.zeros_like(knots[1:-1]).numpy(), color="#eb6834", s=20, marker="x", zorder=3, label="Uniform knots")
ax.set_xlabel("x", color="#52514e")
ax.set_ylabel("y", color="#52514e")
ax.set_title("Spline fit comparison", color="#0b0b0b")
ax.grid(True, color="#e1e0d9", linewidth=0.8)
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color("#c3c2b7")
ax.tick_params(colors="#898781")
ax.legend(frameon=False, labelcolor="#0b0b0b")
fig.tight_layout()
fig.savefig("spline_comparison.png", dpi=150)
plt.show()