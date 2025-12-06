"""
Retrospective Beam-Lookahead (RBL) Optimizer - Proof of Concept
================================================================

Novel optimizer combining:
1. Multi-path beam search (B branches with different strategies)
2. k-step lookahead per branch  
3. Discrete selection (pick best endpoint by loss)
4. Retrospective consistency check (modulate based on momentum alignment)

Foundational work to cite:
- Lookahead Optimizer (Zhang et al., 2019)
- Nesterov Accelerated Gradient / Nadam (Dozat, 2016)
- Evolutionary Strategies (Salimans et al., 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import math
from typing import List, Callable, Dict, Any
import matplotlib.pyplot as plt


class RBLOptimizer:
    """
    Retrospective Beam-Lookahead Optimizer
    
    Args:
        params: Model parameters to optimize
        base_optimizer_cls: Base optimizer class (e.g., Adam)
        base_optimizer_kwargs: Kwargs for base optimizer
        k: Lookahead horizon (steps to unroll)
        B: Beam width (number of parallel trajectories)
        alpha: Interpolation factor for final update
        strategies: List of strategy dicts to create diverse branches
    """
    
    def __init__(
        self,
        params,
        base_optimizer_cls=Adam,
        base_optimizer_kwargs: Dict[str, Any] = None,
        k: int = 5,
        B: int = 3,
        alpha: float = 0.5,
        strategies: List[Dict[str, Any]] = None,
    ):
        self.params = list(params)
        self.base_optimizer_cls = base_optimizer_cls
        self.base_optimizer_kwargs = base_optimizer_kwargs or {'lr': 0.001}
        self.k = k
        self.B = B
        self.alpha = alpha
        
        # default strategies: vary learning rate
        if strategies is None:
            base_lr = self.base_optimizer_kwargs.get('lr', 0.001)
            self.strategies = [
                {'lr': base_lr * 0.5},      # conservative
                {'lr': base_lr},             # standard
                {'lr': base_lr * 2.0},       # aggressive
            ][:B]
        else:
            self.strategies = strategies[:B]
        
        # ensure exactly B strategies
        while len(self.strategies) < B:
            self.strategies.append(self.base_optimizer_kwargs.copy())
        
        # make the "slow" optimizer for actual parameter updates
        self.slow_optimizer = base_optimizer_cls(self.params, **self.base_optimizer_kwargs)
        
        # track momentum history for retrospective check
        self.prev_update = None
        self.update_history = []
        
        # metrics
        self.branch_selections = {i: 0 for i in range(B)}
        self.consistency_scores = []
    
    def _clone_params(self) -> List[torch.Tensor]:
        """Create detached clones of current parameters."""
        return [p.data.clone() for p in self.params]
    
    def _set_params(self, param_values: List[torch.Tensor]):
        """Set parameters to given values."""
        for p, v in zip(self.params, param_values):
            p.data.copy_(v)
    
    def _get_param_vector(self) -> torch.Tensor:
        """Flatten all parameters into a single vector."""
        return torch.cat([p.data.view(-1) for p in self.params])
    
    def step(self, closure: Callable[[], torch.Tensor]):
        """
        Perform one RBL optimization step.
        
        Args:
            closure: A callable that computes the loss and performs backward().
                     Must be callable multiple times for branch evaluation.
        
        Returns:
            Tuple of (final_loss, selected_branch, consistency_score)
        """
        # store original parameters (slow weights)
        theta_t = self._clone_params()
        theta_t_vec = self._get_param_vector()
        
        # stage 1: Branching and Unrolling
        branch_endpoints = []  # θ_{t+k}^{(b)} for each branch
        branch_losses = []     # final loss for each branch
        
        for b, strategy in enumerate(self.strategies):
            # reset to starting position
            self._set_params(theta_t)
            
            # create optimizer for this branch with its strategy
            branch_kwargs = {**self.base_optimizer_kwargs, **strategy}
            branch_opt = self.base_optimizer_cls(self.params, **branch_kwargs)
            
            # unroll k steps
            for step_idx in range(self.k):
                branch_opt.zero_grad()
                loss = closure()
                loss.backward()
                branch_opt.step()
            
            # store endpoint
            branch_endpoints.append(self._clone_params())
            
            # eval final loss at endpoint (no grad needed)
            with torch.no_grad():
                final_loss = closure()
            branch_losses.append(final_loss.item())
        
        # stage 2: Discrete Selection (Beam Search)
        b_star = min(range(self.B), key=lambda i: branch_losses[i])
        self.branch_selections[b_star] += 1
        
        theta_star = branch_endpoints[b_star]
        theta_star_vec = torch.cat([p.view(-1) for p in theta_star])
        
        # comp update vector
        delta_theta = theta_star_vec - theta_t_vec
        
        # stage 3: Retrospective Consistency Check
        if self.prev_update is not None and self.prev_update.numel() == delta_theta.numel():
            # cosine similarity between proposed update and previous momentum
            cos_sim = F.cosine_similarity(
                delta_theta.unsqueeze(0),
                self.prev_update.unsqueeze(0)
            ).item()
            
            # Modulation function: λ = f(C)
            # If aligned (cos_sim > 0): boost slightly
            # If contradictory (cos_sim < 0): dampen
            if cos_sim > 0.5:
                lambda_mod = 1.0 + 0.5 * cos_sim  # up to 1.5x
            elif cos_sim > 0:
                lambda_mod = 1.0
            else:
                lambda_mod = max(0.5, 1.0 + 0.5 * cos_sim)  # down to 0.5x
        else:
            cos_sim = 0.0
            lambda_mod = 1.0
        
        self.consistency_scores.append(cos_sim)
        
        # stage 4: Final Slow Weight Update
        # θ_{t+1} = θ_t + α · λ · Δθ*
        final_update = self.alpha * lambda_mod * delta_theta
        
        # apply update
        self._set_params(theta_t)
        idx = 0
        for p in self.params:
            numel = p.numel()
            p.data.add_(final_update[idx:idx + numel].view_as(p))
            idx += numel
        
        # store update for next retrospective check
        self.prev_update = delta_theta.detach().clone()
        self.update_history.append(final_update.norm().item())
        
        # comp final loss at new position
        with torch.no_grad():
            final_loss = closure()
        
        return final_loss.item(), b_star, cos_sim
    
    def zero_grad(self):
        """Zero gradients of all parameters."""
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()


# =============================================================================
# TEST 1: Rosenbrock Function (Classic Optimization Benchmark)
# =============================================================================

def test_rosenbrock():
    """
    Test RBL on the Rosenbrock function: f(x,y) = (a-x)^2 + b(y-x^2)^2
    Global minimum at (a, a^2) = (1, 1) for a=1, b=100
    """
    print("=" * 60)
    print("TEST 1: Rosenbrock Function")
    print("=" * 60)
    
    a, b = 1.0, 100.0
    
    def run_optimizer(opt_name, optimizer_fn, steps=200):
        """Run an optimizer and track loss."""
        # start at known challenging point
        x = torch.tensor([-1.5], requires_grad=True)
        y = torch.tensor([2.0], requires_grad=True)
        
        losses = []
        positions = [(x.item(), y.item())]
        
        optimizer = optimizer_fn([x, y])
        
        for step in range(steps):
            def closure():
                loss = (a - x)**2 + b * (y - x**2)**2
                return loss
            
            if hasattr(optimizer, 'step') and 'closure' in str(type(optimizer).step.__code__.co_varnames):
                # RBL optimizer
                loss_val, _, _ = optimizer.step(closure)
            else:
                # stdandard optimizer
                optimizer.zero_grad()
                loss = closure()
                loss.backward()
                optimizer.step()
                loss_val = loss.item()
            
            losses.append(loss_val)
            positions.append((x.item(), y.item()))
        
        return losses, positions
    
    # compare optimizers
    results = {}
    
    # Adam baseline
    results['Adam'] = run_optimizer(
        'Adam',
        lambda params: Adam(params, lr=0.01)
    )
    
    # RBL optimizer
    results['RBL'] = run_optimizer(
        'RBL',
        lambda params: RBLOptimizer(
            params,
            base_optimizer_cls=Adam,
            base_optimizer_kwargs={'lr': 0.01},
            k=5,
            B=3,
            alpha=0.5
        )
    )
    
    # results
    for name, (losses, positions) in results.items():
        final_pos = positions[-1]
        print(f"\n{name}:")
        print(f"  Final loss: {losses[-1]:.6f}")
        print(f"  Final position: ({final_pos[0]:.4f}, {final_pos[1]:.4f})")
        print(f"  Distance to optimum: {math.sqrt((final_pos[0]-1)**2 + (final_pos[1]-1)**2):.4f}")
    
    return results


# =============================================================================
# TEST 2: Simple Neural Network (MNIST-like synthetic data)
# =============================================================================

class SimpleNet(nn.Module):
    """Simple 2-layer MLP for classification."""
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def generate_synthetic_data(n_samples=1000, input_dim=784, n_classes=10):
    """Generate synthetic classification data."""
    torch.manual_seed(42)
    X = torch.randn(n_samples, input_dim)
    y = torch.randint(0, n_classes, (n_samples,))
    return X, y


def test_neural_network():
    """Test RBL on a simple neural network."""
    print("\n" + "=" * 60)
    print("TEST 2: Neural Network Training")
    print("=" * 60)
    
    # gen data
    X_train, y_train = generate_synthetic_data(n_samples=500, input_dim=100, n_classes=10)
    batch_size = 50
    
    def run_training(opt_name, model, optimizer, epochs=20):
        """Train model and track metrics."""
        losses = []
        accuracies = []
        
        for epoch in range(epochs):
            epoch_losses = []
            
            # Mini-batch training
            for i in range(0, len(X_train), batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]
                
                def closure():
                    output = model(X_batch)
                    loss = F.cross_entropy(output, y_batch)
                    return loss
                
                if isinstance(optimizer, RBLOptimizer):
                    loss_val, branch, consistency = optimizer.step(closure)
                else:
                    optimizer.zero_grad()
                    loss = closure()
                    loss.backward()
                    optimizer.step()
                    loss_val = loss.item()
                
                epoch_losses.append(loss_val)
            
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            losses.append(avg_loss)
            
            # Compute accuracy
            with torch.no_grad():
                output = model(X_train)
                preds = output.argmax(dim=1)
                acc = (preds == y_train).float().mean().item()
                accuracies.append(acc)
            
            if epoch % 5 == 0 or epoch == epochs - 1:
                print(f"  Epoch {epoch+1:2d}: Loss={avg_loss:.4f}, Acc={acc:.2%}")
        
        return losses, accuracies
    
    results = {}
    
    # Adam baseline
    print("\nTraining with Adam:")
    torch.manual_seed(123)
    model_adam = SimpleNet(input_dim=100, hidden_dim=64, output_dim=10)
    opt_adam = Adam(model_adam.parameters(), lr=0.01)
    results['Adam'] = run_training('Adam', model_adam, opt_adam)
    
    # RBL optimizer
    print("\nTraining with RBL:")
    torch.manual_seed(123)
    model_rbl = SimpleNet(input_dim=100, hidden_dim=64, output_dim=10)
    opt_rbl = RBLOptimizer(
        model_rbl.parameters(),
        base_optimizer_cls=Adam,
        base_optimizer_kwargs={'lr': 0.01},
        k=3,  # short horizon for NN
        B=3,
        alpha=0.5
    )
    results['RBL'] = run_training('RBL', model_rbl, opt_rbl)
    
    # RBL diagnostics
    print(f"\nRBL Branch Selection Distribution: {opt_rbl.branch_selections}")
    avg_consistency = sum(opt_rbl.consistency_scores) / len(opt_rbl.consistency_scores) if opt_rbl.consistency_scores else 0
    print(f"Average Consistency Score: {avg_consistency:.4f}")
    
    return results


# =============================================================================
# TEST 3: Non-convex landscape with local minima
# =============================================================================

def test_rastrigin():
    """
    Test on Rastrigin function - highly non-convex with many local minima.
    f(x) = An + sum(x_i^2 - A*cos(2*pi*x_i))
    Global minimum at origin.
    """
    print("\n" + "=" * 60)
    print("TEST 3: Rastrigin Function (Many Local Minima)")
    print("=" * 60)
    
    A = 10.0
    
    def rastrigin(params):
        """Rastrigin function."""
        x = torch.stack(params)
        return A * len(params) + torch.sum(x**2 - A * torch.cos(2 * math.pi * x))
    
    def run_optimizer(opt_name, optimizer_fn, steps=300):
        # start at known challenging point
        torch.manual_seed(42)
        params = [torch.tensor([2.5], requires_grad=True),
                  torch.tensor([-2.5], requires_grad=True)]
        
        losses = []
        optimizer = optimizer_fn(params)
        
        for step in range(steps):
            def closure():
                return rastrigin(params)
            
            if isinstance(optimizer, RBLOptimizer):
                loss_val, _, _ = optimizer.step(closure)
            else:
                optimizer.zero_grad()
                loss = closure()
                loss.backward()
                optimizer.step()
                loss_val = loss.item()
            
            losses.append(loss_val)
        
        final_pos = [p.item() for p in params]
        return losses, final_pos
    
    results = {}
    
    # Adam
    results['Adam'] = run_optimizer(
        'Adam',
        lambda params: Adam(params, lr=0.05)
    )
    
    # RBL with more aggressive exploration
    results['RBL'] = run_optimizer(
        'RBL',
        lambda params: RBLOptimizer(
            params,
            base_optimizer_cls=Adam,
            base_optimizer_kwargs={'lr': 0.05},
            k=5,
            B=5,  # more branches for exploration, experiment
            alpha=0.6,
            strategies=[
                {'lr': 0.01},
                {'lr': 0.03},
                {'lr': 0.05},
                {'lr': 0.1},
                {'lr': 0.2},
            ]
        )
    )
    
    for name, (losses, final_pos) in results.items():
        print(f"\n{name}:")
        print(f"  Final loss: {losses[-1]:.4f}")
        print(f"  Final position: ({final_pos[0]:.4f}, {final_pos[1]:.4f})")
        print(f"  Distance to global optimum: {math.sqrt(sum(p**2 for p in final_pos)):.4f}")
    
    return results


# =============================================================================
# Visualization
# =============================================================================

def plot_results(rosenbrock_results, nn_results, rastrigin_results):
    """Create comparison plots."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    
    # Rosenbrock loss curves
    ax = axes[0, 0]
    for name, (losses, _) in rosenbrock_results.items():
        ax.plot(losses, label=name, linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Rosenbrock: Loss Curve')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Rosenbrock trajectory
    ax = axes[1, 0]
    for name, (_, positions) in rosenbrock_results.items():
        xs, ys = zip(*positions)
        ax.plot(xs, ys, 'o-', label=name, markersize=2, alpha=0.7)
    ax.plot(1, 1, 'r*', markersize=15, label='Optimum')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Rosenbrock: Optimization Path')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # NN loss curves
    ax = axes[0, 1]
    for name, (losses, _) in nn_results.items():
        ax.plot(losses, label=name, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Neural Network: Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # NN accuracy curves
    ax = axes[1, 1]
    for name, (_, accuracies) in nn_results.items():
        ax.plot(accuracies, label=name, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Neural Network: Training Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Rastrigin loss curves
    ax = axes[0, 2]
    for name, (losses, _) in rastrigin_results.items():
        ax.plot(losses, label=name, linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Rastrigin: Loss Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # summary table
    ax = axes[1, 2]
    ax.axis('off')
    
    adam_final = [
        rosenbrock_results['Adam'][0][-1],
        nn_results['Adam'][0][-1],
        rastrigin_results['Adam'][0][-1]
    ]
    rbl_final = [
        rosenbrock_results['RBL'][0][-1],
        nn_results['RBL'][0][-1],
        rastrigin_results['RBL'][0][-1]
    ]
    
    table_data = [
        ['Rosenbrock', f'{adam_final[0]:.4f}', f'{rbl_final[0]:.4f}', 'RBL' if rbl_final[0] < adam_final[0] else 'Adam'],
        ['Neural Net', f'{adam_final[1]:.4f}', f'{rbl_final[1]:.4f}', 'RBL' if rbl_final[1] < adam_final[1] else 'Adam'],
        ['Rastrigin', f'{adam_final[2]:.4f}', f'{rbl_final[2]:.4f}', 'RBL' if rbl_final[2] < adam_final[2] else 'Adam'],
    ]
    
    table = ax.table(
        cellText=table_data,
        colLabels=['Benchmark', 'Adam', 'RBL', 'Winner'],
        loc='center',
        cellLoc='center',
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    ax.set_title('Final Loss Comparison', pad=20)
    
    plt.tight_layout()
    #plt.savefig('rbl_optimizer_results.png', dpi=150)
    #plt.close()
    #print("\nPlot saved to: rbl_optimizer_results.png")
    plt.show()

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("RBL Optimizer - Proof of Concept")
    print("================================\n")
    
    # Run all tests
    rosenbrock_results = test_rosenbrock()
    nn_results = test_neural_network()
    rastrigin_results = test_rastrigin()
    
    # summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print("\nKey Observations:")
    print("-" * 40)
    
    # final losses
    for test_name, adam_res, rbl_res in [
        ("Rosenbrock", rosenbrock_results['Adam'], rosenbrock_results['RBL']),
        ("Neural Net", nn_results['Adam'], nn_results['RBL']),
        ("Rastrigin", rastrigin_results['Adam'], rastrigin_results['RBL']),
    ]:
        adam_final = adam_res[0][-1]
        rbl_final = rbl_res[0][-1]
        improvement = (adam_final - rbl_final) / adam_final * 100 if adam_final != 0 else 0
        winner = "RBL" if rbl_final < adam_final else "Adam"
        print(f"{test_name:12s}: Adam={adam_final:.4f}, RBL={rbl_final:.4f} -> {winner} wins ({abs(improvement):.1f}% {'better' if winner=='RBL' else 'worse'})")
    
    # plots
    try:
        plot_results(rosenbrock_results, nn_results, rastrigin_results)
    except Exception as e:
        print(f"\nNote: Could not generate plots: {e}")
    
    print("\n" + "=" * 60)
    print("Proof of concept complete!")
    print("=" * 60)