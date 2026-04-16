# Flows---Vibes

## Indexed implementation workflow: ML flow for lattice field theory

1. **Define the target lattice theory setup**
   - Choose dimensionality, lattice size, boundary conditions, and action (for example, scalar φ⁴, gauge, or effective model).
   - Fix the probability target `p(x) is proportional to exp(-S(x))`, observables, and acceptance-quality metrics.

2. **Generate baseline data and diagnostics**
   - Produce reference samples (HMC/MCMC or trusted simulators) on representative lattice sizes.
   - Compute baseline observables (energy/action density, correlators, topological indicators) for later comparison.

3. **Select the flow architecture**
   - Use an invertible normalizing flow with lattice-aware coupling layers.
   - Enforce symmetries where needed (translation/parity/gauge constraints depending on theory).

4. **Specify coupling transforms in each layer**
   - Split lattice variables into passive/active partitions.
   - Update active part with
     - `s_theta(.)`: scale network
     - `t_theta(.)`: shift network
   - Maintain tractable log-Jacobian accumulation for training/inference.

5. **⚑ Activation choice for s_theta and t_theta**
   - **Recommended default:** GELU or SiLU in hidden layers for smooth gradients.
   - Use linear output heads for both networks; constrain s_theta through a bounded map (for stability), e.g. `alpha * tanh(s_theta_hat)`.

6. **⚑ Parameterization choice for s_theta and t_theta**
   - Parameterize the affine coupling as
     - `x_active' = x_active * exp(s_theta(x_passive)) + t_theta(x_passive)`
   - Practical stability option:
     - `s_theta = alpha * tanh(s_theta_hat)`, with `alpha in [1,3]`
     - `t_theta` unconstrained linear head (optionally zero-centered initialization)

7. **Train the model**
   - Optimize reverse KL (or mixed objective) between flow-induced distribution and lattice target.
   - Monitor loss, log-Jacobian statistics, effective sample size, and mode coverage.

8. **Validate physics fidelity**
   - Compare learned-sample observables against baseline chains with uncertainty bands.
   - Run finite-volume checks and autocorrelation diagnostics.

9. **Deploy into the sampling loop**
   - Use flow proposals in independent or Metropolis-corrected generation.
   - Track acceptance rate, decorrelation speed, and wall-clock efficiency.

10. **Iterate and scale**
    - Tune depth/width, coupling schedule, and regularization.
    - Re-check stability of s_theta/t_theta settings when scaling lattice size.
