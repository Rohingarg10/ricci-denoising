# Ricci Flow Image Denoising – My Attempt

## Paper Reference

**"Ricci Curvature and Flow for Image Denoising and Super-Resolution"**
Eli Appleboim, Emil Saucan, and Yehoshua Y. Zeevi
20th European Signal Processing Conference (EUSIPCO 2012)
Bucharest, Romania, August 27-31, 2012

## Background

I first tried playing with Ricci flow on the Fisher manifold (treating images as probability distributions with the Fisher info metric). That turned out way too slow and messy, so I switched to the discrete combinatorial approach described in this paper.

## Algorithm Logic

### Core Idea

Instead of evolving the image directly like in regular diffusion:

* **Normal methods**: ∂I/∂t = O(I)
* **Ricci Flow idea**: ∂G/∂t = -Ric(I), where G is the metric (weights), not the image itself


1. **Geometric Weights**

   * Edge weights: w(e) = √(β + |∇I|²)
   * Cell weights: w(c) = w(ex) × w(ey)

2. **Ricci Curvature**
   Used Forman’s combinatorial Ricci curvature:

   ```
   Ric(e₀) = w(e₀)[w(e₀)/w(c₁) + w(e₀)/w(c₂)] - 
             [√(w(e₀)w(e₁))/w(c₁) + √(w(e₀)w(e₂))/w(c₂) + ...]  
   ```

3. **Metric Evolution**
   Updated the gradient field with something like:

   ```python
   grad_new = grad - dt * ric * grad / (2 * (grad**2 + β))
   ```

4. **Reconstruction**
   Rebuilt the image with a Poisson solver.

5. **Iteration**
   Ran this for a few steps - I ried with both 3 and 7 iterations.


### What I Saw

Instead of edge-preserving denoising, everything turned into a blurred mush. Looked like I just hit the image with a strong Gaussian blur. Both β=1.0, dt=0.01, iterations=3 (like the paper suggests) or β=1.0, dt=0.001, iterations=7, $\alpha = 0.3$ (these are the parameters I used for the test results), it still blurred badly. Either I didn’t calculate Forman’s Ricci curvature correctly, or I applied it wrong in the evolution step. The fact that it just acted like isotropic diffusion suggests I wasn't able to capture the geometry well enough (basically the edges).

## Lessons Learned

Need more help with 
   * exact gradient/metric update rule
   * how to do Poisson reconstruction properly
   * what to do at image edges

Will implement a full debugging soon, as without checking intermediate curvature or metric updates one can't tell where it went wrong.

