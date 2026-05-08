# A01: Score Intuition

## Clip Metadata

Stem: `A01-score_intuition`

Script: `2026-05-05_Diffusion-Score/scripts/A01-score_intuition.py`

Scene class: `ScoreIntuition`

Canonical outputs:

1. `2026-05-05_Diffusion-Score/video_outputs/A01-score_intuition_720p.mp4`
2. `2026-05-05_Diffusion-Score/video_outputs/A01-score_intuition_480p.mp4`

## Purpose

Introduce score as the gradient of log-density, not as a black-box diffusion formula. The viewer should first see `log p(x)` for a simple 2D Gaussian, then see that taking a derivative of this scalar landscape produces a vector field.

The clip should answer:

1. What does `log p(x)` look like for a 2D Gaussian?
2. What does it mean to take `∇ₓ log p(x)`?
3. What does the score look like at one point?
4. What does the full score field look like?
5. How does this field view connect to generation?

## Core Message

The score is:

```tex
s(x) = \nabla_x \log p(x)
```

For a Gaussian, `log p(x)` is highest at the mean and decreases as we move away. Its contour lines stay in the same 2D sample space as `x`, and the score arrow points perpendicular to those contours in the direction where log-density increases fastest.

For:

```tex
p(x) = N(x; \mu, \sigma^2 I)
```

the log-density is:

```tex
\log p(x) = C - \frac{\|x-\mu\|^2}{2\sigma^2}
```

and the score is:

```tex
s(x) = \nabla_x \log p(x) = -\frac{x-\mu}{\sigma^2}
```

## Storyboard

### Scene 1: Title and Equation

Purpose: introduce the definition without treating it as abstract notation.

Visuals:

1. Display the title: "Score = gradient of log-density."
2. Show the equation `s(x) = ∇ₓ log p(x)`.
3. Highlight `log p(x)` first.
4. Highlight `∇ₓ` second.
5. End with the full score expression highlighted.

Narration idea:

"Before diffusion models, start with a distribution. The score is what you get when you differentiate its log-density with respect to the sample position."

### Scene 2: Visualize `log p(x)` in 2D

Purpose: make the scalar field concrete before showing arrows.

Visuals:

1. Use a 2D coordinate plane.
2. Place a Gaussian center at `mu = (1, 1)`.
3. Draw contour rings for `log p(x)`.
4. Use warmer/brighter contour rings near the center to indicate higher log-density.
5. Add a label: "higher log p(x)" near the center and "lower log p(x)" farther away.

Design decision:

Use contour lines instead of a 3D surface. The score lives in the same 2D sample space as `x`; contours make the gradient direction visually clear without switching camera geometry.

### Scene 3: Take the Derivative at One Point

Purpose: show that score is the local direction of fastest increase.

Visuals:

1. Pick one point `x` away from the Gaussian mean.
2. Mark the point on a lower log-density contour.
3. Draw a dashed guide from `x` toward the mean.
4. Draw the score arrow at `x`, pointing inward and perpendicular to the nearby contour.
5. Reveal the Gaussian formulas:
   - `log p(x) = C - ||x - mu||^2 / (2 sigma^2)`
   - `s(x) = -(x - mu) / sigma^2`

Narration idea:

"At this point, the fastest way to increase log-density is to move inward. For a Gaussian, the derivative is exactly a vector back toward the mean."

### Scene 4: Full Gaussian Score Field

Purpose: scale the one-point derivative to the whole plane.

Visuals:

1. Fade in score arrows across a grid.
2. Keep the contour rings visible underneath.
3. Show that every arrow points toward the mean.
4. Animate the sample point moving a few steps along the field.

Key detail:

Normalize arrow lengths for readability, but preserve the direction exactly.

### Scene 5: Bridge to Generation

Purpose: connect score fields to generative dynamics without overloading the first clip.

Visuals:

1. Introduce a two-Gaussian data distribution.
2. Show the corresponding score field.
3. Start several noisy points from broad locations.
4. Animate them moving through the field toward higher-density regions.

Message:

"Once we can estimate the right field at each noise level, generation becomes the problem of moving samples through these fields."

Detailed VE, VP, and flow matching comparisons belong in later clips.

## Implementation Notes

Use a self-contained Manim script for this first clip.

Required helpers:

1. `gaussian_pdf(x, mean, sigma)`
2. `gaussian_log_pdf(x, mean, sigma)`
3. `gaussian_score(x, mean, sigma)`
4. `mixture_pdf(x, components)`
5. `mixture_score(x, components)`
6. `display_vector(vector, max_length)`
7. `field_step(point, components)`
8. Contour rendering helper for circular Gaussian log-density contours.

Equation rendering:

1. Do not use `MathTex` unless LaTeX is installed.
2. Current environment does not have `latex` or `dvisvgm`.
3. Render equations with Manim `Text` and Unicode math symbols, for example `s(x) = ∇ₓ log p(x)`, so the output video does not display broken LaTeX.

Design choices:

1. Use NumPy for math and Manim for rendering.
2. Keep everything in 2D for A01.
3. Keep the coordinate range fixed at roughly `[-3, 3] x [-3, 3]`.
4. Use contours for `log p(x)` and arrows for `s(x)`.
5. Use teal for the primary Gaussian, orange for the second mode, and yellow for emphasized sample paths.

## Render Commands

720p deliverable:

```bash
conda run -n py311 manim -qm --format mp4 --media_dir 2026-05-05_Diffusion-Score/video_outputs -o A01-score_intuition_720p 2026-05-05_Diffusion-Score/scripts/A01-score_intuition.py ScoreIntuition
```

480p preview:

```bash
conda run -n py311 manim -ql --format mp4 --media_dir 2026-05-05_Diffusion-Score/video_outputs -o A01-score_intuition_480p 2026-05-05_Diffusion-Score/scripts/A01-score_intuition.py ScoreIntuition
```

## Review Checklist

1. The score equation uses a real gradient symbol, not broken LaTeX or plain `grad_x`.
2. `log p(x)` is visible as 2D contours before score arrows appear.
3. The one-point score arrow is visibly perpendicular to the contour and points toward higher log-density.
4. The full score field is readable.
5. The bridge to generation is intuitive without explaining VE/VP/flow matching yet.
6. Render time is acceptable.
