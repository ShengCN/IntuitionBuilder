# A01: Score Intuition

## Clip Metadata

Stem: `A01-score_intuition`

Script: `2026-05-05/scripts/A01-score_intuition.py`

Scene class: `ScoreIntuition`

Canonical outputs:

1. `2026-05-05/video_outputs/A01-score_intuition_720p.mp4`
2. `2026-05-05/video_outputs/A01-score_intuition_480p.mp4`

## Purpose

Introduce the score as the local direction where log-density increases fastest, then show how that field intuition becomes the bridge to diffusion-style generation.

The clip should answer:

1. What is the score?
2. What does the score look like for a single Gaussian?
3. What does the score look like for a two-Gaussian data distribution?
4. Why does this field view naturally lead to generative dynamics?

## Core Message

The score is:

```tex
s_t(x) = \nabla_x \log p_t(x)
```

It is not the probability itself. It is the local direction in x-space where log-density increases fastest.

Generation can be visualized as moving noisy samples through fields that point them toward higher-density structure.

## Storyboard

### Scene 1: Title and Thesis

Purpose: establish the core intuition.

Visuals:

1. Display the title: "The score points toward higher probability."
2. Show the equation:

```tex
s_t(x) = \nabla_x \log p_t(x)
```

3. Highlight `\log p_t(x)` first, then `\nabla_x`, then the whole score.
4. Add the explanatory text: "local direction where log-density increases fastest."

Narration idea:

"The score is not the probability itself. It is the local direction in x-space where log-density increases fastest."

### Scene 2: Single Example Point

Purpose: make the score concrete before showing a full field.

Setup:

1. Use a 2D coordinate plane.
2. Place a Gaussian density centered at `(1, 1)`.
3. Pick one sample point away from the center.
4. Draw an arrow from that point in the score direction.

Math:

```tex
p(x) = N(x; \mu, \sigma^2 I)
```

```tex
\nabla_x \log p(x) = -\frac{x - \mu}{\sigma^2}
```

Visual steps:

1. Show the center `(1, 1)`.
2. Show the sample point `x`.
3. Draw the vector from the point toward the center.
4. Show that smaller `sigma` creates a larger score magnitude.

### Scene 3: Full Score Field for One Gaussian

Purpose: show that every point in space has a score.

Visuals:

1. Fade in many arrows over a grid.
2. Arrows point toward `(1, 1)`.
3. Use a soft density-dot background.
4. Move the sample point along a path suggested by the field.

Key detail:

Normalize arrow length for visual readability while preserving true magnitude for color or animation decisions when useful.

### Scene 4: Two-Gaussian Data Distribution

Purpose: move from a single mode to a simple data distribution.

Distribution:

1. Gaussian A:
   - mean `(1, 1)`
   - sigma `0.35`
   - color teal
2. Gaussian B:
   - mean `(-1, -1)`
   - sigma `0.65`
   - color orange
3. Mixture weights:
   - `0.5 / 0.5`

Visuals:

1. Show two density blobs with different widths.
2. Render the score field of the mixture.
3. Mark the tight and wide modes.
4. Emphasize that mixture arrows follow `grad p(x) / p(x)`, not simply the nearest mean.

Math:

```tex
p(x) = w_1 N(x; \mu_1, \sigma_1^2 I) + w_2 N(x; \mu_2, \sigma_2^2 I)
```

```tex
\nabla_x \log p(x) = \frac{\nabla_x p(x)}{p(x)}
```

For the mixture:

```tex
\nabla_x p(x) =
w_1 N_1(x) \left(-\frac{x - \mu_1}{\sigma_1^2}\right)
+ w_2 N_2(x) \left(-\frac{x - \mu_2}{\sigma_2^2}\right)
```

### Scene 5: Bridge to Generative Dynamics

Purpose: connect score fields to diffusion-based generation.

Visuals:

1. Start points from broad noise.
2. Use small animated steps following the mixture field.
3. Let points drift toward high-density regions.
4. End with the same two-mode data distribution.

Message:

"Once we can estimate the right vector field at each noise level, generation becomes the problem of moving samples through these fields."

This scene stays conceptual. Detailed VE, VP, and flow matching comparisons belong in later clips.

## Implementation Notes

Use a self-contained Manim script for this first clip.

Math helpers:

1. `gaussian_pdf(x, mean, sigma)`
2. `gaussian_score(x, mean, sigma)`
3. `mixture_pdf(x, components)`
4. `mixture_score(x, components)`
5. `display_vector(vector, max_length)`
6. `field_step(point, components)`

Rendering helpers:

1. Consistent `NumberPlane`.
2. Density-dot background.
3. Mean markers.
4. Score arrows over a grid.
5. Sample path animation.

Design choices:

1. Use NumPy for math and Manim for rendering.
2. Use Manim `Text` instead of `MathTex` to avoid requiring LaTeX.
3. Keep the coordinate range fixed at roughly `[-3, 3] x [-3, 3]`.
4. Keep arrows sparse enough that the score field remains readable.
5. Use teal/orange for the two modes and yellow for emphasized sample paths.

## Render Commands

720p deliverable:

```bash
conda run -n py311 manim -qm --format mp4 --media_dir 2026-05-05/video_outputs -o A01-score_intuition_720p 2026-05-05/scripts/A01-score_intuition.py ScoreIntuition
```

480p preview:

```bash
conda run -n py311 manim -ql --format mp4 --media_dir 2026-05-05/video_outputs -o A01-score_intuition_480p 2026-05-05/scripts/A01-score_intuition.py ScoreIntuition
```

## Review Checklist

1. Equation is readable.
2. Arrows do not visually clutter the plane.
3. The single-point example is obvious.
4. The two Gaussian modes are visually distinct.
5. The bridge to generation is intuitive without overclaiming.
6. Render time is acceptable.

