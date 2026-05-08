# B01: VE Score Construction

## Clip Metadata

Stem: `B01-ve_score_construction`

Script: `2026-05-05_Diffusion-Score/scripts/B01-ve_score_construction.py`

Scene class: `VEScoreConstruction`

Canonical outputs:

1. `2026-05-05_Diffusion-Score/video_outputs/B01-ve_score_construction_720p.mp4`
2. `2026-05-05_Diffusion-Score/video_outputs/B01-ve_score_construction_480p.mp4`

## Purpose

Show the VE diffusion construction visually: if the clean data is simplified to one point, then adding Gaussian noise at different scales creates noisy training locations around that point. The reverse direction from each noisy location back toward the clean point is the score direction for that noise level.

This clip should answer:

1. How can a score field be constructed from a single data point?
2. What does VE forward noising do?
3. Why does increasing `sigma` cover more of the plane?
4. What does the reverse score direction look like?

## Core Message

For a toy data point `x0`, VE creates noisy copies:

```tex
x_sigma = x0 + sigma epsilon
```

Conditioned on that clean point:

```tex
p_sigma(x | x0) = N(x; x0, sigma^2 I)
```

Its score is:

```tex
score = grad_x log p_sigma(x | x0) = -(x - x0) / sigma^2
```

Visually, the score at a noisy point points back toward the clean data point. Larger `sigma` spreads noisy copies farther away, so the training process sees a wider region of space.

## Storyboard

### Scene 1: Start From One Clean Point

Purpose: simplify the problem before introducing VE.

Visuals:

1. Show a 2D coordinate plane.
2. Put one clean data point at the center.
3. Label it `x0`.
4. Add a caption: "Toy case: the data distribution is one point."

Narration idea:

"To understand VE, shrink the data distribution down to one point. What score field should surround this point?"

### Scene 2: Add Gaussian Noise Once

Purpose: make one noisy copy and its reverse direction concrete.

Visuals:

1. Sample noisy copies near `x0`.
2. Highlight one noisy point `P`.
3. Draw the noise displacement from `x0` to `P`.
4. Draw the reverse arrow from `P` back toward `x0`.
5. Label the reverse arrow "score direction."

Message:

"A noisy point was created by moving away from the clean point. The score direction points back."

### Scene 3: Increase `sigma`

Purpose: show how VE covers more of space.

Visuals:

1. Animate noise clouds for increasing `sigma` values.
2. Show expanding rings around the clean point.
3. Keep the clean point fixed.
4. Add labels such as `small sigma`, `medium sigma`, `large sigma`.

Message:

"In VE diffusion, larger noise levels spread samples farther away. Training over many noise levels covers more of the plane."

### Scene 4: Score Field at One Noise Level

Purpose: turn many noisy examples into a field.

Visuals:

1. Fade in arrows over a grid around the point.
2. Every arrow points inward toward `x0`.
3. Show the formula:

```tex
score = -(x - x0) / sigma^2
```

Key detail:

Normalize arrow length visually, but keep directions exact. The `1 / sigma^2` factor controls magnitude, not the inward direction.

### Scene 5: Reverse Motion

Purpose: connect the learned field to sampling.

Visuals:

1. Start several dots from broad noisy locations.
2. Animate them following inward score directions.
3. End with the dots near the clean data point.
4. Add the final message: "VE learns reverse fields from high noise back to data."

## Implementation Notes

Use a self-contained Manim script for this draft.

Helpers:

1. `gaussian_score(point, center, sigma)`
2. `display_vector(vector, max_length)`
3. `coordinate_circle(plane, center, radius)`
4. `noise_cloud(plane, points, color)`
5. `score_field(plane, center, sigma)`
6. `reverse_path(start, center)`

Design choices:

1. Use Manim `Text`, not `MathTex`, because LaTeX is not required for this draft.
2. Keep the visual grammar from A01: dark background, 2D plane, teal data, yellow emphasized arrows.
3. Use a single point before generalizing to real data distributions in later clips.
4. Avoid overexplaining stochastic differential equations in this clip.
5. Emphasize that this is the conditional score around one clean point, not yet the full data score for many points.

## Render Commands

720p deliverable:

```bash
conda run -n py311 manim -qm --format mp4 --media_dir 2026-05-05_Diffusion-Score/video_outputs -o B01-ve_score_construction_720p 2026-05-05_Diffusion-Score/scripts/B01-ve_score_construction.py VEScoreConstruction
```

480p preview:

```bash
conda run -n py311 manim -ql --format mp4 --media_dir 2026-05-05_Diffusion-Score/video_outputs -o B01-ve_score_construction_480p 2026-05-05_Diffusion-Score/scripts/B01-ve_score_construction.py VEScoreConstruction
```

## Review Checklist

1. The clean point remains visually stable.
2. Increasing `sigma` clearly expands the noisy region.
3. The reverse arrow from a noisy point back to `x0` is obvious.
4. The score field is readable and not over-cluttered.
5. The clip states that this is a toy one-point construction.
6. The final reverse motion intuitively contracts noisy points toward data.
