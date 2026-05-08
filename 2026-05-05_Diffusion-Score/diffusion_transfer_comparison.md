# Diffusion Transfer Comparison Visualization

## Workflow

Use `manim.skill` for Manim project layout, naming, rendering, verification, and concatenation conventions.

This file is the project-level index. Keep detailed per-clip storyboards in `2026-05-05_Diffusion-Score/storyboards/`, Manim code in `2026-05-05_Diffusion-Score/scripts/`, and canonical rendered clips in `2026-05-05_Diffusion-Score/video_outputs/`.

## Goal

1. Explain why the score is the key intuition behind diffusion-based generative methods.
2. Use Manim animations to visualize the transition processes behind VE diffusion, VP diffusion, and flow matching.
3. Build the final video as ordered clips that can be revised independently and concatenated later.

## High-Level Logic

The score is the local direction in which log-density increases:

$$
s_t(x) = \nabla_x \log p_t(x)
$$

The video should first make this field intuition concrete in 2D. Then it should use the same visual language to compare how VE, VP, and flow matching define different dynamics between simple noise and structured data.

The common thread:

1. A distribution induces a field over sample space.
2. A generative process moves samples through a sequence of fields or velocity directions.
3. VE, VP, and flow matching differ in how they define the path, but all can be compared by watching how probability mass moves.

## Current Clip Index


| Clip                | Purpose                                                                  | Storyboard                           | Script                           | Output                                       | Status   |
| ------------------- | ------------------------------------------------------------------------ | ------------------------------------ | -------------------------------- | -------------------------------------------- | -------- |
| A01-score_intuition | Introduce `log p(x)` as 2D contours, then show score as the gradient field and bridge to generation. | `storyboards/A01-score_intuition.md` | `scripts/A01-score_intuition.py` | `video_outputs/A01-score_intuition_720p.mp4` | Parked |
| B01-ve_score_construction | Show VE's one-point score construction by adding Gaussian noise at increasing `sigma` levels. | `storyboards/B01-ve_score_construction.md` | `scripts/B01-ve_score_construction.py` | `video_outputs/B01-ve_score_construction_720p.mp4` | Draft rendered |


## Planned Video Structure

### Section A: Score Foundations

Goal: establish the visual language.

Current status: parked. Do not continue building Section A until the opening score-intuition manuscript is revisited.

1. `A01-score_intuition`: show `log p(x)` contours for one Gaussian, derive one score arrow, expand to a full score field, then bridge to a mixture-field generation sketch.
2. Possible next clip: score fields across noise levels.

### Section B: how to construct the score field? VE Diffusion version.

Goal: show VE's way to construct the score field.

Current draft:

1. `B01-ve_score_construction`: simplify the data distribution to one clean point `x0`.
2. Add Gaussian noise: `x_sigma = x0 + sigma epsilon`.
3. Show that each noisy point `P` gives a reverse score direction back toward `x0`.
4. Increase `sigma` to show wider spatial coverage.
5. Show the inward score field for one noise level.
6. Animate reverse motion from broad noise back to the clean point.

Important framing:

1. This is the conditional score around one clean point, not yet the full data score over a multi-point distribution.
2. Larger `sigma` gives broader coverage; the reverse field at each noise level points back toward higher conditional density.
3. Later clips can generalize from one point to many data samples or compare against VP and flow matching.


### Section C: VP Diffusion

Goal: show variance preserving diffusion.

High-level idea:

1. Start from clean data.
2. Shrink signal while adding noise.
3. Show data structure fading into a controlled noise distribution.
4. Compare the visual path against VE.

### Section D: Flow Matching

Goal: show direct transport through a learned velocity field.

High-level idea:

1. Start with noise samples.
2. End with data samples.
3. Animate paths between source and target distributions.
4. Compare velocity-field intuition with score-field intuition.

### Section E: Side-by-Side Comparison

Goal: compare VE, VP, and flow matching with one consistent visual grammar.

High-level idea:

1. Use the same data distribution in all panels.
2. Show different trajectories from noise to data.
3. End with the message that the methods define different dynamics, but each needs a field that tells samples where to move.

## Current Artifacts

Source:

1. `2026-05-05_Diffusion-Score/storyboards/A01-score_intuition.md`
2. `2026-05-05_Diffusion-Score/scripts/A01-score_intuition.py`
3. `2026-05-05_Diffusion-Score/storyboards/B01-ve_score_construction.md`
4. `2026-05-05_Diffusion-Score/scripts/B01-ve_score_construction.py`

Rendered outputs:

1. `2026-05-05_Diffusion-Score/video_outputs/A01-score_intuition_720p.mp4`
2. `2026-05-05_Diffusion-Score/video_outputs/A01-score_intuition_480p.mp4`
3. `2026-05-05_Diffusion-Score/video_outputs/B01-ve_score_construction_720p.mp4`
4. `2026-05-05_Diffusion-Score/video_outputs/B01-ve_score_construction_480p.mp4`

Render commands:

```bash
conda run -n py311 manim -qm --format mp4 --media_dir 2026-05-05_Diffusion-Score/video_outputs -o A01-score_intuition_720p 2026-05-05_Diffusion-Score/scripts/A01-score_intuition.py ScoreIntuition
```

```bash
conda run -n py311 manim -ql --format mp4 --media_dir 2026-05-05_Diffusion-Score/video_outputs -o A01-score_intuition_480p 2026-05-05_Diffusion-Score/scripts/A01-score_intuition.py ScoreIntuition
```

```bash
conda run -n py311 manim -qm --format mp4 --media_dir 2026-05-05_Diffusion-Score/video_outputs -o B01-ve_score_construction_720p 2026-05-05_Diffusion-Score/scripts/B01-ve_score_construction.py VEScoreConstruction
```

```bash
conda run -n py311 manim -ql --format mp4 --media_dir 2026-05-05_Diffusion-Score/video_outputs -o B01-ve_score_construction_480p 2026-05-05_Diffusion-Score/scripts/B01-ve_score_construction.py VEScoreConstruction
```
