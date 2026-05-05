from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from manim import *


EPS = 1e-9
X_RANGE = (-3.0, 3.0)
Y_RANGE = (-3.0, 3.0)


@dataclass(frozen=True)
class GaussianComponent:
    weight: float
    mean: np.ndarray
    sigma: float
    color: ManimColor


def gaussian_pdf(points: np.ndarray, mean: np.ndarray, sigma: float) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    mean = np.asarray(mean, dtype=float)
    diff = points - mean
    norm_sq = np.sum(diff * diff, axis=-1)
    variance = sigma * sigma
    return np.exp(-0.5 * norm_sq / variance) / (2.0 * np.pi * variance)


def gaussian_score(points: np.ndarray, mean: np.ndarray, sigma: float) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    mean = np.asarray(mean, dtype=float)
    return -(points - mean) / (sigma * sigma)


def mixture_pdf(points: np.ndarray, components: list[GaussianComponent]) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    total = np.zeros(points.shape[:-1], dtype=float)
    for component in components:
        total += component.weight * gaussian_pdf(points, component.mean, component.sigma)
    return total


def mixture_score(points: np.ndarray, components: list[GaussianComponent]) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    was_single = points.ndim == 1
    batched = points[None, :] if was_single else points

    total_pdf = np.zeros(batched.shape[:-1], dtype=float)
    total_grad = np.zeros_like(batched, dtype=float)
    for component in components:
        weighted_pdf = component.weight * gaussian_pdf(
            batched, component.mean, component.sigma
        )
        score = gaussian_score(batched, component.mean, component.sigma)
        total_pdf += weighted_pdf
        total_grad += weighted_pdf[..., None] * score

    result = total_grad / (total_pdf[..., None] + EPS)
    return result[0] if was_single else result


def display_vector(vector: np.ndarray, max_length: float = 0.42) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < EPS:
        return np.zeros(2)
    length = max_length * np.tanh(norm / 2.5)
    return vector / norm * length


def field_step(point: np.ndarray, components: list[GaussianComponent]) -> np.ndarray:
    vector = mixture_score(point, components)
    norm = float(np.linalg.norm(vector))
    if norm < EPS:
        return np.zeros(2)
    return vector / norm * min(0.18, 0.055 * norm)


class ScoreIntuition(Scene):
    def construct(self) -> None:
        self.camera.background_color = "#10131a"

        intro_group = self._show_score_equation()
        self.play(FadeOut(intro_group, shift=UP * 0.25), run_time=0.8)

        plane = self._make_plane()
        header = self._make_header("A score field in 2D")
        self.play(FadeIn(header), Create(plane), run_time=1.2)

        mean = np.array([1.0, 1.0])
        single_sigma = 0.8
        components = [
            GaussianComponent(0.5, np.array([1.0, 1.0]), 0.35, TEAL_C),
            GaussianComponent(0.5, np.array([-1.0, -1.0]), 0.65, ORANGE),
        ]

        one_density, one_center, sample_dot, sample_arrow, sigma_label = (
            self._show_single_point_score(plane, mean, single_sigma, header)
        )
        single_field = self._show_single_gaussian_field(
            plane, mean, single_sigma, one_density, one_center, sample_dot, sample_arrow
        )
        mixture_density, mixture_field, mode_marks = self._show_mixture_field(
            plane, components, header, one_density, single_field, one_center, sigma_label
        )
        self._show_generation_bridge(
            plane, components, mixture_density, mixture_field, mode_marks, header
        )

        closing = Text(
            "Same intuition: learn the field, then move samples through it.",
            font_size=30,
            color=WHITE,
        ).to_edge(DOWN, buff=0.35)
        self.play(FadeIn(closing, shift=UP * 0.15), run_time=0.7)
        self.wait(1.5)

    def _show_score_equation(self) -> VGroup:
        title = Text(
            "The score points toward higher probability",
            font_size=40,
            weight="BOLD",
            color=WHITE,
        )
        score = Text("s_t(x)", font_size=46, color=YELLOW)
        equals = Text("=", font_size=46, color=WHITE)
        grad = Text("grad_x", font_size=46, color=BLUE_B)
        logp = Text("log p_t(x)", font_size=46, color=TEAL_B)
        equation = VGroup(score, equals, grad, logp).arrange(RIGHT, buff=0.16)
        subtitle = Text(
            "local direction where log-density increases fastest",
            font_size=26,
            color=GRAY_B,
        )

        group = VGroup(title, equation, subtitle).arrange(DOWN, buff=0.42)
        group.move_to(ORIGIN)
        self.play(Write(title), run_time=1.0)
        self.play(FadeIn(equation, shift=UP * 0.2), run_time=0.9)
        log_box = SurroundingRectangle(logp, color=TEAL_B, buff=0.08)
        grad_box = SurroundingRectangle(grad, color=BLUE_B, buff=0.08)
        score_box = SurroundingRectangle(equation, color=YELLOW, buff=0.12)
        self.play(Create(log_box), FadeIn(subtitle), run_time=0.8)
        self.wait(0.45)
        self.play(ReplacementTransform(log_box, grad_box), run_time=0.65)
        self.wait(0.35)
        self.play(ReplacementTransform(grad_box, score_box), run_time=0.65)
        self.wait(0.6)
        self.play(FadeOut(score_box), run_time=0.35)
        return group

    def _show_single_point_score(
        self,
        plane: NumberPlane,
        mean: np.ndarray,
        sigma: float,
        header: VGroup,
    ) -> tuple[VGroup, VGroup, Dot, Arrow, Text]:
        new_header = self._make_header("One Gaussian: a local arrow back to the mean")
        self.play(Transform(header, new_header), run_time=0.6)

        density = self._density_dots(
            plane,
            lambda p: gaussian_pdf(p, mean, sigma),
            lambda _p: TEAL_C,
            max_opacity=0.55,
        )
        center = self._mean_marker(plane, mean, "mean (1, 1)", TEAL_C)
        sample = np.array([2.15, 0.15])
        sample_dot = Dot(plane.c2p(*sample), radius=0.075, color=YELLOW).set_z_index(4)
        sample_label = Text("x", font_size=24, color=YELLOW).next_to(
            sample_dot, DOWN + RIGHT, buff=0.08
        )
        sample_group = VGroup(sample_dot, sample_label)

        arrow = self._score_arrow(
            plane,
            sample,
            gaussian_score(sample, mean, sigma),
            color=YELLOW,
            max_length=0.72,
            stroke_width=7,
        )
        arrow_label = Text(
            "score at x",
            font_size=22,
            color=YELLOW,
        ).next_to(arrow, RIGHT, buff=0.12)
        sigma_label = Text(
            "For a Gaussian: score = -(x - mean) / sigma^2",
            font_size=24,
            color=GRAY_A,
        ).to_edge(DOWN, buff=0.35)

        self.play(FadeIn(density, lag_ratio=0.02), FadeIn(center), run_time=1.1)
        self.play(FadeIn(sample_group, scale=0.85), run_time=0.5)
        self.play(GrowArrow(arrow), FadeIn(arrow_label), run_time=0.8)
        self.play(FadeIn(sigma_label, shift=UP * 0.12), run_time=0.6)

        tighter_arrow = self._score_arrow(
            plane,
            sample,
            gaussian_score(sample, mean, 0.52),
            color=RED_B,
            max_length=1.05,
            stroke_width=7,
        )
        tighter_label = Text(
            "smaller sigma, larger magnitude",
            font_size=22,
            color=RED_B,
        ).next_to(tighter_arrow, RIGHT, buff=0.1)
        self.play(
            Transform(arrow, tighter_arrow),
            Transform(arrow_label, tighter_label),
            run_time=0.8,
        )
        self.wait(0.55)
        self.play(FadeOut(arrow_label), FadeOut(sample_label), run_time=0.35)
        return density, center, sample_dot, arrow, sigma_label

    def _show_single_gaussian_field(
        self,
        plane: NumberPlane,
        mean: np.ndarray,
        sigma: float,
        density: VGroup,
        center: VGroup,
        sample_dot: Dot,
        sample_arrow: Arrow,
    ) -> VGroup:
        field_label = Text(
            "Every point has its own score.",
            font_size=28,
            color=WHITE,
        ).to_edge(DOWN, buff=0.35)
        field = self._vector_field(
            plane,
            lambda p: gaussian_score(p, mean, sigma),
            base_color=BLUE_B,
            max_length=0.38,
        )
        self.play(FadeOut(sample_arrow), FadeIn(field_label), run_time=0.5)
        self.play(LaggedStart(*[GrowArrow(a) for a in field], lag_ratio=0.015), run_time=1.8)

        path_points = [
            np.array([2.15, 0.15]),
            np.array([1.82, 0.36]),
            np.array([1.52, 0.57]),
            np.array([1.28, 0.77]),
            np.array([1.10, 0.94]),
        ]
        path = VMobject(color=YELLOW, stroke_width=4).set_points_smoothly(
            [plane.c2p(*p) for p in path_points]
        )
        self.play(Create(path), MoveAlongPath(sample_dot, path), run_time=1.5)
        self.wait(0.45)
        self.play(FadeOut(path), FadeOut(sample_dot), FadeOut(field_label), run_time=0.5)
        return field

    def _show_mixture_field(
        self,
        plane: NumberPlane,
        components: list[GaussianComponent],
        header: VGroup,
        old_density: VGroup,
        old_field: VGroup,
        old_center: VGroup,
        old_bottom_label: Text,
    ) -> tuple[VGroup, VGroup, VGroup]:
        new_header = self._make_header("Two modes: the score follows the log mixture")
        self.play(Transform(header, new_header), run_time=0.6)

        density = self._density_dots(
            plane,
            lambda p: mixture_pdf(p, components),
            lambda p: self._responsibility_color(p, components),
            max_opacity=0.58,
        )
        field = self._vector_field(
            plane,
            lambda p: mixture_score(p, components),
            base_color=WHITE,
            max_length=0.36,
            color_by_magnitude=True,
        )
        mode_marks = VGroup(
            self._mean_marker(plane, components[0].mean, "tight mode", components[0].color),
            self._mean_marker(plane, components[1].mean, "wide mode", components[1].color),
        )
        formula = Text(
            "mixture score = grad p(x) / p(x), not just nearest mean",
            font_size=23,
            color=GRAY_A,
        ).to_edge(DOWN, buff=0.35)

        self.play(
            FadeOut(old_center),
            FadeOut(old_bottom_label),
            Transform(old_density, density),
            Transform(old_field, field),
            run_time=1.3,
        )
        self.play(FadeIn(mode_marks), FadeIn(formula, shift=UP * 0.12), run_time=0.7)
        self.wait(1.0)
        self.play(FadeOut(formula), run_time=0.35)
        return old_density, old_field, mode_marks

    def _show_generation_bridge(
        self,
        plane: NumberPlane,
        components: list[GaussianComponent],
        density: VGroup,
        field: VGroup,
        mode_marks: VGroup,
        header: VGroup,
    ) -> None:
        new_header = self._make_header("Generation: move noisy samples through a field")
        self.play(Transform(header, new_header), run_time=0.6)

        rng = np.random.default_rng(13)
        starts = np.clip(rng.normal(0.0, 1.55, size=(16, 2)), -2.75, 2.75)
        paths = [self._trace_path(start, components) for start in starts]
        dots = VGroup(
            *[
                Dot(plane.c2p(*path[0]), radius=0.045, color=WHITE).set_z_index(5)
                for path in paths
            ]
        )
        curves = VGroup(
            *[
                VMobject(color=YELLOW, stroke_width=2.5)
                .set_opacity(0.55)
                .set_points_smoothly([plane.c2p(*p) for p in path])
                for path in paths
            ]
        )
        caption = Text(
            "Estimate the right field at each noise level; then follow it.",
            font_size=24,
            color=GRAY_A,
        ).to_edge(DOWN, buff=0.35)

        self.play(FadeIn(dots, lag_ratio=0.04), FadeIn(caption), run_time=0.8)
        self.play(Create(curves), run_time=1.0)
        self.play(
            LaggedStart(
                *[
                    MoveAlongPath(dot, curve)
                    for dot, curve in zip(dots, curves, strict=True)
                ],
                lag_ratio=0.025,
            ),
            run_time=4.0,
        )
        self.wait(0.6)
        self.play(
            FadeOut(caption),
            FadeOut(curves),
            FadeOut(dots),
            density.animate.set_opacity(0.85),
            field.animate.set_opacity(0.55),
            mode_marks.animate.set_opacity(0.9),
            run_time=0.7,
        )

    def _make_header(self, text: str) -> VGroup:
        title = Text(text, font_size=27, color=WHITE, weight="BOLD")
        equation = Text("s_t(x) = grad_x log p_t(x)", font_size=22, color=YELLOW)
        return VGroup(title, equation).arrange(RIGHT, buff=0.55).to_edge(UP, buff=0.24)

    def _make_plane(self) -> NumberPlane:
        plane = NumberPlane(
            x_range=[X_RANGE[0], X_RANGE[1], 1],
            y_range=[Y_RANGE[0], Y_RANGE[1], 1],
            x_length=7.4,
            y_length=5.75,
            background_line_style={
                "stroke_color": "#334155",
                "stroke_width": 1,
                "stroke_opacity": 0.35,
            },
            axis_config={
                "stroke_color": "#94a3b8",
                "stroke_width": 1.6,
                "stroke_opacity": 0.75,
            },
        )
        plane.to_edge(DOWN, buff=0.45)
        return plane

    def _mean_marker(
        self, plane: NumberPlane, mean: np.ndarray, label: str, color: ManimColor
    ) -> VGroup:
        dot = Dot(plane.c2p(*mean), radius=0.085, color=color).set_z_index(4)
        ring = Circle(radius=0.17, color=color, stroke_width=3).move_to(dot)
        text = Text(label, font_size=19, color=color).next_to(dot, UP + RIGHT, buff=0.08)
        return VGroup(ring, dot, text)

    def _density_dots(
        self,
        plane: NumberPlane,
        density_fn,
        color_fn,
        max_opacity: float,
    ) -> VGroup:
        xs = np.linspace(X_RANGE[0], X_RANGE[1], 31)
        ys = np.linspace(Y_RANGE[0], Y_RANGE[1], 31)
        grid = np.array([[x, y] for x in xs for y in ys])
        densities = np.asarray([density_fn(point) for point in grid], dtype=float)
        max_density = float(np.max(densities)) + EPS

        dots = []
        for point, density in zip(grid, densities, strict=True):
            normalized = float(density / max_density)
            if normalized < 0.015:
                continue
            radius = 0.018 + 0.036 * np.sqrt(normalized)
            opacity = min(max_opacity, 0.08 + max_opacity * normalized**0.72)
            dot = Dot(
                plane.c2p(point[0], point[1]),
                radius=radius,
                color=color_fn(point),
            )
            dot.set_opacity(opacity)
            dot.set_z_index(0)
            dots.append(dot)
        return VGroup(*dots)

    def _vector_field(
        self,
        plane: NumberPlane,
        field_fn,
        base_color: ManimColor,
        max_length: float,
        color_by_magnitude: bool = False,
    ) -> VGroup:
        arrows = []
        xs = np.linspace(-2.7, 2.7, 10)
        ys = np.linspace(-2.7, 2.7, 10)
        for x in xs:
            for y in ys:
                point = np.array([x, y])
                vector = np.asarray(field_fn(point), dtype=float)
                norm = float(np.linalg.norm(vector))
                if norm < EPS:
                    continue
                color = base_color
                if color_by_magnitude:
                    blend = min(1.0, norm / 6.0)
                    color = interpolate_color(BLUE_B, YELLOW, blend)
                arrows.append(
                    self._score_arrow(
                        plane,
                        point,
                        vector,
                        color=color,
                        max_length=max_length,
                        stroke_width=3.4,
                    ).set_opacity(0.78)
                )
        return VGroup(*arrows)

    def _score_arrow(
        self,
        plane: NumberPlane,
        point: np.ndarray,
        vector: np.ndarray,
        color: ManimColor,
        max_length: float,
        stroke_width: float,
    ) -> Arrow:
        shown = display_vector(vector, max_length=max_length)
        end = point + shown
        return Arrow(
            plane.c2p(point[0], point[1]),
            plane.c2p(end[0], end[1]),
            buff=0.0,
            color=color,
            stroke_width=stroke_width,
            max_tip_length_to_length_ratio=0.28,
        ).set_z_index(3)

    def _responsibility_color(
        self, point: np.ndarray, components: list[GaussianComponent]
    ) -> ManimColor:
        weights = np.array(
            [
                component.weight
                * gaussian_pdf(point, component.mean, component.sigma)
                for component in components
            ]
        )
        total = float(np.sum(weights)) + EPS
        responsibility = float(weights[0] / total)
        return interpolate_color(components[1].color, components[0].color, responsibility)

    def _trace_path(
        self,
        start: np.ndarray,
        components: list[GaussianComponent],
        steps: int = 34,
    ) -> list[np.ndarray]:
        point = np.array(start, dtype=float)
        path = [point.copy()]
        for _ in range(steps):
            point = point + field_step(point, components)
            point[0] = np.clip(point[0], X_RANGE[0] + 0.1, X_RANGE[1] - 0.1)
            point[1] = np.clip(point[1], Y_RANGE[0] + 0.1, Y_RANGE[1] - 0.1)
            path.append(point.copy())
        return path
