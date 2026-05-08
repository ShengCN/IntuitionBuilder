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


def gaussian_log_pdf(points: np.ndarray, mean: np.ndarray, sigma: float) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    mean = np.asarray(mean, dtype=float)
    diff = points - mean
    norm_sq = np.sum(diff * diff, axis=-1)
    variance = sigma * sigma
    return -0.5 * norm_sq / variance - np.log(2.0 * np.pi * variance)


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

        mean = np.array([1.0, 1.0])
        single_sigma = 0.8
        components = [
            GaussianComponent(0.5, np.array([1.0, 1.0]), 0.35, TEAL_C),
            GaussianComponent(0.5, np.array([-1.0, -1.0]), 0.65, ORANGE),
        ]

        intro_group = self._show_score_equation()
        self.play(FadeOut(intro_group, shift=UP * 0.25), run_time=0.8)

        plane = self._make_plane()
        header = self._make_header("First visualize log p(x)")
        self.play(FadeIn(header), Create(plane), run_time=1.2)

        contours, center, log_caption = self._show_log_density_contours(
            plane, mean, single_sigma, header
        )
        sample_dot, score_arrow, local_annotations = self._show_one_point_derivative(
            plane, mean, single_sigma, header, log_caption
        )
        single_field = self._show_single_gaussian_field(
            plane,
            mean,
            single_sigma,
            contours,
            center,
            sample_dot,
            score_arrow,
            local_annotations,
            header,
        )
        mixture_density, mixture_field, mode_marks = self._show_mixture_field(
            plane, components, header, contours, single_field, center
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
            "Score = gradient of log-density",
            font_size=42,
            weight="BOLD",
            color=WHITE,
        )
        score = Text("s(x)", font_size=48, color=YELLOW)
        equals = Text("=", font_size=48, color=WHITE)
        grad = Text("∇ₓ", font_size=48, color=BLUE_B)
        logp = Text("log p(x)", font_size=48, color=TEAL_B)
        equation = VGroup(score, equals, grad, logp).arrange(RIGHT, buff=0.16)
        subtitle = Text(
            "differentiate a scalar log-density landscape",
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
        self.wait(0.35)
        self.play(ReplacementTransform(log_box, grad_box), run_time=0.65)
        self.wait(0.3)
        self.play(ReplacementTransform(grad_box, score_box), run_time=0.65)
        self.wait(0.55)
        self.play(FadeOut(score_box), run_time=0.35)
        return group

    def _show_log_density_contours(
        self,
        plane: NumberPlane,
        mean: np.ndarray,
        sigma: float,
        header: VGroup,
    ) -> tuple[VGroup, VGroup, VGroup]:
        new_header = self._make_header("A 2D Gaussian defines log p(x) everywhere")
        self.play(Transform(header, new_header), run_time=0.6)

        contours = self._gaussian_log_contours(plane, mean, sigma)
        center = self._mean_marker(plane, mean, "mean μ", TEAL_C)
        high_label = Text("higher log p(x)", font_size=21, color=TEAL_A).next_to(
            center, UP + RIGHT, buff=0.28
        ).set_z_index(6)
        low_label = Text("lower log p(x)", font_size=21, color=GRAY_B).move_to(
            plane.c2p(-1.6, -1.95)
        ).set_z_index(6)
        caption = VGroup(
            Text("log p(x) is a scalar landscape over the 2D plane", font_size=24, color=GRAY_A),
            Text("contours connect points with equal log-density", font_size=22, color=GRAY_B),
        ).arrange(DOWN, buff=0.14).to_edge(DOWN, buff=0.3).set_z_index(6)

        self.play(
            LaggedStart(*[Create(ring) for ring in contours], lag_ratio=0.12),
            FadeIn(center),
            run_time=1.5,
        )
        self.play(
            FadeIn(high_label, shift=UP * 0.12),
            FadeIn(low_label, shift=DOWN * 0.12),
            FadeIn(caption, shift=UP * 0.12),
            run_time=0.8,
        )
        self.wait(0.8)
        return contours, VGroup(center, high_label, low_label), caption

    def _show_one_point_derivative(
        self,
        plane: NumberPlane,
        mean: np.ndarray,
        sigma: float,
        header: VGroup,
        old_caption: VGroup,
    ) -> tuple[Dot, Arrow, VGroup]:
        new_header = self._make_header("The score is the derivative at one point")
        self.play(Transform(header, new_header), FadeOut(old_caption), run_time=0.6)

        sample = np.array([2.15, 0.15])
        sample_dot = Dot(plane.c2p(*sample), radius=0.075, color=YELLOW).set_z_index(5)
        sample_label = Text("x", font_size=25, color=YELLOW).next_to(
            sample_dot, DOWN + RIGHT, buff=0.08
        )

        radial_to_mean = mean - sample
        tangent = np.array([-radial_to_mean[1], radial_to_mean[0]], dtype=float)
        tangent /= np.linalg.norm(tangent) + EPS
        tangent_line = DashedLine(
            plane.c2p(*(sample - 0.62 * tangent)),
            plane.c2p(*(sample + 0.62 * tangent)),
            color=GRAY_B,
            dash_length=0.08,
            stroke_width=2.2,
        ).set_z_index(2)
        guide = DashedLine(
            plane.c2p(*sample),
            plane.c2p(*mean),
            color=TEAL_B,
            dash_length=0.08,
            stroke_width=2.4,
        ).set_opacity(0.65)
        tangent_label = Text("local contour", font_size=20, color=GRAY_B).next_to(
            tangent_line, DOWN, buff=0.12
        )

        score_arrow = self._score_arrow(
            plane,
            sample,
            gaussian_score(sample, mean, sigma),
            color=YELLOW,
            max_length=0.82,
            stroke_width=7,
        )
        arrow_label = Text("s(x)", font_size=24, color=YELLOW).next_to(
            score_arrow, RIGHT, buff=0.1
        )
        derivative_caption = VGroup(
            Text("s(x) = ∇ₓ log p(x)", font_size=26, color=YELLOW),
            Text("gradient points toward fastest increase of log p(x)", font_size=22, color=GRAY_A),
        ).arrange(DOWN, buff=0.15).to_edge(DOWN, buff=0.3)

        formula = VGroup(
            Text("Gaussian log-density:", font_size=21, color=GRAY_B),
            Text("log p(x) = C − ||x − μ||² / (2σ²)", font_size=24, color=TEAL_A),
            Text("s(x) = −(x − μ) / σ²", font_size=24, color=YELLOW),
        ).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        formula.to_corner(DOWN + LEFT, buff=0.35)

        self.play(FadeIn(sample_dot), FadeIn(sample_label), run_time=0.5)
        self.play(Create(tangent_line), FadeIn(tangent_label), run_time=0.65)
        self.play(Create(guide), FadeIn(derivative_caption, shift=UP * 0.12), run_time=0.7)
        self.play(GrowArrow(score_arrow), FadeIn(arrow_label), run_time=0.85)
        self.wait(0.5)
        self.play(
            ReplacementTransform(derivative_caption, formula),
            run_time=0.8,
        )
        self.wait(0.7)
        local_annotations = VGroup(
            sample_label,
            tangent_line,
            tangent_label,
            guide,
            arrow_label,
            formula,
        )
        return sample_dot, score_arrow, local_annotations

    def _show_single_gaussian_field(
        self,
        plane: NumberPlane,
        mean: np.ndarray,
        sigma: float,
        contours: VGroup,
        center: VGroup,
        sample_dot: Dot,
        sample_arrow: Arrow,
        local_annotations: VGroup,
        header: VGroup,
    ) -> VGroup:
        new_header = self._make_header("Repeat the derivative across the plane")
        self.play(Transform(header, new_header), run_time=0.6)

        field = self._vector_field(
            plane,
            lambda p: gaussian_score(p, mean, sigma),
            base_color=BLUE_B,
            max_length=0.38,
        )
        field_label = Text(
            "Every point gets a score vector.",
            font_size=25,
            color=GRAY_A,
        ).to_edge(DOWN, buff=0.3)

        self.play(
            FadeOut(sample_arrow),
            FadeOut(local_annotations),
            FadeIn(field_label, shift=UP * 0.12),
            run_time=0.55,
        )
        self.play(
            LaggedStart(*[GrowArrow(arrow) for arrow in field], lag_ratio=0.014),
            contours.animate.set_opacity(0.42),
            center.animate.set_opacity(0.9),
            run_time=1.8,
        )

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
        old_contours: VGroup,
        old_field: VGroup,
        old_center: VGroup,
    ) -> tuple[VGroup, VGroup, VGroup]:
        new_header = self._make_header("A data distribution has a richer score field")
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
            "mixture score = ∇ₓ log p(x) = ∇ₓ p(x) / p(x)",
            font_size=23,
            color=GRAY_A,
        ).to_edge(DOWN, buff=0.32)

        self.play(
            FadeOut(old_contours),
            FadeOut(old_field),
            FadeOut(old_center),
            run_time=0.6,
        )
        self.play(FadeIn(density, lag_ratio=0.02), FadeIn(mode_marks), run_time=1.0)
        self.play(
            LaggedStart(*[GrowArrow(arrow) for arrow in field], lag_ratio=0.01),
            FadeIn(formula, shift=UP * 0.12),
            run_time=1.6,
        )
        self.wait(0.8)
        self.play(FadeOut(formula), run_time=0.35)
        return density, field, mode_marks

    def _show_generation_bridge(
        self,
        plane: NumberPlane,
        components: list[GaussianComponent],
        density: VGroup,
        field: VGroup,
        mode_marks: VGroup,
        header: VGroup,
    ) -> None:
        new_header = self._make_header("Generation: move noisy samples through fields")
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
        equation = Text("s(x) = ∇ₓ log p(x)", font_size=22, color=YELLOW)
        return VGroup(title, equation).arrange(RIGHT, buff=0.55).to_edge(UP, buff=0.24)

    def _make_plane(self) -> NumberPlane:
        plane = NumberPlane(
            x_range=[X_RANGE[0], X_RANGE[1], 1],
            y_range=[Y_RANGE[0], Y_RANGE[1], 1],
            x_length=5.85,
            y_length=5.85,
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

    def _gaussian_log_contours(
        self, plane: NumberPlane, mean: np.ndarray, sigma: float
    ) -> VGroup:
        radii = [0.45, 0.85, 1.25, 1.43, 1.85, 2.25]
        rings = []
        for index, radius in enumerate(radii):
            alpha = 1.0 - index / max(1, len(radii) - 1)
            color = interpolate_color(BLUE_E, TEAL_A, alpha)
            ring = self._coordinate_circle(plane, mean, radius)
            stroke_opacity = 0.38 + 0.38 * alpha
            stroke_width = 3.0 if radius == 1.43 else 2.4
            ring.set_stroke(color=color, width=stroke_width, opacity=stroke_opacity)
            ring.set_fill(opacity=0.0)
            ring.set_z_index(1)
            rings.append(ring)
        return VGroup(*rings)

    def _coordinate_circle(
        self,
        plane: NumberPlane,
        center: np.ndarray,
        radius: float,
        samples: int = 160,
    ) -> VMobject:
        angles = np.linspace(0.0, TAU, samples)
        points = [
            plane.c2p(center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle))
            for angle in angles
        ]
        return VMobject().set_points_smoothly(points).set_fill(opacity=0.0)

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


class ScoreIntuitionScene1(ScoreIntuition):
    def construct(self) -> None:
        self.camera.background_color = "#10131a"

        mean = np.array([1.0, 1.0])
        sigma = 0.8
        sample = np.array([2.15, 0.15])

        title = Text(
            "A probability landscape has an uphill direction",
            font_size=34,
            weight="BOLD",
            color=WHITE,
        ).to_edge(UP, buff=0.28)
        subtitle = Text(
            "Before the formula, look at one point on log p(x).",
            font_size=23,
            color=GRAY_A,
        ).next_to(title, DOWN, buff=0.15)
        title_group = VGroup(title, subtitle)

        plane = self._make_plane()
        plane.shift(DOWN * 0.1)

        density = self._density_dots(
            plane,
            lambda p: gaussian_pdf(p, mean, sigma),
            lambda _p: TEAL_C,
            max_opacity=0.48,
        )
        contours = self._gaussian_log_contours(plane, mean, sigma)
        contours.set_z_index(2)

        center = self._mean_marker(plane, mean, "highest log p(x)", TEAL_A)
        center.set_z_index(5)
        low_label = Text("lower log p(x)", font_size=20, color=GRAY_B)
        low_label.move_to(plane.c2p(-1.35, -1.75)).set_z_index(5)

        self.play(FadeIn(title_group, shift=DOWN * 0.12), run_time=0.8)
        self.play(Create(plane), run_time=0.75)
        self.play(FadeIn(density, lag_ratio=0.015), run_time=1.0)
        self.play(
            LaggedStart(*[Create(ring) for ring in contours], lag_ratio=0.1),
            FadeIn(center),
            FadeIn(low_label, shift=DOWN * 0.08),
            run_time=1.45,
        )

        map_caption = VGroup(
            Text("Contour lines: same log-density", font_size=23, color=GRAY_A),
            Text("Brighter means higher log p(x)", font_size=22, color=TEAL_A),
        ).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        map_caption.to_corner(DOWN + LEFT, buff=0.35).set_z_index(6)
        self.play(FadeIn(map_caption, shift=UP * 0.1), run_time=0.55)
        self.wait(0.35)

        sample_dot = Dot(plane.c2p(*sample), radius=0.08, color=YELLOW).set_z_index(6)
        sample_label = Text("x", font_size=26, color=YELLOW).next_to(
            sample_dot, DOWN + RIGHT, buff=0.08
        )
        question = Text(
            "From here, which way goes uphill fastest?",
            font_size=26,
            color=WHITE,
        )
        question.to_edge(DOWN, buff=0.35).set_z_index(6)

        self.play(FadeIn(sample_dot, scale=0.85), FadeIn(sample_label), run_time=0.55)
        self.play(ReplacementTransform(map_caption, question), run_time=0.65)
        self.wait(0.35)

        radial_to_mean = mean - sample
        tangent = np.array([-radial_to_mean[1], radial_to_mean[0]], dtype=float)
        tangent /= np.linalg.norm(tangent) + EPS
        local_contour = DashedLine(
            plane.c2p(*(sample - 0.62 * tangent)),
            plane.c2p(*(sample + 0.62 * tangent)),
            color=GRAY_B,
            dash_length=0.08,
            stroke_width=2.4,
        ).set_z_index(4)
        contour_label = Text("local contour", font_size=19, color=GRAY_B).next_to(
            local_contour, DOWN, buff=0.1
        )

        score_arrow = self._score_arrow(
            plane,
            sample,
            gaussian_score(sample, mean, sigma),
            color=YELLOW,
            max_length=0.9,
            stroke_width=7,
        )
        arrow_label = Text("uphill direction", font_size=22, color=YELLOW).next_to(
            score_arrow, RIGHT, buff=0.12
        )

        self.play(Create(local_contour), FadeIn(contour_label), run_time=0.65)
        self.play(
            GrowArrow(score_arrow),
            FadeIn(arrow_label, shift=RIGHT * 0.08),
            run_time=0.85,
        )
        self.wait(0.45)

        score_text = Text(
            "This local uphill arrow is the score.",
            font_size=27,
            color=WHITE,
        )
        equation = Text("s(x) = ∇ₓ log p(x)", font_size=34, color=YELLOW)
        equation_hint = Text(
            "a direction attached to the point x",
            font_size=21,
            color=GRAY_A,
        )
        reveal_content = VGroup(score_text, equation, equation_hint).arrange(
            DOWN, buff=0.16
        )
        reveal_backdrop = BackgroundRectangle(
            reveal_content,
            color="#10131a",
            fill_opacity=0.82,
            buff=0.16,
        )
        reveal = VGroup(reveal_backdrop, reveal_content)
        reveal.to_edge(DOWN, buff=0.28).set_z_index(6)

        self.play(
            FadeOut(question),
            FadeIn(reveal, shift=UP * 0.08),
            arrow_label.animate.set_color(YELLOW),
            contours.animate.set_stroke(opacity=0.5),
            run_time=0.9,
        )
        box = SurroundingRectangle(equation, color=YELLOW, buff=0.11)
        self.play(Create(box), run_time=0.45)
        self.wait(1.4)
