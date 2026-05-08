from __future__ import annotations

import numpy as np
from manim import *


EPS = 1e-9
X_RANGE = (-3.2, 3.2)
Y_RANGE = (-3.2, 3.2)


def gaussian_score(point: np.ndarray, center: np.ndarray, sigma: float) -> np.ndarray:
    point = np.asarray(point, dtype=float)
    center = np.asarray(center, dtype=float)
    return -(point - center) / (sigma * sigma)


def display_vector(vector: np.ndarray, max_length: float = 0.42) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < EPS:
        return np.zeros(2)
    length = max_length * np.tanh(norm / 2.4)
    return vector / norm * length


class VEScoreConstruction(Scene):
    def construct(self) -> None:
        self.camera.background_color = "#0b1020"

        center = np.array([0.0, 0.0])
        sigmas = [0.35, 0.75, 1.25, 1.95]
        colors = [TEAL_A, BLUE_B, GOLD_C, ORANGE]
        labels = [
            "small sigma: noisy copies stay nearby",
            "medium sigma: coverage grows",
            "large sigma: more of the plane is visited",
            "very large sigma: broad noise surrounds the point",
        ]

        title = Text(
            "VE diffusion: construct scores by adding noise",
            font_size=34,
            weight="BOLD",
            color=WHITE,
        ).to_edge(UP, buff=0.25)
        subtitle = Text(
            "Toy case: the data distribution is one clean point.",
            font_size=22,
            color=GRAY_A,
        ).next_to(title, DOWN, buff=0.12)

        plane = self._make_plane()
        panel = self._make_panel()
        data_mark = self._data_marker(plane, center)

        self.play(FadeIn(VGroup(title, subtitle), shift=DOWN * 0.12), run_time=0.8)
        self.play(Create(plane), FadeIn(panel), FadeIn(data_mark), run_time=1.0)

        panel_step_1 = self._panel_text(
            [
                "Start with one data point:",
                "x0",
                "Question: what field should surround it?",
            ],
            [GRAY_A, TEAL_A, WHITE],
        )
        self.play(FadeIn(panel_step_1, shift=LEFT * 0.08), run_time=0.7)
        self.wait(0.4)

        rng = np.random.default_rng(8)
        base_noise = rng.normal(size=(96, 2))
        cloud = self._noise_cloud(plane, center + sigmas[0] * base_noise, colors[0])
        ring = self._coordinate_circle(plane, center, radius=2.0 * sigmas[0])
        ring.set_stroke(colors[0], width=3, opacity=0.72).set_z_index(1)
        sigma_label = Text(labels[0], font_size=23, color=colors[0])
        sigma_label.to_edge(DOWN, buff=0.34)

        panel_step_2 = self._panel_text(
            [
                "VE forward noising:",
                "x_sigma = x0 + sigma epsilon",
                "Noise creates training locations.",
            ],
            [GRAY_A, YELLOW, WHITE],
        )
        self.play(
            ReplacementTransform(panel_step_1, panel_step_2),
            Create(ring),
            FadeIn(cloud, lag_ratio=0.015),
            FadeIn(sigma_label, shift=UP * 0.08),
            run_time=1.2,
        )

        highlighted_point = np.array([0.96, -0.56])
        noisy_dot = Dot(plane.c2p(*highlighted_point), radius=0.075, color=YELLOW)
        noisy_dot.set_z_index(5)
        noisy_label = Text("P", font_size=23, color=YELLOW).next_to(
            noisy_dot, DOWN + RIGHT, buff=0.07
        )
        displacement = Arrow(
            plane.c2p(*center),
            plane.c2p(*highlighted_point),
            buff=0.08,
            stroke_width=4,
            color=BLUE_B,
            max_tip_length_to_length_ratio=0.18,
        ).set_z_index(4)
        displacement_label = Text("noise", font_size=20, color=BLUE_B).next_to(
            displacement, DOWN, buff=0.1
        )
        reverse_arrow = self._score_arrow(
            plane,
            highlighted_point,
            gaussian_score(highlighted_point, center, sigma=0.75),
            color=YELLOW,
            max_length=0.9,
            stroke_width=7,
        )
        reverse_label = Text("score direction", font_size=22, color=YELLOW).next_to(
            reverse_arrow, UP + RIGHT, buff=0.08
        )

        self.play(FadeIn(noisy_dot), FadeIn(noisy_label), run_time=0.45)
        self.play(Create(displacement), FadeIn(displacement_label), run_time=0.65)
        self.play(GrowArrow(reverse_arrow), FadeIn(reverse_label), run_time=0.85)
        self.wait(0.45)

        panel_step_3 = self._panel_text(
            [
                "For one clean point:",
                "score(P) points back to x0",
                "score = -(P - x0) / sigma^2",
            ],
            [GRAY_A, WHITE, YELLOW],
        )
        self.play(ReplacementTransform(panel_step_2, panel_step_3), run_time=0.75)
        self.wait(0.45)

        self.play(
            FadeOut(displacement),
            FadeOut(displacement_label),
            FadeOut(reverse_arrow),
            FadeOut(reverse_label),
            FadeOut(noisy_dot),
            FadeOut(noisy_label),
            run_time=0.45,
        )

        for sigma, color, label in zip(sigmas[1:], colors[1:], labels[1:], strict=True):
            next_cloud = self._noise_cloud(plane, center + sigma * base_noise, color)
            next_ring = self._coordinate_circle(
                plane, center, radius=min(3.0, 2.0 * sigma)
            )
            next_ring.set_stroke(color, width=3, opacity=0.72).set_z_index(1)
            next_label = Text(label, font_size=23, color=color).to_edge(DOWN, buff=0.34)

            self.play(
                Transform(cloud, next_cloud),
                Transform(ring, next_ring),
                Transform(sigma_label, next_label),
                run_time=0.9,
            )
            self.wait(0.2)

        panel_step_4 = self._panel_text(
            [
                "Train over many sigma values.",
                "Small sigma: local detail",
                "Large sigma: broad coverage",
            ],
            [WHITE, TEAL_A, ORANGE],
        )
        self.play(ReplacementTransform(panel_step_3, panel_step_4), run_time=0.75)
        self.wait(0.45)

        field_sigma = 1.25
        field = self._score_field(plane, center, field_sigma)
        field_caption = Text(
            "A noisy point P teaches an inward score vector.",
            font_size=23,
            color=GRAY_A,
        ).to_edge(DOWN, buff=0.34)
        self.play(
            FadeOut(cloud),
            FadeOut(ring),
            ReplacementTransform(sigma_label, field_caption),
            run_time=0.6,
        )
        self.play(
            LaggedStart(*[GrowArrow(arrow) for arrow in field], lag_ratio=0.012),
            run_time=1.5,
        )

        panel_step_5 = self._panel_text(
            [
                "At fixed sigma:",
                "p_sigma(x | x0) is a Gaussian",
                "score = grad_x log p_sigma",
            ],
            [GRAY_A, WHITE, YELLOW],
        )
        self.play(ReplacementTransform(panel_step_4, panel_step_5), run_time=0.75)
        self.wait(0.45)

        starts = self._reverse_starts()
        moving_dots = VGroup(
            *[
                Dot(plane.c2p(*point), radius=0.048, color=WHITE).set_z_index(6)
                for point in starts
            ]
        )
        paths = VGroup(
            *[
                VMobject(color=YELLOW, stroke_width=2.4)
                .set_opacity(0.58)
                .set_points_smoothly(
                    [plane.c2p(*p) for p in self._reverse_path(point, center)]
                )
                for point in starts
            ]
        )
        reverse_caption = Text(
            "Reverse sampling follows the learned fields: high sigma back to data.",
            font_size=22,
            color=WHITE,
        ).to_edge(DOWN, buff=0.34)

        self.play(
            FadeOut(field_caption),
            FadeIn(reverse_caption, shift=UP * 0.08),
            FadeIn(moving_dots, lag_ratio=0.03),
            run_time=0.75,
        )
        self.play(Create(paths), run_time=0.9)
        self.play(
            LaggedStart(
                *[
                    MoveAlongPath(dot, path)
                    for dot, path in zip(moving_dots, paths, strict=True)
                ],
                lag_ratio=0.025,
            ),
            run_time=3.4,
        )

        final_panel = self._panel_text(
            [
                "VE recipe:",
                "add noise at many sigma levels",
                "learn the reverse score directions",
            ],
            [TEAL_A, WHITE, YELLOW],
        )
        self.play(
            ReplacementTransform(panel_step_5, final_panel),
            field.animate.set_opacity(0.34),
            paths.animate.set_opacity(0.35),
            moving_dots.animate.set_color(YELLOW),
            run_time=0.85,
        )
        self.wait(1.4)

    def _make_plane(self) -> NumberPlane:
        plane = NumberPlane(
            x_range=[X_RANGE[0], X_RANGE[1], 1],
            y_range=[Y_RANGE[0], Y_RANGE[1], 1],
            x_length=5.9,
            y_length=5.9,
            background_line_style={
                "stroke_color": "#334155",
                "stroke_width": 1,
                "stroke_opacity": 0.32,
            },
            axis_config={
                "stroke_color": "#94a3b8",
                "stroke_width": 1.5,
                "stroke_opacity": 0.72,
            },
        )
        plane.to_edge(DOWN, buff=0.45)
        plane.shift(LEFT * 2.15)
        return plane

    def _make_panel(self) -> VGroup:
        panel = RoundedRectangle(
            width=5.25,
            height=3.8,
            corner_radius=0.18,
            stroke_color="#334155",
            stroke_width=1.5,
            fill_color="#111827",
            fill_opacity=0.74,
        )
        panel.to_edge(RIGHT, buff=0.35).shift(DOWN * 0.05)
        label = Text("VE construction", font_size=22, color=GRAY_A)
        label.next_to(panel.get_top(), DOWN, buff=0.22)
        return VGroup(panel, label)

    def _panel_text(self, lines: list[str], colors: list[ManimColor]) -> VGroup:
        texts = VGroup(
            *[
                Text(line, font_size=21 if index != 1 else 24, color=color)
                for index, (line, color) in enumerate(zip(lines, colors, strict=True))
            ]
        )
        texts.arrange(DOWN, buff=0.24, aligned_edge=LEFT)
        texts.move_to(RIGHT * 3.75 + DOWN * 0.05)
        return texts.set_z_index(5)

    def _data_marker(self, plane: NumberPlane, center: np.ndarray) -> VGroup:
        dot = Dot(plane.c2p(*center), radius=0.09, color=TEAL_A).set_z_index(6)
        ring = Circle(radius=0.2, color=TEAL_A, stroke_width=3).move_to(dot)
        label = Text("x0", font_size=23, color=TEAL_A).next_to(dot, UP + RIGHT, buff=0.08)
        return VGroup(ring, dot, label)

    def _noise_cloud(
        self, plane: NumberPlane, points: np.ndarray, color: ManimColor
    ) -> VGroup:
        dots = []
        for point in points:
            clipped = np.clip(point, [X_RANGE[0] + 0.1, Y_RANGE[0] + 0.1], [X_RANGE[1] - 0.1, Y_RANGE[1] - 0.1])
            dot = Dot(plane.c2p(*clipped), radius=0.028, color=color)
            dot.set_opacity(0.62)
            dot.set_z_index(3)
            dots.append(dot)
        return VGroup(*dots)

    def _coordinate_circle(
        self,
        plane: NumberPlane,
        center: np.ndarray,
        radius: float,
        samples: int = 180,
    ) -> VMobject:
        angles = np.linspace(0.0, TAU, samples)
        points = [
            plane.c2p(center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle))
            for angle in angles
        ]
        return VMobject().set_points_smoothly(points).set_fill(opacity=0.0)

    def _score_field(
        self, plane: NumberPlane, center: np.ndarray, sigma: float
    ) -> VGroup:
        arrows = []
        xs = np.linspace(-2.8, 2.8, 9)
        ys = np.linspace(-2.8, 2.8, 9)
        for x in xs:
            for y in ys:
                point = np.array([x, y], dtype=float)
                if np.linalg.norm(point - center) < 0.18:
                    continue
                arrows.append(
                    self._score_arrow(
                        plane,
                        point,
                        gaussian_score(point, center, sigma),
                        color=YELLOW,
                        max_length=0.38,
                        stroke_width=3.2,
                    ).set_opacity(0.72)
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
            plane.c2p(*point),
            plane.c2p(*end),
            buff=0.0,
            color=color,
            stroke_width=stroke_width,
            max_tip_length_to_length_ratio=0.28,
        ).set_z_index(4)

    def _reverse_starts(self) -> list[np.ndarray]:
        angles = np.linspace(0.0, TAU, 12, endpoint=False)
        radii = np.array([2.65, 2.35, 2.85, 2.15, 2.55, 2.75] * 2)
        return [
            np.array([r * np.cos(angle), r * np.sin(angle)], dtype=float)
            for r, angle in zip(radii, angles, strict=True)
        ]

    def _reverse_path(
        self, start: np.ndarray, center: np.ndarray, steps: int = 36
    ) -> list[np.ndarray]:
        point = np.array(start, dtype=float)
        path = [point.copy()]
        for _ in range(steps):
            point = center + 0.88 * (point - center)
            path.append(point.copy())
        return path
