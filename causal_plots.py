"""
EconoCausal - Interactive Plotly Visualizations Engine
Week 2 Module: Production-Grade Plotly Figures for Causal Uplift & Qini Analysis.
"""

from typing import Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from causal_eval import QiniResult, UpliftResult


def create_qini_figure(
    qini_result: QiniResult,
    title: str = "Cumulative Qini Gain Curve (Model vs. Random Baseline)",
) -> go.Figure:
    """
    Renders an interactive Plotly figure for the cumulative Qini curve, random baseline,
    and theoretical oracle curve (if available).
    """
    df = qini_result.curve_df
    fig = go.Figure()

    # 1. Random Baseline (dashed grey)
    fig.add_trace(
        go.Scatter(
            x=df["population_percent"],
            y=df["random_baseline"],
            mode="lines",
            name="Random Rollout Baseline",
            line=dict(color="#7f8c8d", width=2, dash="dash"),
            hovertemplate=(
                "<b>Random Baseline</b><br>"
                "Targeted: %{x:.1f}%<br>"
                "Baseline Gain: %{y:.2f} conversions<extra></extra>"
            ),
        )
    )

    # 2. Theoretical Oracle (if present)
    if "oracle_qini" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["population_percent"],
                y=df["oracle_qini"],
                mode="lines",
                name="Theoretical Oracle (Ground Truth DGP)",
                line=dict(color="#27ae60", width=2, dash="dot"),
                hovertemplate=(
                    "<b>Theoretical Oracle</b><br>"
                    "Targeted: %{x:.1f}%<br>"
                    "Oracle Gain: %{y:.2f} conversions<extra></extra>"
                ),
            )
        )

    # 3. Trained Causal Model (solid blue with shaded area to baseline)
    fig.add_trace(
        go.Scatter(
            x=df["population_percent"],
            y=df["model_qini"],
            mode="lines",
            name="Double ML Causal Ranking",
            line=dict(color="#2980b9", width=3),
            fill="tonexty",
            fillcolor="rgba(41, 128, 185, 0.12)",
            hovertemplate=(
                "<b>Causal DML Model</b><br>"
                "Targeted: %{x:.1f}% (%{customdata:,} customers)<br>"
                "Cumulative Qini Gain: %{y:.2f} incremental conversions<extra></extra>"
            ),
            customdata=df["n_targeted"],
        )
    )

    # Score annotation banner
    convention_label = (
        "Radcliffe Global Ratio" if qini_result.convention == "radcliffe_global"
        else ("Radcliffe Local Ratio" if qini_result.convention == "radcliffe_local" else "IPW Causal Transform")
    )
    score_text = (
        f"<b>Net Qini Score (AUQC - AURC): {qini_result.qini_score:+.2f}</b><br>"
        f"Model AUQC: {qini_result.model_auc:.2f} | Random AURC: {qini_result.random_auc:.2f}<br>"
        f"Convention: {convention_label}"
    )
    if qini_result.qini_ratio is not None:
        score_text += f"<br>Normalized vs Oracle: {qini_result.qini_ratio:.1%}"

    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.04, y=0.95,
        text=score_text,
        showarrow=False,
        bgcolor="rgba(255, 255, 255, 0.88)",
        bordercolor="#2980b9",
        borderwidth=1.5,
        borderpad=8,
        font=dict(size=11, family="sans-serif", color="#2c3e50"),
        align="left",
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=15, weight="bold", family="sans-serif")),
        xaxis=dict(
            title="Percentage of Evaluation Population Targeted (%)",
            ticksuffix="%",
            range=[0, 100],
            gridcolor="#ecf0f1",
            zerolinecolor="#bdc3c7",
        ),
        yaxis=dict(
            title="Cumulative Incremental Conversions (Qini Gain)",
            gridcolor="#ecf0f1",
            zerolinecolor="#bdc3c7",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255, 255, 255, 0.7)",
        ),
        hovermode="x unified",
        template="plotly_white",
        margin=dict(l=60, r=40, t=60, b=80),
        height=480,
    )

    return fig


def create_uplift_figure(
    uplift_result: UpliftResult,
    title: str = "Cumulative Uplift Curve (Incremental Conversion Rate)",
) -> go.Figure:
    """
    Renders an interactive Plotly figure for cumulative average treatment effect
    as the targeted fraction increases from 0% to 100%.
    """
    df = uplift_result.curve_df
    fig = go.Figure()

    # 1. Random Targeting Baseline (Sample ATE horizontal line)
    fig.add_trace(
        go.Scatter(
            x=df["population_percent"],
            y=df["random_baseline"] * 100.0,
            mode="lines",
            name=f"Random Targeting Baseline (ATE: {uplift_result.sample_ate:+.2%})",
            line=dict(color="#e74c3c", width=2, dash="dash"),
            hovertemplate=(
                "<b>Random Rollout ATE</b><br>"
                "Targeted: %{x:.1f}%<br>"
                "Average Lift: %{y:.2f}%<extra></extra>"
            ),
        )
    )

    # 2. Cumulative Uplift Curve
    fig.add_trace(
        go.Scatter(
            x=df["population_percent"],
            y=df["cumulative_uplift"] * 100.0,
            mode="lines",
            name="Causal DML Ranked Uplift",
            line=dict(color="#27ae60", width=3),
            hovertemplate=(
                "<b>Trained Causal Model</b><br>"
                "Targeted: %{x:.1f}% (%{customdata[0]:,} customers)<br>"
                "Cumulative Lift: %{y:.2f}%<br>"
                "Incremental Conversions: +%{customdata[1]:.1f}<extra></extra>"
            ),
            customdata=np.column_stack([df["n_targeted"], df["incremental_gain"]]),
        )
    )

    # Marker for optimal fraction
    opt_pct = uplift_result.best_targeting_fraction * 100.0
    fig.add_vline(
        x=opt_pct,
        line_width=1.5,
        line_dash="dot",
        line_color="#2980b9",
        annotation_text=f"Max Uplift Peak: {opt_pct:.0f}%",
        annotation_position="top right",
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=15, weight="bold", family="sans-serif")),
        xaxis=dict(
            title="Percentage of Evaluation Population Targeted (%)",
            ticksuffix="%",
            range=[0, 100],
            gridcolor="#ecf0f1",
        ),
        yaxis=dict(
            title="Cumulative Incremental Conversion Rate (%)",
            ticksuffix="%",
            gridcolor="#ecf0f1",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
        ),
        hovermode="x unified",
        template="plotly_white",
        margin=dict(l=60, r=40, t=60, b=80),
        height=480,
    )

    return fig


def create_decile_figure(
    decile_df: pd.DataFrame,
    title: str = "Targeting Decile Analysis: Observed Uplift & Model Predicted CATE",
) -> go.Figure:
    """
    Renders a bar + line combination chart showing incremental conversion uplift
    and predicted effect across customer deciles (Decile 1 = top responders).
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 1. Bar chart: Observed empirical uplift per decile
    bar_colors = [
        "#27ae60" if u >= 0 else "#e74c3c" for u in decile_df["observed_uplift"]
    ]
    fig.add_trace(
        go.Bar(
            x=decile_df["decile"],
            y=decile_df["observed_uplift"] * 100.0,
            name="Observed Empirical Lift (Treated - Control)",
            marker_color=bar_colors,
            opacity=0.85,
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Observed Lift: %{y:.2f}%<br>"
                "Treated Rate: %{customdata[0]:.1%}<br>"
                "Control Rate: %{customdata[1]:.1%}<br>"
                "Incremental Buyers: %{customdata[2]:.1f}<extra></extra>"
            ),
            customdata=np.column_stack([
                decile_df["treated_conversion_rate"],
                decile_df["control_conversion_rate"],
                decile_df["incremental_conversions"],
            ]),
        ),
        secondary_y=False,
    )

    # 2. Line chart: Model mean predicted effect
    fig.add_trace(
        go.Scatter(
            x=decile_df["decile"],
            y=decile_df["mean_predicted_effect"] * 100.0,
            name="Model Mean Predicted CATE (Lift %)",
            mode="lines+markers",
            line=dict(color="#2980b9", width=3),
            marker=dict(size=8, symbol="diamond"),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Predicted CATE: %{y:.2f}%<br>"
                "Range: [%{customdata[0]:.2f}%, %{customdata[1]:.2f}%]<extra></extra>"
            ),
            customdata=np.column_stack([
                decile_df["min_predicted_effect"] * 100.0,
                decile_df["max_predicted_effect"] * 100.0,
            ]),
        ),
        secondary_y=False,
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=15, weight="bold")),
        xaxis=dict(title="Customer Decile (Ranked by Predicted Treatment Effect)"),
        yaxis=dict(
            title="Incremental Lift (% pts)",
            ticksuffix="%",
            gridcolor="#ecf0f1",
            zerolinecolor="#2c3e50",
            zerolinewidth=1.5,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.28,
            xanchor="center",
            x=0.5,
        ),
        template="plotly_white",
        margin=dict(l=60, r=40, t=60, b=85),
        height=480,
    )

    return fig


def create_effect_distribution_figure(
    pred_effects: np.ndarray,
    mean_val: float,
    median_val: float,
    title: str = "Distribution of Out-of-Sample Individual Treatment Effects (ITE / CATE)",
) -> go.Figure:
    """
    Renders a clean histogram and summary markers of customer-level treatment effect estimates.
    """
    effects_pct = pred_effects * 100.0
    mean_pct = mean_val * 100.0
    med_pct = median_val * 100.0

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=effects_pct,
            nbinsx=40,
            marker_color="#3498db",
            opacity=0.75,
            name="Customer ITE Density",
            hovertemplate="ITE Bin: %{x:.2f}%<br>Customer Count: %{y}<extra></extra>",
        )
    )

    fig.add_vline(
        x=mean_pct,
        line_color="#e74c3c",
        line_width=2.5,
        line_dash="solid",
        annotation_text=f"Mean ITE: {mean_pct:+.2f}%",
        annotation_position="top right",
    )

    fig.add_vline(
        x=med_pct,
        line_color="#8e44ad",
        line_width=2.0,
        line_dash="dash",
        annotation_text=f"Median ITE: {med_pct:+.2f}%",
        annotation_position="top left",
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=15, weight="bold")),
        xaxis=dict(
            title="Estimated Treatment Effect (Percentage Point Conversion Lift)",
            ticksuffix="%",
            gridcolor="#ecf0f1",
        ),
        yaxis=dict(title="Number of Evaluation Customers", gridcolor="#ecf0f1"),
        template="plotly_white",
        margin=dict(l=60, r=40, t=60, b=60),
        height=420,
    )

    return fig


def create_propensity_overlap_figure(
    propensities: np.ndarray,
    treatment: np.ndarray,
    title: str = "Observational Positivity & Overlap: Propensity Score Distribution",
) -> go.Figure:
    """
    Visualizes the propensity score distribution across treated and control observations
    to diagnose overlap / common support in observational data.
    """
    fig = go.Figure()

    treated_props = propensities[treatment == 1]
    control_props = propensities[treatment == 0]

    fig.add_trace(
        go.Histogram(
            x=treated_props,
            name=f"Treated (N={len(treated_props):,})",
            nbinsx=35,
            marker_color="#e67e22",
            opacity=0.6,
            hovertemplate="Treated Propensity: %{x:.3f}<br>Count: %{y}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Histogram(
            x=control_props,
            name=f"Control (N={len(control_props):,})",
            nbinsx=35,
            marker_color="#3498db",
            opacity=0.6,
            hovertemplate="Control Propensity: %{x:.3f}<br>Count: %{y}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=15, weight="bold")),
        barmode="overlay",
        xaxis=dict(title="Estimated Propensity Score e(W) = P(T=1|W)", range=[0, 1], gridcolor="#ecf0f1"),
        yaxis=dict(title="Observation Count", gridcolor="#ecf0f1"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        template="plotly_white",
        margin=dict(l=60, r=40, t=60, b=70),
        height=420,
    )

    return fig
