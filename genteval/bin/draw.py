from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Data provided by the user.
# The `names` list is for plot labels.
# `mape_fidelity` corresponds to the y-axis data for the MAPE plot.
# `cost_per_million_spans` corresponds to the x-axis data.
names = [
    "GenT 1min",
    "GenT 1min",
    "GenT 1min",
    "GenT 5min",
    "GenT 5min",
    "GenT 5min",
    "GenT 10min",
    "GenT 10min",
    "GenT 10min",
    "1:5",
    "1:10",
    "1:20",
    "1:50",
    "1:100",
    "1:200",
    "1:400",
]
mape_fidelity = [
    77.9203,
    83.6981,
    86.3573,
    85.3659,
    81.1399,
    85.6173,
    78.034,
    81.5327,
    81.8724,
    93.3925,
    90.0514,
    85.7399,
    80.4434,
    76.7223,
    71.5568,
    67.3589,
]
cos_fidelity = [
    93.0012,
    92.6428,
    95.3917,
    93.8652,
    91.7925,
    93.1178,
    92.2587,
    92.8246,
    93.7162,
    95.7172,
    93.7642,
    89.3188,
    87.4752,
    85.2549,
    83.1646,
    74.7693,
]
cost_per_million_spans = [
    0.06728186647,
    0.07992974856,
    0.06727664625,
    0.1328741667,
    0.1811816844,
    0.1355887104,
    0.1104541818,
    0.1170866497,
    0.1396778491,
    0.0004211112545,
    0.0002101919145,
    0.0001025013107,
    0.0000417333021,
    0.00002156001651,
    0.00001154392669,
    0.000006515202462,
]
transmission_cost_per_million_spans = [
    0.0000548780515,
    0.00005385381691,
    0.00005674405609,
    0.00001458986899,
    0.00001533189168,
    0.0000151105013,
    0.000008325251241,
    0.000008089263699,
    0.000008412834247,
    0.0004211112545,
    0.0002101919145,
    0.0001025013107,
    0.0000417333021,
    0.00002156001651,
    0.00001154392669,
    0.000006515202462,
]


def main():
    # Helper to render and save a plot for a given metric
    def draw_and_save(
        x_values,
        x_title,
        y_values,
        y_title: str,
        plot_title: str,
        out_fname: str,
        dpi: int = 300,
    ):
        # --- Data Preparation and Analysis ---
        # Find the GenT data points to calculate the standard deviation.
        gent_1_y_data = [y for i, y in enumerate(y_values) if names[i] == "GenT 1min"]
        gent_1_cost_data = [
            cpm for i, cpm in enumerate(x_values) if names[i] == "GenT 1min"
        ]

        gent_5_y_data = [y for i, y in enumerate(y_values) if names[i] == "GenT 5min"]
        gent_5_cost_data = [
            cpm for i, cpm in enumerate(x_values) if names[i] == "GenT 5min"
        ]

        gent_10_y_data = [y for i, y in enumerate(y_values) if names[i] == "GenT 10min"]
        gent_10_cost_data = [
            cpm for i, cpm in enumerate(x_values) if names[i] == "GenT 10min"
        ]

        # Calculate means and std for GenT methods
        gent_1_y_mean = np.mean(gent_1_y_data)
        gent_1_cost_mean = np.mean(gent_1_cost_data)
        gent_1_y_std = np.std(gent_1_y_data)
        gent_1_cost_std = np.std(gent_1_cost_data)

        gent_5_y_mean = np.mean(gent_5_y_data)
        gent_5_cost_mean = np.mean(gent_5_cost_data)
        gent_5_y_std = np.std(gent_5_y_data)
        gent_5_cost_std = np.std(gent_5_cost_data)

        gent_10_y_mean = np.mean(gent_10_y_data)
        gent_10_cost_mean = np.mean(gent_10_cost_data)
        gent_10_y_std = np.std(gent_10_y_data)
        gent_10_cost_std = np.std(gent_10_cost_data)

        # Separate the non-GenT data for plotting.
        other_y_data = [y for i, y in enumerate(y_values) if "GenT" not in names[i]]
        other_cost_data = [
            cpm for i, cpm in enumerate(x_values) if "GenT" not in names[i]
        ]
        other_names = [name for name in names if "GenT" not in name]

        # --- Plotting the Data ---
        plt.figure(figsize=(12, 8))

        # Set a professional style
        plt.style.use("default")
        plt.rcParams["font.size"] = 11
        plt.rcParams["axes.labelsize"] = 12
        plt.rcParams["axes.titlesize"] = 14

        # Plot the GenT 1min data points with error bars
        plt.errorbar(
            x=gent_1_cost_mean,
            y=gent_1_y_mean,
            xerr=gent_1_cost_std,
            yerr=gent_1_y_std,
            marker="o",
            markersize=12,
            label="GenT 1min (mean ± std)",
            color="red",
            alpha=0.8,
            capsize=5,
            capthick=2,
            linewidth=2,
        )

        # Plot individual GenT 1min points
        plt.scatter(
            x=gent_1_cost_data,
            y=gent_1_y_data,
            marker="o",
            s=60,
            color="red",
            alpha=0.4,
            edgecolors="darkred",
            linewidth=1,
        )

        # Plot the GenT 5min data points with error bars
        plt.errorbar(
            x=gent_5_cost_mean,
            y=gent_5_y_mean,
            xerr=gent_5_cost_std,
            yerr=gent_5_y_std,
            marker="s",
            markersize=12,
            label="GenT 5min (mean ± std)",
            color="darkred",
            alpha=0.8,
            capsize=5,
            capthick=2,
            linewidth=2,
        )

        # Plot individual GenT 5min points
        plt.scatter(
            x=gent_5_cost_data,
            y=gent_5_y_data,
            marker="s",
            s=60,
            color="darkred",
            alpha=0.4,
            edgecolors="maroon",
            linewidth=1,
        )

        # Plot the GenT 10min data points with error bars
        plt.errorbar(
            x=gent_10_cost_mean,
            y=gent_10_y_mean,
            xerr=gent_10_cost_std,
            yerr=gent_10_y_std,
            marker="^",
            markersize=12,
            label="GenT 10min (mean ± std)",
            color="blue",
            alpha=0.8,
            capsize=5,
            capthick=2,
            linewidth=2,
        )

        # Plot individual GenT 10min points
        plt.scatter(
            x=gent_10_cost_data,
            y=gent_10_y_data,
            marker="^",
            s=60,
            color="blue",
            alpha=0.4,
            edgecolors="darkblue",
            linewidth=1,
        )

        # Plot the other data points as a scatter plot with improved styling
        plt.scatter(
            x=other_cost_data,
            y=other_y_data,
            marker="D",  # Diamond markers for better distinction
            s=120,
            label="Head Sampling",
            alpha=0.8,
            edgecolors="black",
            linewidth=1,
        )

        # Add improved text labels for the other data points
        for i, txt in enumerate(other_names):
            plt.annotate(
                txt,
                (other_cost_data[i], other_y_data[i]),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.7},
                arrowprops={"arrowstyle": "->", "color": "gray", "alpha": 0.5},
            )

        # Add annotations for GenT methods (only once, at their mean positions)
        plt.annotate(
            "GenT 1min",
            (gent_1_cost_mean, gent_1_y_mean),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.7},
            arrowprops={"arrowstyle": "->", "color": "gray", "alpha": 0.5},
        )

        plt.annotate(
            "GenT 5min",
            (gent_5_cost_mean, gent_5_y_mean),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.7},
            arrowprops={"arrowstyle": "->", "color": "gray", "alpha": 0.5},
        )

        plt.annotate(
            "GenT 10min",
            (gent_10_cost_mean, gent_10_y_mean),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.7},
            arrowprops={"arrowstyle": "->", "color": "gray", "alpha": 0.5},
        )

        # --- Enhanced Plot Customization ---
        plt.title(plot_title, fontsize=18, fontweight="bold", pad=20)
        plt.xlabel(x_title, fontsize=14, fontweight="bold")
        plt.ylabel(y_title, fontsize=14, fontweight="bold")

        plt.xscale("log")

        # Improve grid
        plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
        plt.grid(True, which="minor", linestyle=":", linewidth=0.3, alpha=0.5)

        # Enhanced legend
        plt.legend(loc="best", frameon=True, fancybox=True, shadow=True, fontsize=11)

        # Set axis limits for better visualization
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)
        plt.xlim(x_min * 0.5, x_max * 2)
        plt.ylim(y_min - 2, y_max + 2)

        # Add a subtle background color
        plt.gca().set_facecolor("#f8f9fa")

        # Layout and save
        plt.tight_layout()

        out_path = Path("output/visualizations") / out_fname
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            plt.savefig(
                out_path,
                dpi=dpi,
                bbox_inches="tight",
                format="png",
                facecolor="white",
                edgecolor="none",
            )
            print(f"Saved improved figure to: {out_path.resolve()}")
        except (OSError, ValueError) as e:
            print(f"Failed to save figure to {out_path}: {e}")
        finally:
            plt.close()

    # Always render and save both plots
    draw_and_save(
        x_values=cost_per_million_spans,
        x_title="Total Cost per Million Spans (log scale)",
        y_values=mape_fidelity,
        y_title="MAPE Fidelity (%)",
        plot_title="MAPE Fidelity vs. Total Cost per Million Spans",
        out_fname="mape_vs_total.png",
    )
    draw_and_save(
        x_values=cost_per_million_spans,
        x_title="Total Cost per Million Spans (log scale)",
        y_values=cos_fidelity,
        y_title="Cosine Similarity Fidelity (%)",
        plot_title="Cosine Similarity Fidelity vs. Total Cost per Million Spans",
        out_fname="cos_vs_total.png",
    )
    draw_and_save(
        x_values=transmission_cost_per_million_spans,
        x_title="Transmission Cost per Million Spans (log scale)",
        y_values=mape_fidelity,
        y_title="MAPE Fidelity (%)",
        plot_title="MAPE Fidelity vs. Total Cost per Million Spans",
        out_fname="mape_vs_transmission.png",
    )
    draw_and_save(
        x_values=transmission_cost_per_million_spans,
        x_title="Transmission Cost per Million Spans (log scale)",
        y_values=cos_fidelity,
        y_title="Cosine Similarity Fidelity (%)",
        plot_title="Cosine Similarity Fidelity vs. Total Cost per Million Spans",
        out_fname="cos_vs_transmission.png",
    )
    # No interactive show; plots are saved to disk.


if __name__ == "__main__":
    main()
