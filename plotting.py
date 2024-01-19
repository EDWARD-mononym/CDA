import numpy as np
import matplotlib.pyplot as plt



def make_radar_chart(title, methods, stats, attribute_labels, plot_markers):
    labels = np.array(attribute_labels)
    
    num_vars = len(labels)
    
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    for method_stats, name in zip(stats, methods):
        method_stats += method_stats[:1]
        ax.plot(angles, method_stats, label=name, linewidth=2, marker='o')
        ax.fill(angles, method_stats, alpha=0.25)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels)

    for label, angle in zip(ax.get_xticklabels(), angles):
        if angle in (0, np.pi):
            label.set_horizontalalignment('center')
        elif 0 < angle < np.pi:
            label.set_horizontalalignment('left')
        else:
            label.set_horizontalalignment('right')

    ax.set_ylim(80, 100)  # Adjust the y-scale to the data range
    ax.spines['polar'].set_visible(False)
    ax.set_rticks(plot_markers)
    ax.set_rlabel_position(180 / num_vars)

    plt.title(title, size=20, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    plt.show()

# Example data
attribute_labels = ["Rotate", "Load", "Radial"]
plot_markers = [20, 40, 60, 80, 100]
methods = ['CDAN', 'DANN', 'Deep Coral', 'MCD', 'COSDA']

# Accuracy stats for each method as percentages (Rotate, Load, Radial)
stats = [
    [85.06, 86.40, 87.27],  # CDAN
    [90.49, 84.98, 89.44],  # DANN
    [93.05, 87.98, 91.86],  # Deep Coral
    [95.55, 87.37, 90.12],  # MCD
    [99.42, 86.72, 95.58]   # COSDA
]

# Continue with the same 'attribute_labels', 'plot_markers', 'methods', and 'stats' from the previous example
make_radar_chart("Comparison of Methods", methods, stats, attribute_labels, plot_markers)
