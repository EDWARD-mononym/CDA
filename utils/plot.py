import matplotlib.pyplot as plt

def save_plot(R, scenario, save_file):
    plt.clf()

    first_target = scenario[1]
    x = scenario[1:]
    y = R.loc[first_target].tolist()[1:]

    plt.plot(y)
    plt.xticks(ticks=range(len(x)), labels=x)

    plt.ylim(0, 1)

    plt.title(f"Acc on {first_target} for scenario: {scenario}")
    plt.xlabel("Models")
    plt.ylabel("Acc")

    # Save PNG file
    plt.savefig(save_file)