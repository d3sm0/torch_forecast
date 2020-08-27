import matplotlib.pyplot as plt


def plot_output(y, y_hat):
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(y, label="y")
    ax.plot(y_hat, label="y_hat")
    ax.set_xlabel("time")
    ax.set_ylabel("y")
    plt.tight_layout()
    plt.legend()
    return fig
