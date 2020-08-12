import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

import analysis.benchmark as bench


def linear_regression(df):
    x = df.iloc[:, 0].values.reshape(-1, 1)
    y = df.iloc[:, 1].values.reshape(-1, 1)
    linear_regressor = LinearRegression()
    linear_regressor.fit(x, y)
    print("R^2:", linear_regressor.score(x, y))
    y_pred = linear_regressor.predict(x)
    plt.scatter(x, y)
    plt.xlabel(f"sample {df.columns[0]}")
    plt.ylabel(f"next iteration benchmark {df.columns[1]}")
    plt.plot(x, y_pred, color="r")
    plt.show()

    x_sm = sm.add_constant(x)
    est = sm.OLS(y, x_sm).fit()
    print(est.summary())


def make_conf_histogram(results, filename):
    num_rows = len(results)
    fig, axs = plt.subplots(num_rows, 3)
    plt.subplots_adjust(hspace=0.35)

    graphs = ["hit", "miss", "all"]
    all_data = dict()
    for name in graphs:
        all_data[name] = list()

    colors = ["lightgreen", "red"]
    for i, res in enumerate(results):
        hit_miss = [[row["conf"] for row in data] for data in res.hits_misses()]

        axs[i][0].hist(hit_miss[0], bins=15, color=colors[0], range=(0, 1))
        axs[i][1].hist(hit_miss[1], bins=15, color=colors[1], range=(0, 1))
        axs[i][2].hist(hit_miss, bins=15, color=colors, stacked=True, range=(0, 1))

        if res.name == "All":
            acc = round(bench.mean_accuracy(results[:-1]), 3)
            prec = round(bench.mean_precision(results[:-1]), 3)
        else:
            acc = round(res.accuracy(), 3)
            prec = round(res.precision(), 3)

        title = f"Class: {res.name} (acc={acc}, " + f"prec={prec}, n={res.pop})"
        axs[i][1].set_title(title)

    fig.set_figheight(2.5 * num_rows)
    fig.set_figwidth(10)
    fig.savefig(filename, bbox_inches="tight")


def plot_multiline(xy_pairs, xlab=str(), ylab=str(), vert_lines=None):
    fig, ax = plt.subplots()
    plt.xlabel(xlab)
    plt.ylabel(ylab)

    lines = list()
    for (x_coords, y_coords, label) in xy_pairs:
        line = ax.plot(x_coords, y_coords, label=label)
        lines.append(line)

    leg = ax.legend()
    line_dict = dict()
    for leg_line, orig_line in zip(leg.get_lines(), lines):
        leg_line.set_picker(10)
        line_dict[leg_line] = orig_line

    def onpick(event):
        leg_line = event.artist
        orig_line = line_dict[leg_line][0]
        vis = not orig_line.get_visible()
        orig_line.set_visible(vis)
        if vis:
            leg_line.set_alpha(1.0)
        else:
            leg_line.set_alpha(0.2)
        fig.canvas.draw()

    fig.canvas.mpl_connect("pick_event", onpick)

    if vert_lines is not None:
        for x in vert_lines:
            plt.axvline(x=x, color="black", linestyle="dashed")
    plt.show()
