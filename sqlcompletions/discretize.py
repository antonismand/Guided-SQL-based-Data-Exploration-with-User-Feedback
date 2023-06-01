import json

import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib
from sqlcompletions.sqlparser import schema


def get_num_of_digits(num):
    return str(num)[::-1].find(".")


def get_num_of_digits_in_df(df, col):
    num = df[col].iloc[0]
    return get_num_of_digits(num)


def save_stats(attributes, df):

    table = attributes[0].split(".")[0]
    columns = [col.split(".")[1] for col in attributes]

    for col in columns:
        stats = {
            "freedman_diaconis": {},
            "digits": get_num_of_digits_in_df(df, col),
            "min": df[col].min(),
            "max": df[col].max(),
        }

        (
            stats["freedman_diaconis"]["bin_width"],
            stats["freedman_diaconis"]["bin_count"],
        ) = freedman_diaconis(df[col].to_numpy(), plot=False)

        print(f"{table}.{col}: {list(stats.items())} ")

        with open(
            f"data/processed/bins/{table}-{col}.json",
            "w+",
            encoding="utf-8",
        ) as f:
            json.dump(stats, f, ensure_ascii=False, indent=4)


def plot_hist(data, bin_count, bin_width):
    plt.hist(data, bin_count)
    plt.title(f"bins = {bin_count}, width = {bin_width}")
    plt.show()


def freedman_diaconis(data, plot=False):
    n = data.size

    q3, q1 = np.percentile(data, [75, 25])
    iqr = q3 - q1

    bin_width = (2 * iqr) / (n ** (1 / 3))
    bin_count = int(np.ceil((data.max() - data.min()) / bin_width))

    bin_width = round(bin_width, 4)

    if plot:
        plot_hist(data, bin_count, bin_width)

    return bin_width, bin_count


def sturge_rule(data, plot=False):
    n = data.size
    bin_count = int(np.ceil(np.log2(n)) + 1)
    range = data.max() - data.min()
    bin_width = range / bin_count
    bin_width = round(bin_width, 4)

    if plot:
        plot_hist(data, bin_count, bin_width)

    return bin_width, bin_count


def mad(data):
    median = np.median(data)
    diff = np.abs(data - median)
    mad = np.median(diff)
    return mad


def calculate_bounds(data, z_thresh=3.5):
    MAD = mad(data)
    median = np.median(data)
    const = z_thresh * MAD / 0.6745
    return (median - const, median + const)


def avg_bin_width_for_attr(attribute):
    db = schema()
    widths = np.array([])

    for line in open("data/processed/where_log_th500.json"):
        row = json.loads(line)[0]
        for attr, val in row.items():
            for cond in val:
                if (
                    cond[0] == "between"
                    and not isinstance(cond[1], dict)
                    and not isinstance(cond[2], dict)
                    and not isinstance(cond[2], str)
                ):
                    attr_name = db.get_column_name(int(attr))
                    if attr_name == attribute:
                        tmp = abs(cond[2] - cond[1])
                        widths = np.append(widths, tmp)

    range = calculate_bounds(widths)
    filtered = np.array([x for x in widths if range[0] <= x <= range[1]])
    return round(np.mean(filtered), 2)


def plot_bin_width_for_attr(attribute, range=None):
    vals = np.array([])
    db = schema()

    for line in open("data/processed/where_log_th500.json"):
        row = json.loads(line)[0]
        for attr, val in row.items():
            for cond in val:
                if (
                    cond[0] == "between"
                    and not isinstance(cond[1], dict)
                    and not isinstance(cond[2], dict)
                    and not isinstance(cond[2], str)
                ):
                    attr_name = db.get_column_name(int(attr))
                    if attr_name == attribute:
                        tmp = abs(cond[2] - cond[1])
                        vals = np.append(vals, tmp)

    if range == "auto":
        range = calculate_bounds(vals)

    plt.hist(vals, bins=100, range=range)
    plt.title("Bin width (LOG) for " + attribute)
    plt.show()


def plot_hist_for_attr(attribute, range=None, step=0.01, bins=None):

    db = schema()
    vals = []

    min_number = np.inf
    max_number = -np.inf

    for line in open("data/processed/where_log_th500.json"):
        row = json.loads(line)[0]
        for attr, val in row.items():
            for cond in val:
                if (
                    cond[0] == "between"
                    and not isinstance(cond[1], dict)
                    and not isinstance(cond[2], dict)
                    and not isinstance(cond[2], str)
                ):
                    attr_name = db.get_column_name(int(attr))
                    if attr_name == attribute:
                        if range is None or (
                            range[0] <= cond[1] and range[1] >= cond[2]
                        ):
                            tmp = list(np.arange(cond[1], cond[2], step))
                            vals.extend(tmp)

                            if range is None:
                                min_number = np.min([min_number, cond[1]])
                                max_number = np.max([max_number, cond[2]])

    if bins is None:
        if range is None:
            bins = abs(max_number - min_number) / step
        else:
            bins = abs(range[1] - range[0]) / step
        bins = int(np.round(bins))

    _, ax = plt.subplots()
    ax.hist(vals, range=range, bins=bins)
    ax.set(title="Users' selections (LOG) for " + attribute)

    # plt.show()
    tikzplotlib.clean_figure()
    tikzplotlib.save(
        f"plots/log-{attribute}.tex",
        extra_axis_parameters=["scaled x ticks=false", "scaled y ticks=false"],
        axis_height="\\figH",
        axis_width="\\figW",
    )
