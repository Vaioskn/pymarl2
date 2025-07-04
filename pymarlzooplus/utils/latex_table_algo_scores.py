import numpy as np


def compute_stats(data, min_val, max_val):
    means = [m for m, s in data if m is not None]
    normalized_means = [(m - min_val) / (max_val - min_val) if max_val > min_val else 0 for m in means]

    if len(means) == 0:
        return np.nan, np.nan

    mean_avg = np.mean(normalized_means)
    std = 1.15 * np.std(normalized_means) / np.sqrt(len(normalized_means))

    return mean_avg, std


def print_latex_table(all_algorithms, environments):

    for alg in all_algorithms:
        print(f"\\textbf{{{alg}}}", end="")
        for env_name in environments.keys():
            algo_data = environments[env_name].get(alg)
            if algo_data:
                all_means = [mean for data in environments[env_name].values() for mean, _ in data if mean is not None]
                min_val = min(all_means) if all_means else np.nan
                max_val = max(all_means) if all_means else np.nan

                avg, adj_std = compute_stats(algo_data, min_val, max_val)
                if np.isnan(avg) or np.isnan(adj_std):
                    print(" & N/A", end="")
                else:
                    print(f" & \({avg:.2f} \pm {adj_std:.2f}\)", end="")
            else:
                print(" & N/A", end="")
        print(" \\\\")


if __name__ == '__main__':

    # Example

    print(
        "\\textbf{Algorithms\\textbackslash Environments} & "
        "\\textbf{LBF} & "
        "\\textbf{RWARE} & "
        "\\textbf{Spread (MPE)} & "
        "\\textbf{Overcooked} & "
        "\\textbf{Petting Zoo} & "
        "\\textbf{Pressure Plate}\\\\"
    )
    print("\\midrule")

    _environments = {
        "LBF": {
            "QMIX": [(0., 0.), (0., 0.), (0., 0.), (0., 0.), (0., 0.), (0., 0.), (0., 0.)],
            "QPLEX": [(0., 0.), (0., 0.), (0., 0.), (0., 0.), (0., 0.), (0., 0.), (0., 0.)],
            "MAA2C": [(0., 0.), (0., 0.), (0., 0.), (0., 0.), (0., 0.), (0., 0.), (0., 0.)],
            "MAPPO": [(0., 0.), (0., 0.), (0., 0.), (0., 0.), (0., 0.), (0., 0.), (0., 0.)],
            "HAPPO": [(0., 0.), (0., 0.), (0., 0.), (0., 0.), (0., 0.), (0., 0.), (0., 0.)],
            "MAT-DEC": [(0., 0.), (0., 0.), (0., 0.), (0., 0.), (0., 0.), (0., 0.), (0., 0.)],
            "COMA": [(0., 0.), (0., 0.), (0., 0.), (0., 0.), (0., 0.), (0., 0.), (0., 0.)],
            "EOI": [(0., 0.), (0., 0.), (0., 0.), (0., 0.), (0., 0.), (0., 0.), (0., 0.)],
            "MASER": [(0., 0.), (0., 0.), (0., 0.), (0., 0.), (0., 0.), (0., 0.), (0., 0.)],
            "EMC": [(0., 0.), (0., 0.), (0., 0.), (0., 0.), (0., 0.), (0., 0.), (0., 0.)],
            "CDS": [(0., 0.), (0., 0.), (0., 0.), (0., 0.), (0., 0.), (0., 0.), (0., 0.)]
        },
        "RWARE": {
            "QMIX": [(0., 0.), (0., 0.), (0., 0.)],
            "QPLEX": [(0., 0.), (0., 0.), (0., 0.)],
            "MAA2C": [(0., 0.), (0., 0.), (0., 0.)],
            "MAPPO": [(0., 0.), (0., 0.), (0., 0.)],
            "HAPPO": [(0., 0.), (0., 0.), (0., 0.)],
            "MAT-DEC": [(0., 0.), (0., 0.), (0., 0.)],
            "COMA": [(0., 0.), (0., 0.), (0., 0.)],
            "EOI": [(0., 0.), (0., 0.), (0., 0.)],
            "EMC": [(0., 0.), (0., 0.), (0., 0.)],
            "MASER": [(0., 0.), (0., 0.), (0., 0.)],
            "CDS": [(0., 0.), (0., 0.), (0., 0.)]
        },
        "Spread (MPE)": {
            "QMIX": [(0., 0.), (0., 0.), (0., 0.)],
            "QPLEX": [(0., 0.), (0., 0.), (0., 0.)],
            "MAA2C": [(0., 0.), (0., 0.), (0., 0.)],
            "MAPPO": [(0., 0.), (0., 0.), (0., 0.)],
            "HAPPO": [(0., 0.), (0., 0.), (0., 0.)],
            "MAT-DEC": [(0., 0.), (0., 0.), (0., 0.)],
            "COMA": [(0., 0.), (0., 0.), (0., 0.)],
            "EOI": [(0., 0.), (0., 0.), (0., 0.)],
            "MASER": [(0., 0.), (0., 0.), (0., 0.)],
            "EMC": [(0., 0.), (0., 0.), (0., 0.)],
            "CDS": [(0., 0.), (0., 0.), (0., 0.)]
        },
        "Petting Zoo": {
            "QMIX": [(0., 0.), (0., 0.), (0., 0.)],
            "QPLEX": [(0., 0.), (0., 0.), (0., 0.)],
            "MAA2C": [(0., 0.), (0., 0.), (0., 0.)],
            "MAPPO": [(0., 0.), (0., 0.), (0., 0.)],
            "HAPPO": [(0., 0.), (0., 0.), (0., 0.)],
            "MAT-DEC": [(0., 0.), (0., 0.), (0., 0.)],
            "COMA": [(0., 0.), (0., 0.), (0., 0.)],
            "EOI": [(0., 0.), (0., 0.), (0., 0.)],
            "MASER": [(0., 0.), (0., 0.), (0., 0.)],
            "EMC": [(0., 0.), (0., 0.), (0., 0.)],
            "CDS": [(0., 0.), (0., 0.), (0., 0.)]
        },
        "Overcooked": {
            "QMIX": [(0., 0.), (0., 0.), (0., 0.)],
            "QPLEX": [(0., 0.), (0., 0.), (0., 0.)],
            "MAA2C": [(0., 0.), (0., 0.), (0., 0.)],
            "MAPPO": [(0., 0.), (0., 0.), (0., 0.)],
            "HAPPO": [(0., 0.), (0., 0.), (0., 0.)],
            "MAT-DEC": [(0., 0.), (0., 0.), (0., 0.)],
            "COMA": [(0., 0.), (0., 0.), (0., 0.)],
            "EOI": [(0., 0.), (0., 0.), (0., 0.)],
            "MASER": [(0., 0.), (0., 0.), (0., 0.)],
            "EMC": [(0., 0.), (0., 0.), (0., 0.)],
            "CDS": [(0., 0.), (0., 0.), (0., 0.)]
        },
        "PressurePlate": {
            "QMIX": [(0., 0.), (0., 0.)],
            "QPLEX": [(0., 0.), (0., 0.)],
            "MAA2C": [(0., 0.), (0., 0.)],
            "MAPPO": [(0., 0.), (0., 0.)],
            "HAPPO": [(0., 0.), (0., 0.)],
            "MAT-DEC": [(0., 0.), (0., 0.)],
            "COMA": [(0., 0.), (0., 0.)],
            "EOI": [(0., 0.), (0., 0.)],
            "MASER": [(0., 0.), (0., 0.)],
            "EMC": [(0., 0.), (0., 0.)],
            "CDS": [(0., 0.), (0., 0.)]
        }
    }

    _all_algorithms = set()
    for env in _environments.values():
        _all_algorithms.update(env.keys())

    print_latex_table(_all_algorithms, _environments)
