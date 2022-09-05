from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import glob
import re
import numpy as np


def retrieve_info(root: Path) -> list:
    """Retrieve results from hpc or development log files

    Args:
        root (Path): Root folder where the log files are

    Returns:
        list: 
            A 3D list seperating the data into groups (e.g. result, best)\n
            \t[Type][Experiment][Values]
    """

    files = glob.glob(f"{root}/*")
    files.sort(reverse=True)
       

    ALL_TIME_MAX = 0

    avg = defaultdict(list)
    persistency = defaultdict(dict)

    for log_file in files:
        with open(log_file, "r") as f:
            print(log_file)
            all_lines = f.read()

            # Skip all experiments that errored
            if re.findall(r"slurmstepd", all_lines) != []:
                print("ERROR")
                continue

            name = re.findall(r"Name\s+: ([\w+|_]+)", all_lines)
            print(name)

            # hyper_param = re.findall(r"HYPER.*", all_lines)
            exp_id = re.findall(r"- EXP_ID\s+\| large_(\d+)", all_lines)
            max_score = re.findall(
                r"The highest fscore   : \((0\.\d+), (0\.\d+)\)", all_lines)
            scores = re.findall(
                r"(Prec.*|Recall.*|Fscore.*|Acc.*)%", all_lines)
            # Filtering out the experiments only done on 800x800 and that have
            # correct scores (acc, prec, recall, F-score)
            if len(scores) == 4 and len(exp_id) == 1:
                exp_id = int(exp_id[0])

                if exp_id not in persistency.keys():
                    persistency[f"{exp_id}"] = {
                        "name": name[0],
                        "file": log_file.split("/")[-1]
                    }

                # Get F-score and ignore the first 12 experiments, since they
                # did not have great results and the `Best possible` score
                if (sc := float(scores[3].split(":")[1])) > 5 and exp_id > 12:
                    print(exp_id, sc)
                    ALL_TIME_MAX = sc if sc > ALL_TIME_MAX else ALL_TIME_MAX
                    avg[f"{exp_id:02}"].append(sc)
                    if max_score != []:
                        avg[f"{exp_id}_b"].append(float(max_score[0][0])*100)
                        print(max_score[0][0])
                    else:
                        avg[f"{exp_id}_b"].append(np.NaN)
        print()

    avg = dict(sorted(avg.items()))
    new_names = [f'{_a}{b}' for _a in ['Base'] + list(range(1, 100)) for b in ['', '_b']]
    for k, new_k in zip(avg.keys(), new_names):
        persistency[k]['renamed'] = new_k

    print(f"Boxplt | Orignal | {'Creation file':20} | Experiment desc")
    for k, v in reversed(persistency.items()):
        if "renamed" in v.keys() and "name" in v.keys():
            print(f"{v['renamed']:6} | {k:7} | {v['file']:20} | {v['name']}")
    print("-"*50)
    print(f"Max F-score: {ALL_TIME_MAX}")

    key_list = list(avg.keys())

    # Going over the retrieved scores and divide on:
    #   result and Best possible score
    data_groups = [
        [avg[k] for k in key_list[::2]],
        [avg[k] for k in key_list[1::2]]
    ]
    return data_groups


def box_plot_generator(data_groups: list) -> None:
    """Generating a box plot for all the retrieved scores from the log files

    Args:
        data_groups (list): The data seperate in groups (result, best) (3D list)

    Base on the answer of Kuzeko on SO question:
        https://stackoverflow.com/questions/16592222/matplotlib-group-boxplots
    """
    ax = plt.gca()

    ax.grid(True, linestyle='dotted')
    ax.set_axisbelow(True)

    plt.xlabel('Model iteration')
    plt.ylabel('F-score')
    plt.title('Results from experiments')

    # Labels for the data:
    labels_list = ["Base"] + list(range(1, len(data_groups[0])))

    width = 1/len(labels_list)
    xlocations = [x*((3 + len(data_groups))*width)
                  for x in range(len(data_groups[0]))]

    space = len(data_groups)/2

    # Offset the positions per group:

    group_positions = []
    for num, dg in enumerate(data_groups):
        _off = (0 - space + (0.6+num))
        group_positions.append([x+_off*(width+0.01) for x in xlocations])
    colors = ['pink', 'lightblue', 'lightgreen', 'violet']
    elements = []

    # Box plot creation
    for dg, pos, c in zip(data_groups, group_positions, colors):
        elements.append(
            ax.boxplot(dg,
                       positions=pos, widths=width,
                       boxprops=dict(facecolor=c),
                       medianprops=dict(color='grey'),
                       patch_artist=True
                       )
        )

    ax.set_xticks(xlocations)
    ax.set_xticklabels(labels_list, rotation=0)

    ax.legend(
        [element["boxes"][0] for element in elements],
        ["Results", "Best possible"]
    )

    plt.show()


if __name__ == "__main__":
    data = retrieve_info(Path("../logs"))
    print(data)
    box_plot_generator(data)
