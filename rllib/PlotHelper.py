from matplotlib import pyplot as plt
import os


class PlotHelper:

    @staticmethod
    def plot_from_dict(plot, savefig="plot.pdf"):
        plt.clf()
        for key, values in plot.items():
            func = getattr(plt, key)
            if isinstance(values, list) and all(isinstance(x, dict) for x in values):
                for value in values:
                    args = value.pop("args") if "args" in value else []
                    func(*args, **value)
            elif isinstance(values, dict):
                func(*values["args"], **{k: v for k, v in values.items() if k != "args"})
            else:
                func(values)

        plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.legend()
        plt.draw()
        if savefig:
            if "/" in savefig:
                os.makedirs(os.path.dirname(savefig), exist_ok=True)
            plt.savefig(savefig)
