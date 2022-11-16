import argparse
import logging

from termcolor import colored


class CustomFormatter(logging.Formatter):
    def __init__(self, format, use_termcolor=False):
        super().__init__()
        self.format_str = format
        self.use_termcolor = use_termcolor
        self.formats = {
                logging.DEBUG   : colored(format, "grey"),
                logging.INFO    : colored(format, "green"),
                logging.WARNING : colored(format, "yellow"),
                logging.ERROR   : colored(format, "red"),
                logging.CRITICAL: colored(format, "red", attrs=["bold"]),
                }

    def format(self, record):
        if self.use_termcolor:
            log_fmt = self.formats.get(record.levelno)
            formatter = logging.Formatter(log_fmt)
            return formatter.format(record)

        return logging.Formatter(self.format_str).format(record)


def logging_setup(file, name, **kwargs):
    log_file_name = f"{file.split('.')[0]}.log"
    sh, fh = logging.StreamHandler(), logging.FileHandler(log_file_name, mode='w')
    fmt = "[%(filename)s:%(lineno)s - %(funcName)15s() ] %(message)s"
    sh.setFormatter(CustomFormatter(fmt, use_termcolor=True))
    fh.setFormatter(CustomFormatter(fmt))
    logging.basicConfig(handlers=[sh, fh], **kwargs)
    return logging.getLogger(name)


def ranged_type(value_type, min=None, max=None):
    def range_checker(arg: str):
        try:
            f = value_type(arg)
        except ValueError:
            raise argparse.ArgumentTypeError(f"must be a valid {value_type}")
        if min is not None and f < min:
            raise argparse.ArgumentTypeError(f"must be >= {min}")
        if max is not None and f > max:
            raise argparse.ArgumentTypeError(f"must be <= {max}")
        return f

    return range_checker


def parse_args(args_list=["q", "a", "e", "n", "c", "f", "d", "s", "t", "alg"]):
    parser = argparse.ArgumentParser()
    if "q" in args_list:
        parser.add_argument("q", help="Dimension of the grid (quantization)", type=ranged_type(int, min=0), )

    if "a" in args_list:
        parser.add_argument("a", help="Alpha Parameter [min: 0.0, max: 1.0]", type=ranged_type(float, min=0, max=1), )

    if "e" in args_list:
        parser.add_argument("e", help="Epsilon Parameter [min: 0.0, max: 1.0]", type=ranged_type(float, min=0, max=1), )

    if "n" in args_list:
        parser.add_argument("n",
                            help="Total number of episodes to train the agent. [min: 0]",
                            type=ranged_type(int, min=0), )

    if "c" in args_list:
        parser.add_argument("c",
                            help="Number of episodes between checkpoints. [min: 0]",
                            type=ranged_type(int, min=0), )

    if "f" in args_list:
        parser.add_argument("f", help="File path where the final Q-Table will be saved.", type=str, )

    if "d" in args_list:
        parser.add_argument("-d", "--draw", help="Draw the game to the screen. "
                                                 "This is useful in combination with the --test flag, but rendering will slow down the training process. "
                                                 "[Default: False]", action="store_true", )

    if "s" in args_list:
        parser.add_argument("-s",
                            "--speed",
                            help="Speed factor for frame rendering when --draw is enabled. ",
                            type=ranged_type(float, min=0.001),
                            default=1.0, )

    if "t" in args_list:
        parser.add_argument("-t",
                            "--test",
                            help="Test the model from the given file, instead of training [Default: False]",
                            action="store_true", )

    if "alg" in args_list:
        parser.add_argument("--alg",
                            help="Learning Algorithm (Q-LEARNING or SARSA)",
                            type=str,
                            choices=["Q-LEARNING", "SARSA"],
                            default="Q-LEARNING", )

    return parser.parse_args()
