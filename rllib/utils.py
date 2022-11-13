import argparse


class Utils:
    @staticmethod
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

    @staticmethod
    def parse_args(args_list=["q", "a", "e", "n", "c", "f", "d", "s", "t", "alg"]):
        parser = argparse.ArgumentParser()
        if "q" in args_list:
            parser.add_argument(
                "q",
                help="Dimension of the grid (quantization)",
                type=Utils.ranged_type(int, min=0),
            )

        if "a" in args_list:
            parser.add_argument(
                "a",
                help="Alpha Parameter [min: 0.0, max: 1.0]",
                type=Utils.ranged_type(float, min=0, max=1),
            )

        if "e" in args_list:
            parser.add_argument(
                "e",
                help="Epsilon Parameter [min: 0.0, max: 1.0]",
                type=Utils.ranged_type(float, min=0, max=1),
            )

        if "n" in args_list:
            parser.add_argument(
                "n",
                help="Total number of episodes to train the agent. [min: 0]",
                type=Utils.ranged_type(int, min=0),
            )

        if "c" in args_list:
            parser.add_argument(
                "c",
                help="Number of episodes between checkpoints. [min: 0]",
                type=Utils.ranged_type(int, min=0),
            )

        if "f" in args_list:
            parser.add_argument(
                "f",
                help="File path where the final Q-Table will be saved.",
                type=str,
            )

        if "d" in args_list:
            parser.add_argument(
                "-d",
                "--draw",
                help="Draw the game to the screen. "
                "This is useful in combination with the --test flag, but rendering will slow down the training process. "
                "[Default: False]",
                action="store_true",
            )

        if "s" in args_list:
            parser.add_argument(
                "-s",
                "--speed",
                help="Speed factor for frame rendering when --draw is enabled. ",
                type=Utils.ranged_type(float, min=0.001),
                default=1.0,
            )

        if "t" in args_list:
            parser.add_argument(
                "-t",
                "--test",
                help="Test the model from the given file, instead of training [Default: False]",
                action="store_true",
            )

        if "alg" in args_list:
            parser.add_argument(
                "--alg",
                help="Learning Algorithm (Q-LEARNING or SARSA)",
                type=str,
                choices=["Q-LEARNING", "SARSA"],
                default="Q-LEARNING",
            )

        return parser.parse_args()
