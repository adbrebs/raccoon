from functools import partial
import pickle
import textwrap


def wrap_text(text, indent_level=0, width=80):
    """Wrap string text so that it doesn't exceed a specified width."""
    space = ' '*3
    return textwrap.fill(text, width,
                         initial_indent=indent_level * space,
                         subsequent_indent=(indent_level + 1) * space)


class Colors:
    BLUE = '\033[0;34m'
    GREEN = '\033[0;32m'
    YELLOW = '\x1b[0;33m'
    RED = '\033[0;31m'
    MAGENTA = '\x1b[1;35m'
    GREY = '\x1b[0;37m'
    ENDC = '\x1b[0m'


def print_color(color, info, value="", newline=False, indent_level=0):
    """Helper function to print the following information:
    [info] value
    where [info] is printed with the given color

    Args:
        color (strings defined in Colors class):
            Correponds to the color id, for example Colors.GREEN.
        info (string): Information printed in brackets at the beginning of the line.
        value (string): Information printed after [info].
        newline (bool, default False): If True, append a new line \n before the printed line.
        indent_level (int, default 0): Indicates the level of indentation. 0 is no indentation.
    """
    prefix = "\n" if newline else ""
    print(prefix + str_color(color, wrap_text(f"[{info}] ", indent_level=indent_level)) + value)


print_green = partial(print_color, Colors.GREEN)
print_red = partial(print_color, Colors.RED)
print_blue = partial(print_color, Colors.BLUE)
print_yellow = partial(print_color, Colors.YELLOW)


def str_color(color, s):
    """Helper function to convert a string into a colored/styled string.
    You then need to call `print` to print the colored string.
    Args:
        color (strings defined in Colors class):
            Correpond to the color id, for example Colors.GREEN.
        s (string):
            The string that we want to display into a different color.
    """
    return color + s + Colors.ENDC


str_green = partial(str_color, Colors.GREEN)
str_red = partial(str_color, Colors.RED)
str_blue = partial(str_color, Colors.BLUE)
str_magenta = partial(str_color, Colors.MAGENTA)
str_yellow = partial(str_color, Colors.YELLOW)
str_grey = partial(str_color, Colors.GREY)


def pretty_time(orig_seconds):
    """Transforms seconds into a string with days, hours, minutes and seconds."""
    days, seconds = divmod(round(orig_seconds), 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)

    out = []
    if days > 0:
        out.append(f"{days}d")
    if hours > 0:
        out.append(f"{hours}h")
    if minutes > 0:
        out.append(f"{minutes}m")
    if seconds > 0:
        out.append(f"{seconds}s")
    else:
        if out:
            s = ""
        elif orig_seconds == 0:
            s = "0s"
        else:
            s = "<0s"
        out.append(s)

    return "".join(out)
