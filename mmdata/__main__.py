"""
mmdata.__main__ module
"""

import sys
from mmdata.cli import main, parse_arguments


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
