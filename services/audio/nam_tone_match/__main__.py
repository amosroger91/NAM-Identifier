# Main entry point for NAM tone matching

import sys
from .cli import main


def run():
    """
    Run the application.
    """
    sys.exit(main())


if __name__ == '__main__':
    run()