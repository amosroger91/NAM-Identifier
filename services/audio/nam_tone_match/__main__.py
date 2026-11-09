import sys

from .cli import audition_worker, main

sys.exit(audition_worker() if sys.argv[1:] == ['--audition-worker'] else main())
