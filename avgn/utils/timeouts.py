# for signalling timeouts
import signal

class TimeoutException(Exception):  # Custom exception class
    pass

def timeout_handler(signum, frame):  # Custom signal handler
    raise TimeoutException

signal.signal(signal.SIGALRM, timeout_handler)