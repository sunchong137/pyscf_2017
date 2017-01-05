import sys
from datetime import datetime

stdout = sys.stdout

def time():
    stdout.write(datetime.now().strftime("%y %b %d %H:%M:%S") + "  ")
    stdout.flush()

def result(msg, *args):
    time()
    stdout.write("********" + "  " + msg + "\n")
    stdout.flush()

def section(msg, *args):
    time()
    stdout.write("########" + "  " + msg + "\n")
    stdout.flush()

def debug(msg, *args):
    time()
    stdout.write("  DEBUG " + "  " + msg + "\n" )

def info(msg, *args):
    time()
    stdout.write("  INFO  " + "  " + msg + "\n")

if __name__ == "__main__":
    msg = "This is a test"
    result(msg)
    section(msg)
           
