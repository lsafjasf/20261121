import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USER_SITE = "/home/hj/.local/lib/python3.14/site-packages"
if os.path.isdir(USER_SITE) and USER_SITE not in sys.path:
    sys.path.insert(0, USER_SITE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
