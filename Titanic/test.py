#!/usr/bin/env python


# coding: utf-8


import sys


try:

    import Tkinter as tk      # Python 2


except ImportError:

    import tkinter as tk      # Python 3


print("Tcl Version: {}".format(tk.Tcl().eval('info patchlevel')))


print("Tk Version: {}".format(tk.Tk().eval('info patchlevel')))


sys.exit()
