"""Get data directory"""
import socket

host = socket.gethostname()
if host == "redondo":
    datadir = "/data1/sbaruah/mica-character-attribution"
elif host.endswith("hpc.usc.edu"):
    datadir = "/scratch1/sbaruah/mica-character-attribution"