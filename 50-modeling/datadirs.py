"""Get data directory"""
import socket

host = socket.gethostname()
if host == "redondo":
    # lab server
    datadir = "/data1/sbaruah/mica-character-attribution"
elif host.endswith("hpc.usc.edu"):
    # university HPC compute
    datadir = "/scratch1/sbaruah/mica-character-attribution"
else:
    # AWS EC2
    datadir = "/home/ubuntu/data/mica-character-attribution"