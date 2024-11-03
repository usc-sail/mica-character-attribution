import socket

host = socket.gethostname()
if host == "hermosa":
    datadir = "/data1/sbaruah/mica-character-attribution"
elif host.endswith("hpc.usc.edu"):
    datadir = "/scratch1/sbaruah/mica-character-attribution"