import socket

host = socket.gethostname()
if host == "redondo":
    datadir = "/data1/sbaruah/mica-character-attribution"
elif host.endswith("hpc.usc.edu"):
    datadir = "/project/shrikann_35/sbaruah/mica-character-attribution"