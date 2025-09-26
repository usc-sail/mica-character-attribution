import socket

HOST = socket.gethostname()
if HOST == "redondo":
    # lab server
    DATADIR = "/data1/sbaruah/mica-character-attribution"
elif HOST.endswith("hpc.usc.edu"):
    # university HPC compute
    DATADIR = "/scratch1/sbaruah/mica-character-attribution"
else:
    # AWS EC2
    DATADIR = "/home/ubuntu/data/mica-character-attribution"