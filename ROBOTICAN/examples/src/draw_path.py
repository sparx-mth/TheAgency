import csv
import matplotlib.pyplot as plt
import time

for drone_id in ["R1", "R2", "R3"]:
    xs, ys = [], []
    with open(f"paths/{drone_id}_path.csv") as f:
        r = csv.DictReader(f)
        for row in r:
            xs.append(float(row["x"]))
            ys.append(float(row["y"]))
    plt.plot(xs, ys, label=drone_id)
now = time.time()
str_time = time.strftime("%Y%m%d_%H%M%S")
plt.axis("equal")
plt.legend()
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("Drone paths")
plt.savefig(f"paths/f{str_time}_drone_paths.png", dpi=150)
