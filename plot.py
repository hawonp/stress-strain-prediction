import os

from matplotlib import pyplot as plt

loss_path = "output/losses.txt"
scores_path = "output/scores.txt"

# delete old plots
os.remove("output/training_loss.png")
os.remove("output/testing_loss.png")
os.remove("output/scores.png")

with open(loss_path, "r") as f:
    losses = f.readlines()
    losses = [x.strip().split(",") for x in losses]
    losses = [(float(x[0]), float(x[1])) for x in losses]

with open(scores_path, "r") as f:
    scores = f.readlines()
    scores = [x.strip().split(",") for x in scores]
    scores = [(float(x[0]), float(x[1])) for x in scores]

# plot losses and save
plt.plot([x[0] for x in losses], label="Training Loss")
plt.legend()
plt.savefig("output/training_loss.png")
plt.clf()

plt.plot([x[1] for x in losses], label="Testing Loss")
plt.legend()
plt.savefig("output/testing_loss.png")

# plot scores and save
plt.clf()
plt.plot([x[0] for x in scores], label="Training Score")
plt.plot([x[1] for x in scores], label="Testing Score")
plt.legend()
plt.savefig("output/scores.png")
