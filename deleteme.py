import time

amount = 20

for i in range(400):
    if i % 20 == 0:
        print(f"\r{i}")
    else:
        print(f"\r{i}", end="")

    time.sleep(0.1)
