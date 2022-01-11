from shannon import ent, js_div
import numpy as np
import math

d1 = [15, 12, 5, 6, 2, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1]
d2 = [13, 15, 7, 6, 6, 3, 3, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1]
d3 = [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 11, 12, 13, 24, 15]
d4 = [1, 1, 1, 1, 1, 1, 1, 2, 4, 6, 26, 3, 3, 1, 1, 1, 1]

ds1 = [d1, d2, d3, d4]

for i, d in enumerate(ds1):
    log = math.log(len(d), 2)
    print(f"Device {i} entropy {ent(d)}, ideal {log}")


print("Distribution distance metric Jensen-Shannon divergence")
print(" 0 = identical ")
print(" 1 = max. different ")

# d1 i d2 su slični, udaljenost je mala
print(js_div(d1, d2))
print(js_div(d2, d1))

# d1 i d3 su manje slični, veća udaljenost
print(js_div(d1, d3))
print(js_div(d3, d1))


def get_avg_distance(ds):
    shape = (len(ds), len(ds))
    ret = np.zeros(shape)

    for i in range(shape[0]):
        for j in range(i, shape[1]):
            ret[i, j] = js_div(ds[i], ds[j])

    avg_js_div = np.mean(list(ret[i] for i in zip(*np.triu_indices_from(ret, k=1))))
    return avg_js_div


print("Average for DS1", get_avg_distance(ds1))

ds2 = [
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 0],
]
print("Average for DS2", get_avg_distance(ds2))

ds3 = [
    [100, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
]
print("Average for DS3", get_avg_distance(ds3))
