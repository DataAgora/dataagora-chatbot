import json

base_url = "static/my_weights_12/my_weights_{o}.txt"

save_base_url = "static/weights/weights_{o}_{n}.json"

for i in range(77):
    full_base = base_url.format(o=i)
    with open(full_base, "r") as f:
        weights_arr = json.loads(f.read())
        for j, weights in enumerate(weights_arr):
            full_save = save_base_url.format(o=i, n=j)
            with open(full_save, "w") as ff:
                ff.write(json.dumps(weights))