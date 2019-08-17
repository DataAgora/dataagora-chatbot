import json
import os

base_url = "static/weights.json"

save_base_url = "static/weights_12/weights_{o}.json"

with open(base_url, "w") as f:
    for i in range(77):
        full_base = save_base_url.format(o=i)
        print(full_base)
        with open(full_base, "r") as ff:
            f.write(ff.read())
            f.write('\n')
    