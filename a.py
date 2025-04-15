import os
import json

output_cache_file = "sgl-cache-tp1-radix-conv/cache_telemetry.json"

if os.path.exists(output_cache_file):
    with open(output_cache_file, "r") as f:
        output_cache = json.load(f)

    # delete previous cache telemtry data to restart
    # os.remove(output_cache_file)

    qps = 0.62
    
    # tag with current qps and save
    output_cache["qps"] = qps
    new_output_cache_file = "sgl-cache-tp1-radix-conv/cache_telemetry_qps_0.62.json"
    
    with open(new_output_cache_file, "w") as f:
        json.dump(output_cache, f)