from wandb import Api

run = Api().run("gatech-sysml/veeksha/j7q5k7d7")
print("before:", run.tags)
if hasattr(run, "add_tag"):
    run.add_tag("BEST_CONFIG")
else:
    run.update({"tags": list(set((run.tags or []) + ["BEST_CONFIG"]))})
run = Api().run("gatech-sysml/veeksha/j7q5k7d7")
print("after:", run.tags)