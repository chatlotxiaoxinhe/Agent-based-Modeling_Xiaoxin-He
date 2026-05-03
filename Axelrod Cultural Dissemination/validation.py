from model import AxelrodModel

results = []
for seed in range(10):
    m = AxelrodModel(width=10, height=10, num_features=5, num_traits=10,
                     neighborhood_size=4, torus=False,
                     events_per_step=2000, seed=seed)
    while m.running:
        m.step()
    r = m.datacollector.model_vars["Cultural Regions"][-1]
    results.append(r)

print(f"Mean: {sum(results)/len(results):.1f}, range [{min(results)}, {max(results)}]")
