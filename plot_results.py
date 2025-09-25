import seaborn as sns
import pandas as pd

input_name = "networks_inc_n"
x_var = "n"
df = pd.read_csv(f"./output/{input_name}_results.csv")
print(df)

df["time_cpu"] = df["cpu_time"]
df["time_gpu"] = df["gpu_time"]
df["id"] = df.index
df2 = pd.wide_to_long(df, "time", i="id", j="system", sep="_", suffix=".+")

print(df2)
plt = sns.lineplot(
    data=df2,
    x=x_var, y="time", hue="system",
    markers=True, dashes=False
)
plt.set(yscale='log')
plt.figure.savefig(f"./output/{input_name}_results_fig.png")
