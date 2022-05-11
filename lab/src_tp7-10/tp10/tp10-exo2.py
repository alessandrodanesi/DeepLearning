import matplotlib
from matplotlib import pyplot as plt
# matplotlib.use('TkAgg')
import seaborn as sns
from utils import PositionalEncoding
sns.set()

pos_embedding = PositionalEncoding(d_model=50, max_len=2600)
pe = pos_embedding.pe[0]
scalar_prod = (pe @ pe.T).numpy()
sns.heatmap(scalar_prod)
plt.show()

plt.plot(pe[0], label="$pe_0$")
plt.plot(pe[1000], label="$pe_{1000}$")
plt.plot(pe[2555], label="$pe_{2555}$")
plt.legend()
plt.show()