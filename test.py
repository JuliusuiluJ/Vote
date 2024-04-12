# import matplotlib.pyplot as plt
# import numpy as np
# s = 0.01*np.random.randn(1000)+0.5
# count, bins, ignored = plt.hist(s, 15, density=True)
# plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
# plt.axis([0,1,0,50])
# plt.show()
# s = 0.1*np.random.randn(1000)+0.5
# count, bins, ignored = plt.hist(s, 15, density=True)
# plt.plot(bins, np.ones_like(bins), linewidth=2, color='g')
# plt.axis([0,1,0,50])
# plt.show()
# plt.show()

import matplotlib.pyplot as plt
import numpy as np
import math
sigmas= np.array([0.15,0.2,0.1])
mus = np.array([[0.5,0.3],[0.3,0.9],[0.7,0.7]])
mus = pos_candidats = np.random.uniform(0, 1, (5, 2))
sigmas = np.random.uniform(0.1, 0.2, (len(mus)))
#print(mus)
#print(sigmas)

dimension = 2
# Définir la fonction

import matplotlib.pyplot as plt
import numpy as np

# Définir la fonction
def f(x):
    s=0
    if 0 < min(x) and max(x)< 1:
        s+=1
    for i in range(len(sigmas)):
        gamma_inv = 1 / (sigmas[i] ** 2) * np.identity(dimension)
        s+= 1 / (math.sqrt(sigmas[i]**2 * math.pi)**dimension) * math.exp(-0.5 * (x - mus[i]) .dot( gamma_inv ).dot((x-mus[i])))
    return min(s/(1+len(mus)),(1+max([1 / (math.sqrt(sigmas[i]**2 * math.pi)**dimension) for i in range(len(mus))]))/(1+len(mus)))


# Générer des valeurs x et y
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)

# Créer une grille de coordonnées
X, Y = np.meshgrid(x, y)

# Calculer les valeurs Z
Z = np.zeros(X.shape)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i,j]=f([X[i,j],Y[i,j]])

# Créer le graphique 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
#ax.scatter(X, Y, Z, marker='o')
ax.set_title("Graphique de la fonction densite f echantillon "+str(len(Z)**2))
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.show()
