import numpy as np
import math
import matplotlib.pyplot as plt


def densite_gaussienne(mus, sigmas, n, dimension):
    # fonction de densite
    def f(x):
        s=0
        if 0 < min(x) and max(x)< 1:
            s+=1
        for i in range(len(sigmas)):
            gamma_inv = 1 / (sigmas[i] ** 2) * np.identity(dimension)
            s+= 1 / (math.sqrt(sigmas[i]**2 * math.pi)**dimension) * math.exp(-0.5 * (x - mus[i]) .dot( gamma_inv ).dot((x-mus[i])))
        return s/(1+len(mus))

    #calcul de c

    # c1 = 1+sum([1/ np.sqrt(sigmas[i]**2*math.pi)**dimension for i in range(len(mus))])
    # majoration theorique
    
    c2=1
    if len(mus)!=0:
        c2 = (1+max([1 / (math.sqrt(sigmas[i]**2 * math.pi)**dimension) for i in range(len(mus))]))/(1+len(mus))
    # plus grosse bosse
    c = c2
    # on majore par rapport à la plus grosse bosse pour pouvoir laisser le partage de vote
    print(c)
    samples = []
    while len(samples)<n:
            # print(len(samples))
            x = np.random.uniform(0,1,(dimension))
            y = np.random.uniform(0,c)
            if f(x)>y :
                samples.append(x)

    return samples

def affichage_densite(mus, sigmas, dimension,points):
    c=1
    if len(mus)!=0:
        c = (1+max([1 / (math.sqrt(sigmas[i]**2 * math.pi)**dimension) for i in range(len(mus))]))/(1+len(mus))
    def f(x):
        s=0
        if 0 < min(x) and max(x)< 1:
            s+=1
        for i in range(len(sigmas)):
            gamma_inv = 1 / (sigmas[i] ** 2) * np.identity(dimension)
            s+= 1 / (math.sqrt(sigmas[i]**2 * math.pi)**dimension) * math.exp(-0.5 * (x - mus[i]) .dot( gamma_inv ).dot((x-mus[i])))
        return min(s/(1+len(mus)),c)
    
     
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

    for i, point in enumerate(points):
        z = 1.2*c
        ax.scatter(*point, z, color='r')
        ax.plot([point[0], point[0]], [point[1], point[1]], [0, z], color='k')
        ax.text(*point, z, str(i), color='k')


    ax.set_title("fonction densite f. nombre d'echantillon "+str(len(Z)**2))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()

     