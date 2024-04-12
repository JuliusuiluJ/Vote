import numpy as np
import densite
from collections import defaultdict
import sys
from scipy.optimize import linprog

dimension = 2
coeffs= [1,10] # TODO
proba_influence = 0


def dist(a,b):
    return sum((a[i]-b[i])**2 for i in range(len(a)))


def avant(voeu,i,j):
    for v in voeu :
        if i==v :
            return i
        if j==v:
            return j
    return -1


def voeux(pos_candidats,pos_electeurs):
    voeux_elec = []
    voeux_approb = [] 
    distances = []
    for e in pos_electeurs:
        distance = []
        for i,c in enumerate(pos_candidats) :
            distance.append((dist(e,c),i))
        distance.sort()
        voeu = [j for (i,j) in distance]
        #print(distance)
        voeu_approb = [j for (i,j) in distance if i<dimension**(1/2)/3*2]
        voeux_elec.append(voeu)
        voeux_approb.append(voeu_approb)
        distances.append(distance)
    voeux_elec=np.array(voeux_elec)
    return distances


def MJ1(distances):
    depouillage = defaultdict(int)
    for distance in distances:
        voeu = [j for (i,j) in distance]
        depouillage[voeu[0]]+=1
    maxi = -1
    indice = -1
    for i in depouillage:
        if depouillage[i]>maxi:
            maxi = depouillage[i]
            indice = i 
    #print('MJ1 ',indice)
    return indice


def MJ2(distances):
    depouillage = defaultdict(int)
    for distance in distances:
        voeu = [j for (i,j) in distance]
        depouillage[voeu[0]]+=1
    maxi1 = -1
    indice1 = -1
    maxi2 = -1
    indice2 = -1
    for i in depouillage:
        if depouillage[i] > maxi2:
            if depouillage[i] >= maxi1:
                maxi2 = maxi1
                indice2 = indice1
                maxi1 = depouillage[i]
                indice1 = i 
            else :
                maxi2 = depouillage[i]
                indice2 = i
    depouillage = {indice1 : 0, indice2 : 0}
    for distance in distances:
        voeu = [j for (i,j) in distance]
        depouillage[avant(voeu,indice1,indice2)]+=1
    if depouillage[indice1]>depouillage[indice2]:
        #print('MJ2 ',indice1)
        return indice1
    #print('MJ2 ',indice2)
    return indice2


def C(distances):
    # Créer une liste de tous les candidats
    candidates = list(set(candidate for distance in distances for (i,candidate) in distance))
    while len(candidates) > 1:
        # Créer un dictionnaire pour stocker les scores de chaque paire de candidats
        pairs = defaultdict(int)

        # Parcourir chaque oeote
        for distance in distances:
            vote = [j for (i,j) in distance]
            # Comparer chaque candidat à tous les autres dans le vote
            for i in range(len(vote)):
                for j in range(i+1, len(vote)):
                    # Si les deux candidats sont toujours en lice
                    if vote[i] in candidates and vote[j] in candidates:
                        # Si le candidat i est préféré au candidat j, ajouter 1 au score de la paire (i, j)
                        if avant(vote,i,j)==i:
                            pairs[(vote[i], vote[j])] += 1
                        else:
                            pairs[(vote[j], vote[i])] -= 1

        # Trouver le candidat qui a le moins gagné de duels
        loser = min(candidates, key=lambda candidate: min(pairs[(candidate, other)] for other in candidates if other != candidate))

        # Éliminer ce candidat
        candidates.remove(loser)
    return candidates[0]


def CR(distances):
    votes = []
    for distance in distances:
            vote = [j for (i,j) in distance]
            votes.append(vote)

    # Nombre de candidats
    num_candidates = len(votes[0])
    
    # Matrice des duels
    duel_matrix = np.zeros((num_candidates,num_candidates))
    
    # Remplir la matrice des duels
    for vote in votes:
        for i in range(num_candidates):
            for j in range(i+1, num_candidates):
                if vote.index(i) > vote.index(j):
                    duel_matrix[i,j] += 1
                else:
                    duel_matrix[j,i] += 1
    
    # Convertir la matrice des duels en un tableau numpy
    A = np.zeros((num_candidates,num_candidates))
    for i in range(num_candidates):
        for j in range(num_candidates):
            if duel_matrix[i,j]>duel_matrix[j,i]:
                A[i,j]=1
            elif duel_matrix[i,j]<duel_matrix[j,i]:
                A[i,j]=-1

    # exemple
    # A = np.array([[0, 1, 1, 1, -1],
    #               [-1, 0, 1, -1, 1],
    #               [-1, -1, 0, 1, 1],
    #               [-1, 1, -1, 0, 1],
    #               [1, -1, -1, -1, 0]
    #               ])
                
    # Définir les contraintes de l'optimisation linéaire
    b_ub = np.zeros(num_candidates)
    b_eq = np.array([1])
        # on veut juste trouver un point donc on s'en fiche de c
    c = np.zeros(num_candidates)

    # Résoudre le problème d'optimisation linéaire
    res = linprog(c, A_ub=-A, b_ub=b_ub, A_eq=np.ones((1, num_candidates)), b_eq=b_eq, method='highs')
    gagnant = np.random.choice([i for i in range(num_candidates)], p=res.x)
    return gagnant


def APP(distances):
    depouillage = defaultdict(int)
    for distance in distances:
        voeu = [j for (i,j) in distance if i<0.2]#dimension**(1/2)/3*2]
        for v in voeu:
            depouillage[v]+=1
    maxi = 0
    indice = -1
    for i in depouillage:
        if depouillage[i]>maxi:
            maxi=depouillage[i]
            indice=i
    #print('APP',indice)
    return indice

def VST(distances):
    votes = []
    for distance in distances:
            vote = [j for (i,j) in distance]
            votes.append(vote)
    while len(votes[0])>1:
        depouillage = defaultdict(int)
        for voeu in votes:
            depouillage[voeu[0]]+=1
        mini = np.inf
        indice = -1
        for i in depouillage:
            if depouillage[i]<mini:
                mini = depouillage[i]
                indice = i 
        for voeu in votes:
            voeu.remove(indice)
    #print('VST ',votes[0][0])
    return votes[0][0]

def BOR(distances):
    depouillage = defaultdict(int)
    n = len(distances[0])
    for distance in distances:
            vote = [j for (i,j) in distance]
            for i,v in enumerate(vote) :
                depouillage[v]+=n-i
    maxi = 0
    indice = -1
    for i in depouillage:
        if depouillage[i]>maxi:
            maxi=depouillage[i]
            indice=i
    #print('BOR',indice)
    return indice

def elections(candidats,electeurs,sys_votes):
    pos_candidats = np.random.uniform(0, 1, (candidats, dimension))
    densite_electeur_mu = []
    densite_electeur_sigma = []
    for c in range(candidats):
        if np.random.random()<proba_influence:
            densite_electeur_mu.append(pos_candidats[c])
    densite_electeur_mu = np.array(densite_electeur_mu)
    densite_electeur_sigma = np.random.uniform(0.1, 0.2, (len(densite_electeur_mu)))
    #print(len(densite_electeur_mu))
    pos_electeurs = densite.densite_gaussienne(densite_electeur_mu,densite_electeur_sigma,electeurs,dimension)
    gagnants = defaultdict(int)
    res_sys = defaultdict(int)
    distances = voeux(pos_candidats,pos_electeurs)
    #print(voeux_approb)
    for i in sys_votes:
        g = sys_votes[i](distances)
        gagnants[g]+=1
        res_sys[sys_votes[i]]=g
    #print(gagnants)
    return res_sys

    densite.affichage_densite(densite_electeur_mu,densite_electeur_sigma,dimension,pos_candidats)

dimension = 5
coeffs= [1,10] # TODO
proba_influence = 1
candidats, electeurs = 10, 100
n = 100
sys_votes = {
        0: CR,
        1: C,
        2: BOR,
        3: APP,
        4: VST,
        5: MJ2,
        6: MJ1,
    }

data = np.zeros((len(sys_votes), len(sys_votes)))
for k in range(n):
    res=elections(candidats, electeurs, sys_votes)
    for i in sys_votes:
        for j in sys_votes:
            if res[sys_votes[i]]==res[sys_votes[j]]:
                data[i,j]+=1
    #if not k%(n/10) :
    #    print(data/(k+1))
data=data/n
print('')
print('nombre de candidats :',candidats,'nombre de dimensions :',dimension,'nombre d\'électeurs :',electeurs,'nombre de simulations :',n)

rounded_data = np.round(data, 2)
sys_votes2 = {}
for i in sys_votes:
    sys_votes2[i]=sys_votes[i].__name__
named_data = np.empty((data.shape[0] + 2, data.shape[1] + 2), dtype=object)
named_data[1:-1, 1:-1] = rounded_data
named_data[0, 1:-1] = list(sys_votes2.values())
named_data[1:-1, 0] = list(sys_votes2.values())
named_data[0, 0] = ""
named_data[-1, 1:-1] = np.round(np.sum(data, axis=1), 2)
named_data[1:-1, -1] = np.round(np.sum(data, axis=0), 2)
named_data[-1, 0] = "TOT"
named_data[0, -1] = "TOT"
named_data[-1, -1] = np.round(np.sum(data), 2)


total = np.sum(data, axis=1).reshape(-1, 1)

with np.printoptions(precision=2, suppress=True):
    np.savetxt(sys.stdout, named_data, fmt='%-4s', delimiter=' | ')

print('')
print('')