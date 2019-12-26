import numpy as np
import random
import sys
import statistics
import time
from random import sample


class mapa:
    ciudad=np.zeros((2, 3))
    def __init__(self,nCiudades):
        self.ciudad=np.zeros((nCiudades, nCiudades),int)

    def generarMapa(self):
        j=1
        for i in range(int(sys.argv[1])):
            L=sample(range(1,100),int(sys.argv[1])-i)
            if (i==0):
                self.ciudad[i]=L
                continue
            for x in range(i):
                if (x<i):
                    L.insert(x,self.ciudad[x][j])
            if (j<int(sys.argv[1])):
                self.ciudad[j]=L
            j=j+1
        np.fill_diagonal(self.ciudad, 0)

    def mostrarMapa(self):
        for i in range(int(sys.argv[1])):
            if (i<=9):
                print("ciudad ",i, "" ,self.ciudad[i])
            else:
                print("ciudad ",i ,self.ciudad[i])


class geneticoHormigas:
    matrizFeromonas=np.zeros((3, 3))
    def __init__(self,nCiudades):
            self.matrizFeromonas = np.ones((nCiudades, nCiudades),int)
            np.fill_diagonal(self.matrizFeromonas, 0)

 


if __name__ == "__main__":
    print("Numero de ciudades a configurar",sys.argv[1])
    print("\n")
    map = mapa(int(sys.argv[1]))
    genetico = geneticoHormigas(int(sys.argv[1]))
    
    map.generarMapa()
    
