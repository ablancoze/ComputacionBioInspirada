import numpy as np
import random
import sys
from random import sample

class mapa():
    ciudad=np.zeros((3, 3))
    nCiudades=3

    def __init__(self,_nCiudades):
        self.ciudad=np.zeros((_nCiudades, _nCiudades),int)
        self.nCiudades = _nCiudades
        self.generarMapa()

    #Permite generar un mapa de ciudades con costes simetricos
    def generarMapa(self):
        j=1
        for i in range(self.nCiudades):
            L=sample(range(1,100),self.nCiudades-i)
            if (i==0):
                self.ciudad[i]=L
                continue
            for x in range(i):
                if (x<i):
                    L.insert(x,self.ciudad[x][j])
            if (j<self.nCiudades):
                self.ciudad[j]=L
            j=j+1
        np.fill_diagonal(self.ciudad, 0)

    #Muestra el mapa generado.
    def mostrarMapa(self):
        for i in range(self.nCiudades):
            if (i<=9):
                print("ciudad ",i, "" ,self.ciudad[i])
            else:
                print("ciudad ",i ,self.ciudad[i])
        
        print("")
