import random
import sys
import statistics
import time
import algoritmoHormiga
import algoritmoGenetico
import mapa
from random import sample

class main():
    alpha = 0.0
    nCiudades = 6
    ciudades = type(mapa)
    hormiga = type(algoritmoHormiga)
    genetico = type(algoritmoGenetico)

    def __init__(self,_alpha,_nCiudades):
        self.alpha = _alpha
        self.nCiudades = _nCiudades
        self.ciudades = mapa.mapa(_nCiudades)
        self.hormiga = algoritmoHormiga.algoritmoHormiga(_nCiudades)
        self.genetico = algoritmoGenetico.algoritmoGenetico(_alpha)

    def hormigas(self):
        return self.hormiga

    def geneticos(self):
        return self.genetico

    def mapaCiudades(self):
        return self.ciudades
    


###############################################################
###############################################################
###############################################################
###############################################################
###############################################################
mainProgram = main(0.0,6)
mainProgram.mapaCiudades().mostrarMapa()
mainProgram.hormigas().mostrarMapaFeromonas()