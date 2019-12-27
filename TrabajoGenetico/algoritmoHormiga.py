import numpy as np
import random
import sys 
import math



class algoritmoHormiga ():
    matrizFeromonas=np.zeros((3, 3))
    nCiudades = 3

    def __init__ (self, _nCiudades):
        self.matrizFeromonas=np.zeros((_nCiudades, _nCiudades),int)
        self.nCiudades = _nCiudades
        np.fill_diagonal(self.matrizFeromonas, 0)

    #Muestra el mapa de feromonas generado
    def mostrarMapaFeromonas(self):
        for i in range(self.nCiudades):
            if (i<=9):
                print("ciudad ",i, "" ,self.matrizFeromonas[i])
            else:
                print("ciudad ",i ,self.matrizFeromonas[i])
        
        print("")

    # Elige un paso para una hormiga, teniendo en cuenta los valores
    # las feromonas y descartando los nodos ya visitados.
    def eligeNodo(self, ciudades, ferom, visitados):
        #Se calcula la tabla de pesos de cada ciudad
        listaValores = []
        disponibles = []
        actual = visitados[-1]

        # Influencia de cada valor (alfa: feromonas; beta: valor)
        alfa = 0.5
        beta = 0.5

        # El parámetro beta (peso de los valores) es 0.5 y alfa=1.0
        for i in range(len(ciudades)):
            if i not in visitados:
                fer  = math.pow((1.0 + ferom[actual][i]), alfa)
                peso = math.pow(1.0/ciudades[actual][i], beta) * fer
                disponibles.append(i)
                listaValores.append(peso)

        # Se elige aleatoriamente una de los nodos disponibles,
        # teniendo en cuenta su peso relativo.
        valor = random.random() * sum(listaValores)

        acumulado = 0.0
        i = -1
        while valor > acumulado:
            i += 1
            acumulado += listaValores[i]

        # i termina con el valor de la ciudad disponible siguiente
        return disponibles[i]

    # Genera una " hormiga " , que eligirá un camino (nodos que visita) teniendo en cuenta
    # los valores y los rastros de feromonas. Devuelve una tupla
    # con el camino (nodos visitados) y su longCamino (Suma de valores).
    def eligeCamino(self, ciudades, feromonas):
        # El nodo inicial siempre es el 0
        camino = [0]
        longCamino = 0

        # Elegir cada paso según los valores y las feromonas
        while len(camino) < len(ciudades):
            nodo = self.eligeNodo(ciudades, feromonas, camino)
            longCamino += ciudades[camino[-1]][nodo]
            camino.append(nodo)

        # Para terminar hay que volver al nodo de origen (0)
        longCamino += ciudades [camino[-1]][0]
        camino.append(0)

        return (camino, longCamino)

    # Actualiza la matriz de feromonas siguiendo el camino recibido
    def rastroFeromonas(self, feromonas, camino, dosis):
        for i in range (len(camino) - 1):
            feromonas[camino[i]][camino[i+1]] += dosis

    # Evapora todas las feromonas multiplicándolas por una constante
    # = 0.9 ( en otras palabras, el coefienciente de evaporación es 0.1)
    def evaporaFeromonas(self, feromonas):     
        for lista in feromonas:
            for i in range(len(lista)):
                lista[i] *= 0.9


    # algoritmo de la colonia de hormigas. Recibe una matriz de
    # distancias y devuelve una tupla con el mejor camino que ha 
    # obtenido (lista de índices) y su longitud
    def hormigas(self, ciudades, numHormigas, distMedia):
        # Primero se crea una matriz de feromonas vacía del mismo
        # tamaño que la de distancias
        n = len(ciudades)
        feromonas = [[0 for i in range(n)] for j in range(n)]
        
        # El mejor camino y su longitud (inicialmente "infinita")
        mejorCamino = []
        longMejorCamino = sys.maxsize
        
        # En cada iteración se genera una hormiga, que elige una camino,
        # y si es mejor que el mejor que teníamos, deja su rastro de
        # feromonas (mayor cuanto más corto sea el camino)
        for iter in range(numHormigas):
            (camino,longCamino) = self.eligeCamino(ciudades, feromonas)
            
            if longCamino <= longMejorCamino:
                mejorCamino     = camino
                longMejorCamino = longCamino
                
            self.rastroFeromonas(feromonas, camino, distMedia/longCamino)
            # En cualquier caso, las feromonas se van evaporando
            self.evaporaFeromonas(feromonas)
            
        # Se devuelve el mejor camino que se haya encontrado
        return (mejorCamino, longMejorCamino)
