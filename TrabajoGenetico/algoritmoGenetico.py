import numpy as np
import random

class algoritmoGenetico ():
    alpha = 0.0

    def __init__ (self, _alpha):
        self.alpha = _alpha

    #Funcion que define la bondad de un individuo de la poblacion
    def fitness(self, number):
        self.alpha
        number = abs(number-self.alpha)
        return number


    #Funcion que compara la bondad de dos individuos x e y
    def compare(self, x, y):
        if self.fitness(x) < self.fitness(y) :
            rst = -1
        elif self.fitness(x) > self.fitness(y)  :
            rst = 1
        else :
            rst = 0
        return rst

    #Combina los individuos de dos en dos con un corte cut
    def reproduction(self, population):
        cut = 0.6
        #Elitismo: guardamos las dos mejores, no se tocan los dos primeros valores por ser los mejores
        for i in range(2, len(population)-2,2):
            p1 = population[i]
            p2 = population[i+1]
            population[i] = p1*cut + p2*(1-cut)
            population[i+1] = p1*(1-cut) + p2*(1-cut)

    #Función que muta un 5% de los individuos dividiendo su valor entre dos
    def mutation(self, population):
        for i in range(0,len(population)-1):
            if random.randint(0,19) == 1 :
                population[i] = population[i]/2


    #Función que indica si el algoritmo genético ha llegado a su fin.
    #Acaba cuando la media de fallo de los 10 primeros es menor que 0.1
    def end(self, population):
        avg = 0.0
        for i in range(0,10):
            avg = avg + fitness(population[i])
        avg = avg/10.0
        print ('La media es:', avg)
        if avg < 0.011:
            return True
        else:
            return False