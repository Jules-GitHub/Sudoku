from matplotlib.pyplot import fill_between
import tensorflow as tf
import numpy as np
from math import sqrt
from random import randint, shuffle


def init(n):
    return [[0 for j in range(n)] for i in range(n)]


def affichage_grille(grille):
    n = len(grille)
    r = int(sqrt(n))
    for i in range(n):
        for j in range(n):
            print(grille[i][j], end="")
            if (j%r==r-1):
                print(" ", end="")
        print()
        if (i%r==r-1):
            print()


def checkGrid(grille):
    n = len(grille)
    for i in range(n):
        for j in range(n):
            if grille[i][j]==0:
                return False
    return True


listeNombres = [1, 2, 3, 4, 5, 6, 7, 8, 9]


def fillGrid(grille):
    n = len(grille)
    r = int(sqrt(n))
    #Find next empty cell
    for i in range(0,n**2):
        row = i//n
        col = i%n
        if grille[row][col] == 0:
            shuffle(listeNombres)
            for value in listeNombres:
                #Check that this value has not already be used on this row
                if not(value in grille[row]):
                #Check that this value has not already be used on this column
                    if not value in [grille[k][col] for k in range(n)]:
                        #Identify which of the 9 squares we are working on
                        isquare = (row//r)*r
                        jsquare = (col//r)*r
                        square=[grille[k][jsquare : jsquare+r] for k in range(isquare, isquare+r)]
                        #Check that this value has not already be used on this 3x3 square
                        if not value in [el for ligne in square for el in ligne]:
                            grille[row][col] = value
                            if checkGrid(grille):
                                return True
                            else:
                                if fillGrid(grille):
                                    return True
            break
    grille[row][col] = 0


def deconstruit_un_coup(grille):
    n = len(grille)
    while True:
        i, j = randint(0,n-1), randint(0,n-1)
        if (grille[i][j] != 0):
            break
    move = grille[i][j]
    grille[i][j] = 0
    return (i, j, move)


def joue(grille, i, j, move):
    grille[i][j] = move


def creation_donnees(repetition, profondeurPartie, n):
    train_grilles, train_move = [], []
    for i in range(repetition):
        partie = init(n)
        fillGrid(partie)
        for j in range(min(profondeurPartie, n**2)):
            move = deconstruit_un_coup(partie)
            train_grilles.append(partie.copy())
            train_move.append(move)
    return train_grilles, train_move


def melange_jeu(grille, profondeur):
    n = len(grille)
    for i in range(min(profondeur, n**2)):
        deconstruit_un_coup(grille)


def train():
    model = tf.keras.Sequential()
    #Couches du modèle
    model.add(tf.keras.layers.Flatten(input_shape=(None, 32, 4, 4)))
    model.add(tf.keras.layers.Dense(256, activation=tf.math.sigmoid))
    model.add(tf.keras.layers.Dense(128, activation=tf.math.sigmoid))
    model.add(tf.keras.layers.Dense(128, activation=tf.math.sigmoid))
    model.add(tf.keras.layers.Dense(64, activation=tf.math.sigmoid))
    model.add(tf.keras.layers.Dense(64, activation=tf.math.sigmoid))
    model.add(tf.keras.layers.Dense(32, activation=tf.math.sigmoid))
    model.add(tf.keras.layers.Dense(32, activation=tf.math.sigmoid))
    model.add(tf.keras.layers.Dense(16, activation=tf.math.sigmoid))
    model.add(tf.keras.layers.Dense(16, activation=tf.math.sigmoid))
    model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))
    #Compilation
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=[])
    #Entrainement
    for i in range(20):
        print("n°",i)
        inData, outData = creation_donnees(100, 65, 4)
        model.fit(inData, outData, epochs=2)
    #On retourne le modèle entraîné
    return model


def simulation(model, melange):
    pass


modele = train()