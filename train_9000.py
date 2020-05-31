#!/usr/bin/python3
#----------------------------------------------------------------
# Mein Dynamisches Neuronales Netz in github2
# Dateiname: n5000.py       3 DNN-Areale
# R.J.Nickerl mit github
# 05.04.20 Python 3.8
#--------------------------------------------------------------
try:
    import tkinter as tk
except ImportError:
    import Tkinter as tk                        
from random import randint
import numpy
from numpy import asarray
from numpy import savetxt
from numpy import loadtxt
import math
import scipy.special
import matplotlib.pyplot as plt
import csv

import time
from time import *

Dateiname = strftime("%d-%m-%Y")
Datum = strftime("%d.%m.%Y")
Uhrzeit = strftime("%H:%M:%S")

#import RPi.GPIO as GPIO
#import datetime

# Dynamisches Neuronales Netz class definition
#+DNN1++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class dynNN1:
    def __init__(self,inputnodes, hiddennodes, outputnodes):
        # initialise the dynNN
        # set number of nodes in each input, hidden, output and daempfungs layer
        print("++++++++++++++++ Initialise DNN 1 ++++++++++++++++")
        self.inodes1 = inputnodes        # Anzahl Input Nodes
        self.hnodes1 = hiddennodes       # Anzahl Hidden Nodes
        self.dnodes1 = self.hnodes1      # Anzahl Damp Nodes (=Dämpfungsknoten)
        self.onodes1 = outputnodes       # Anzahl Output Nodes
        
        # link weight matrices: wih, who and dhh
        # weights inside the arrays are:
        # w_i_j, where link is from node i to node j in the next layer
        # d_i_j, where reverse link is from node i to node j in the same layer
        # w11 w21   d11 d21
        # w12 w22   d12 d22
        self.wih1  = numpy.random.normal(0.0, pow(self.hnodes1, -0.5), (self.hnodes1, self.inodes1))
        self.who1  = numpy.random.normal(0.0, pow(self.onodes1, -0.5), (self.onodes1, self.hnodes1))
        self.dihh1 = numpy.random.normal(0.0, pow(self.dnodes1, -0.5), (self.hnodes1, self.hnodes1))
        self.dohh1 = numpy.random.normal(0.0, pow(self.dnodes1, -0.5), (self.onodes1, self.onodes1))

        self.evo_wih1  = numpy.zeros( [self.hnodes1, self.inodes1] )
        self.evo_who1  = numpy.zeros( [self.onodes1, self.hnodes1] )
        self.evo_dihh1 = numpy.zeros( [self.hnodes1, self.hnodes1] )
        self.evo_dohh1 = numpy.zeros( [self.onodes1, self.onodes1] )      
    
        # Vector: Dihh[i] self.hidden_outputs werden random erstbesetzt
        # Vector: Dohh[i] self.output_outputs werden random erstbesetzt
        self.hidden_outputs1 = numpy.random.normal(0.0, pow(self.hnodes1, -0.5), (self.hnodes1))
        self.hidden_outputs1_stable = self.hidden_outputs1
        self.output_outputs1 = numpy.random.normal(0.0, pow(self.hnodes1, -0.5), (self.onodes1))
        self.output_outputs1_stable = self.output_outputs1

    def status1(self):
        # status dynNN
        print("++++++++++++++++ DNN 1 ++++++++++++++++")
        print("wih1= ")
        print(self.wih1.round(3))
        #print("who1= ")
        #print(self.who1.round(3))
        #print("dihh1= ")
        #print(self.dihh1.round(3))
        #print("dohh1= ")
        #print(self.dohh1.round(3))
        print()         
        pass

    def search(self, inputs_list):
        # search dynNN
        inputs = inputs_list
        print("..........SUCHE...............")
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih1, inputs)
        print(self.wih, inputs)
        pass

    def vektor_hi1(self, inputs_list):
        #Schritt 6: Vektor hi berechnen: hi = wih @ INPUT_akt[i]-Vektor + dihh @ self.hidden_outputs
        inputs = inputs_list
        self.hidden_inputs1 = numpy.dot(self.wih1, inputs) + numpy.dot(self.dihh1, self.hidden_outputs1)
        self.hidden_outputs1=self.hidden_inputs1
        return self.hidden_outputs1
        pass
    
    def sprung_antwort_hidden1(self, stufenwert):
        # Schritt 7: Sprungantwort der hidden Neuronen hs (s=Sprung) entspr. dem Stufenwer (stufen_wert) berechnen
        ##print("Schritt 7: Sprungantwort der hidden Neuronen")
        self.swert1 = stufenwert # Höhe des Schwellwertes der Nodes bei dem der Knoten "schaltet"
        for i in range(self.hnodes1):
            if self.hidden_outputs1[i] > self.swert1:
                self.hidden_outputs1[i] = 1
            else:
                self.hidden_outputs1[i] = 0
        return self.hidden_outputs1
        pass    

    def vektor_ho1(self):
        # Schritt 9: Vektor ho berechnen: ho = who @ self.hidden_outputs-Vektor + dohh @ self.output_outputs
        ##print("Schritt 9: Vektor ho berechnen")
        self.output_inputs1 = numpy.dot(self.who1, self.hidden_outputs1) + numpy.dot(self.dohh1, self.output_outputs1)
        self.output_outputs1=self.output_inputs1
        return self.output_outputs1
        pass

    def sprung_antwort_output1(self, stufenwert):
        # Schritt 10: Sprungantwort der output Neuronen o1 - on entspr. dem Stufenwert berechnen
        ##print("Schritt 9: Sprungantwort der output Neuronen")
        self.swert1 = stufenwert # Höhe des Schwellwertes der Nodes bei dem der Knoten "schaltet"
        for i in range(self.onodes1):
            if self.output_outputs1[i] > self.swert1:
                self.output_outputs1[i] = 1
            else:
                self.output_outputs1[i] = 0
        return self.output_outputs1
        pass

    def daempf_anpassung1(self):
        # Ziel ist ein chaotisches Schwingen (0,1,0,1,0,1,...) zu detektieren und zum Zeitpkt.=0 zu dämpfen
        # Dämpfungsnodes um 1 = einen Zeitpunkt verkleinern
        # Dämpfungswerte dhh anpassen. 
        # Dämpfungsmatrix erzeugen wenn Zeitwert z=0 dann dämpfen   
        print("ccccccccc daempfung_anpassung1 ccccccccc")
        for i in range(hidden_nodes):
            for k in range(hidden_nodes):
                if self.hs_alt1[i] != self.hs1[i]:
                    self.dhh1[i,k] = 1.25*self.dhh1[i,k]  # Detektion von 0,1,0,.. Feuern des Neurons hs[i]=>Chaos=>Erhöhung der Dämpfung          
                    pass
                if self.hs_alt1[i] == self.hs1[i]:
                    self.dhh1[i,k] = 0.75*self.dhh1[i,k]  # Detektion von 0,0,0, oder 1,1,1.. kein/immer Feuern des Neurons => Dämpfung auf 75% verkleiner          
                    pass
        pass

    def reset_train1(self):
        self.wih1  = self.wih1 - self.evo_wih1
        self.who1  = self.who1 - self.evo_who1
        self.dihh1 = self.dihh1 - self.evo_dihh1
        self.dohh1 = self.dohh1 - self.evo_dohh1    
        pass
    
    def train1(self):
        for evo_step in range (int(math.sqrt(self.inodes1))):       # Änderungen: wih1
            evo_x = numpy.random.rand()
            evo_y = numpy.random.rand()
            self.evo_wih1[int(evo_x*self.hnodes1), int(evo_y*self.inodes1)] = numpy.random.normal(0.0, pow(self.hnodes1, -0.5))
            self.evo_who1[int(evo_x*self.onodes1), int(evo_y*self.hnodes1)] = numpy.random.normal(0.0, pow(self.onodes1, -0.5))
            self.evo_dihh1[int(evo_x*self.hnodes1), int(evo_y*self.hnodes1)] = numpy.random.normal(0.0, pow(self.hnodes1, -0.5))
            self.evo_dohh1[int(evo_x*self.onodes1), int(evo_y*self.onodes1)] = numpy.random.normal(0.0, pow(self.onodes1, -0.5))

        self.wih1  = self.wih1 + self.evo_wih1
        self.who1  = self.who1 + self.evo_who1
        self.dihh1 = self.dihh1 + self.evo_dihh1
        self.dohh1 = self.dohh1 + self.evo_dohh1           
        pass

#+DNN2++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class dynNN2:
    def __init__(self,inputnodes, hiddennodes, outputnodes):
        print("++++++++++++++++ Initialise DNN 2 ++++++++++++++++")
        self.inodes2 = inputnodes        # Anzahl Input Nodes
        self.hnodes2 = hiddennodes       # Anzahl Hidden Nodes
        self.dnodes2 = self.hnodes2      # Anzahl Damp Nodes (=Dämpfungsknoten)
        self.onodes2 = outputnodes       # Anzahl Output Nodes       
        self.wih2 = numpy.random.normal(0.0, pow(self.hnodes2, -0.5), (self.hnodes2, self.inodes2))
        self.who2 = numpy.random.normal(0.0, pow(self.onodes2, -0.5), (self.onodes2, self.hnodes2))
        self.dihh2 = numpy.random.normal(0.0, pow(self.dnodes2, -0.5), (self.hnodes2, self.hnodes2))
        self.dohh2 = numpy.random.normal(0.0, pow(self.dnodes2, -0.5), (self.onodes2, self.onodes2))
        self.evo_wih2  = numpy.zeros( [self.hnodes2, self.inodes2] )
        self.evo_who2  = numpy.zeros( [self.onodes2, self.hnodes2] )
        self.evo_dihh2 = numpy.zeros( [self.hnodes2, self.hnodes2] )
        self.evo_dohh2 = numpy.zeros( [self.onodes2, self.onodes2] )              
        self.hidden_outputs2 = numpy.random.normal(0.0, pow(self.hnodes2, -0.5), (self.hnodes2))
        self.hidden_outputs2_stable = self.hidden_outputs2
        self.output_outputs2 = numpy.random.normal(0.0, pow(self.hnodes2, -0.5), (self.onodes2))
        self.output_outputs2_stable = self.output_outputs2
        print("++++++++++++++++ DNN 2 ++++++++++++++++")

    def status2(self):
        print("++++++++++++++++ DNN 2 ++++++++++++++++")
        print("wih2= ")
        print(self.wih2.round(3))
        #print("who2= ")
        #print(self.who2.round(3))
        #print("dihh2= ")
        #print(self.dihh2.round(3))
        #print("dohh2= ")
        #print(self.dohh2.round(3))
        print()         
        pass
    def search(self, inputs_list):
        inputs = inputs_list
        hidden_inputs = numpy.dot(self.wih2, inputs)
        print(self.wih, inputs)
        pass
    def vektor_hi2(self, inputs_list):
        inputs = inputs_list
        self.hidden_inputs2 = numpy.dot(self.wih2, inputs) + numpy.dot(self.dihh2, self.hidden_outputs2)
        self.hidden_outputs2=self.hidden_inputs2
        return self.hidden_outputs2
        pass    
    def sprung_antwort_hidden2(self, stufenwert):
        self.swert2 = stufenwert # Höhe des Schwellwertes der Nodes bei dem der Knoten "schaltet"
        for i in range(self.hnodes2):
            if self.hidden_outputs2[i] > self.swert2:
                self.hidden_outputs2[i] = 1
            else:
                self.hidden_outputs2[i] = 0
        return self.hidden_outputs2
        pass    
    def vektor_ho2(self):
        self.output_inputs2 = numpy.dot(self.who2, self.hidden_outputs2) + numpy.dot(self.dohh2, self.output_outputs2)
        self.output_outputs2=self.output_inputs2
        return self.output_outputs2
        pass
    def sprung_antwort_output2(self, stufenwert):
        self.swert2 = stufenwert # Höhe des Schwellwertes der Nodes bei dem der Knoten "schaltet"
        for i in range(self.onodes2):
            if self.output_outputs2[i] > self.swert2:
                self.output_outputs2[i] = 1
            else:
                self.output_outputs2[i] = 0
        return self.output_outputs2
        pass
    def daempf_anpassung2(self):
        for i in range(hidden_nodes):
            for k in range(hidden_nodes):
                if self.hs_alt2[i] != self.hs2[i]:
                    self.dhh2[i,k] = 1.25*self.dhh2[i,k]  # Detektion von 0,1,0,.. Feuern des Neurons hs[i]=>Chaos=>Erhöhung der Dämpfung          
                    pass
                if self.hs_alt2[i] == self.hs2[i]:
                    self.dhh2[i,k] = 0.75*self.dhh2[i,k]  # Detektion von 0,0,0, oder 1,1,1.. kein/immer Feuern des Neurons => Dämpfung auf 75% verkleiner          
                    pass
        return self.hs2
        pass
    
    def reset_train2(self):
        self.wih2  = self.wih2 - self.evo_wih2
        self.who2  = self.who2 - self.evo_who2
        self.dihh2 = self.dihh2 - self.evo_dihh2
        self.dohh2 = self.dohh2 - self.evo_dohh2    
        pass
    
    def train2(self):
        for evo_step in range (int(math.sqrt(self.inodes2))):       # Änderungsmatrix
            evo_x = numpy.random.rand()
            evo_y = numpy.random.rand()
            self.evo_wih2[int(evo_x*self.hnodes2), int(evo_y*self.inodes2)] = numpy.random.normal(0.0, pow(self.hnodes2, -0.5))
            self.evo_who2[int(evo_x*self.onodes2), int(evo_y*self.hnodes2)] = numpy.random.normal(0.0, pow(self.onodes2, -0.5))
            self.evo_dihh2[int(evo_x*self.hnodes2), int(evo_y*self.hnodes2)] = numpy.random.normal(0.0, pow(self.hnodes2, -0.5))
            self.evo_dohh2[int(evo_x*self.onodes2), int(evo_y*self.onodes2)] = numpy.random.normal(0.0, pow(self.onodes2, -0.5))

        self.wih2  = self.wih2 + self.evo_wih2
        self.who2  = self.who2 + self.evo_who2
        self.dihh2 = self.dihh2 + self.evo_dihh2
        self.dohh2 = self.dohh2 + self.evo_dohh2           
        pass

#+DNN3++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class dynNN3:
    def __init__(self,inputnodes, hiddennodes, outputnodes):
        print("++++++++++++++++ Initialise DNN 3 ++++++++++++++++")
        self.inodes3 = inputnodes        # Anzahl Input Nodes
        self.hnodes3 = hiddennodes       # Anzahl Hidden Nodes
        self.dnodes3 = self.hnodes3      # Anzahl Damp Nodes (=Dämpfungsknoten)
        self.onodes3 = outputnodes       # Anzahl Output Nodes       
        self.wih3 = numpy.random.normal(0.0, pow(self.hnodes3, -0.5), (self.hnodes3, self.inodes3))
        self.who3 = numpy.random.normal(0.0, pow(self.onodes3, -0.5), (self.onodes3, self.hnodes3))
        self.dihh3 = numpy.random.normal(0.0, pow(self.dnodes3, -0.5), (self.hnodes3, self.hnodes3))
        self.dohh3 = numpy.random.normal(0.0, pow(self.dnodes3, -0.5), (self.onodes3, self.onodes3))
        self.evo_wih3  = numpy.zeros( [self.hnodes3, self.inodes3] )
        self.evo_who3  = numpy.zeros( [self.onodes3, self.hnodes3] )
        self.evo_dihh3 = numpy.zeros( [self.hnodes3, self.hnodes3] )
        self.evo_dohh3 = numpy.zeros( [self.onodes3, self.onodes3] )          
        self.hidden_outputs3 = numpy.random.normal(0.0, pow(self.hnodes3, -0.5), (self.hnodes3))
        self.hidden_outputs3_stable = self.hidden_outputs3
        self.output_outputs3 = numpy.random.normal(0.0, pow(self.hnodes3, -0.5), (self.onodes3))
        self.output_outputs3_stable = self.output_outputs3
 
    def status3(self):
        print("++++++++++++++++ DNN 3 ++++++++++++++++")
        print("wih3= ")
        print(self.wih3.round(3))
        #print("who3= ")
        #print(self.who3.round(3))
        #print("dihh3= ")
        #print(self.dihh3.round(3))
        #print("dohh3= ")
        #print(self.dohh3.round(3))
        print()         
        pass
    
    def search(self, inputs_list):
        inputs = inputs_list
        hidden_inputs = numpy.dot(self.wih3, inputs)
        print(self.wih, inputs)
        pass
    
    def vektor_hi3(self, inputs_list):
        inputs = inputs_list
        self.hidden_inputs3 = numpy.dot(self.wih3, inputs) + numpy.dot(self.dihh3, self.hidden_outputs3)
        self.hidden_outputs3 = self.hidden_inputs3
        return self.hidden_outputs3
        pass
    
    def sprung_antwort_hidden3(self, stufenwert):
        self.swert3 = stufenwert # Höhe des Schwellwertes der Nodes bei dem der Knoten "schaltet"
        for i in range(self.hnodes3):
            if self.hidden_outputs3[i] > self.swert3:
                self.hidden_outputs3[i] = 1
            else:
                self.hidden_outputs3[i] = 0
        return self.hidden_outputs3
        pass
    
    def vektor_ho3(self):
        self.output_inputs3 = numpy.dot(self.who3, self.hidden_outputs3) + numpy.dot(self.dohh3, self.output_outputs3)
        self.output_outputs3=self.output_inputs3
        return self.output_outputs3
        pass
    
    def sprung_antwort_output3(self, stufenwert):
        self.swert3 = stufenwert # Höhe des Schwellwertes der Nodes bei dem der Knoten "schaltet"
        for i in range(self.onodes3):
            if self.output_outputs3[i] > self.swert3:
                self.output_outputs3[i] = 1
            else:
                self.output_outputs3[i] = 0   
        return self.output_outputs3
        pass
    
    def daempf_anpassung3(self):
        for i in range(hidden_nodes):
            for k in range(hidden_nodes):
                if self.hs_alt3[i] != self.hs3[i]:
                    self.dhh3[i,k] = 1.25*self.dhh3[i,k]  # Detektion von 0,1,0,.. Feuern des Neurons hs[i]=>Chaos=>Erhöhung der Dämpfung          
                    pass
                if self.hs_alt3[i] == self.hs3[i]:
                    self.dhh3[i,k] = 0.75*self.dhh3[i,k]  # Detektion von 0,0,0, oder 1,1,1.. kein/immer Feuern des Neurons => Dämpfung auf 75% verkleiner          
                    pass
        return self.hs3
        pass
    
    def reset_train3(self):
        self.wih3  = self.wih3 - self.evo_wih3
        self.who3  = self.who3 - self.evo_who3
        self.dihh3 = self.dihh3 - self.evo_dihh3
        self.dohh3 = self.dohh3 - self.evo_dohh3    
        pass
    
    def train3(self):
        for evo_step in range (int(math.sqrt(self.inodes3))):       # Änderungen: wih1
            evo_x = numpy.random.rand()
            evo_y = numpy.random.rand()
            self.evo_wih3[int(evo_x*self.hnodes3), int(evo_y*self.inodes3)] = numpy.random.normal(0.0, pow(self.hnodes3, -0.5))
            self.evo_who3[int(evo_x*self.onodes3), int(evo_y*self.hnodes3)] = numpy.random.normal(0.0, pow(self.onodes3, -0.5))
            self.evo_dihh3[int(evo_x*self.hnodes3), int(evo_y*self.hnodes3)] = numpy.random.normal(0.0, pow(self.hnodes3, -0.5))
            self.evo_dohh3[int(evo_x*self.onodes3), int(evo_y*self.onodes3)] = numpy.random.normal(0.0, pow(self.onodes3, -0.5))

        self.wih3  = self.wih3 + self.evo_wih3
        self.who3  = self.who3 + self.evo_who3
        self.dihh3 = self.dihh3 + self.evo_dihh3
        self.dohh3 = self.dohh3 + self.evo_dohh3           
        pass

##Ende class DNN###################################################################################################################################

def synchron_calc():
    for zykl in range(alpha_laenge):            # Detektion jeder einzelnen Input-Zahl über die gesamte alpha-laenge auf 01-Flanken
        for out_n in range(output_nodes):
            if C1[out_n,zykl-1] == 0:
                if C1[out_n,zykl] == 1:         # Detektion von 01-Flanken
                    C1_edge[out_n,zykl-1] = 1   # 01-Flanken in C1_edge "sammeln"
            if C2[out_n,zykl-1] == 0:
                if C2[out_n,zykl] == 1:
                    C2_edge[out_n,zykl-1] = 1
            if C3[out_n,zykl-1] == 0:
                if C3[out_n,zykl] == 1:
                    C3_edge[out_n,zykl-1] = 1        
    C_sumedge = C1_edge + C2_edge + C3_edge     # Summen-Vektor über alle Flanken aller 3 Areale
    for zykl in range(alpha_laenge):        
            for out_n in range(output_nodes):
                if C_sumedge[out_n,zykl] == 1:
                    C_sumedge[out_n,zykl] = 0   # "Einzelflanken" werden im Vektor auf Null gesetzt
                Ergebnis[out_n] = Ergebnis[out_n] + C_sumedge[out_n,zykl]   # Aus den einzelnen Vektoren wird die Flankenmatrix Ergebnis[out_n] aufgebaut
    #for out_n in range(output_nodes):
        #Ergebnis[out_n] = Ergebnis[out_n] / numpy.sum(Ergebnis)             # Das Ergebnis wir auf das Summenergebnis "normiert"          
     #   Ergebnis[out_n] = Ergebnis[out_n] / numpy.sum(Ergebnis)             # Das Ergebnis wir auf das Summenergebnis "normiert"          
    return Ergebnis
    pass

def zeichne_output():
    fig, axs = plt.subplots(3, 3)
    axs[0, 0].imshow( B1, interpolation="nearest") # statt "spline16"
    axs[0, 0].set_title('*** B1 ***')
    axs[0, 1].imshow( B2, interpolation="nearest") # statt "spline16"
    axs[0, 1].set_title('*** B2 ***')
    axs[0, 2].imshow( B3, interpolation="nearest") # statt "spline16"
    axs[0, 2].set_title('***B3 ***')
    axs[1, 0].imshow( C1, interpolation="nearest") # statt "spline16"
    axs[1, 0].set_title('*** C1 ***')
    axs[1, 1].imshow( C2, interpolation="nearest") # statt "spline16"
    axs[1, 1].set_title('*** C2 ***')
    axs[1, 2].imshow( C3, interpolation="nearest") # statt "spline16"
    axs[1, 2].set_title('*** C3 ***')
    axs[2, 0].imshow( C1_edge, interpolation="nearest") # statt "spline16"
    axs[2, 0].set_title('*** C1_edge ***')
    axs[2, 1].imshow( C2_edge, interpolation="nearest") # statt "spline16"
    axs[2, 1].set_title('*** C2_edge ***')
    axs[2, 2].imshow( C3_edge, interpolation="nearest") # statt "spline16"
    axs[2, 2].set_title('*** C3_edge ***')
    plt.show()
    
def store_new(): # Die Dateien z.B.: "wih1.csv". usw. werden von python neu angelegt oder überschrieben wenn schon vorhanden
    savetxt('C:/Users/Rolan/1_GIT/DNN/DNN_data/wih1.csv', n.wih1, delimiter=',')
    savetxt('C:/Users/Rolan/1_GIT/DNN/DNN_data/wih2.csv', m.wih2, delimiter=',') 
    savetxt('C:/Users/Rolan/1_GIT/DNN/DNN_data/wih3.csv', o.wih3, delimiter=',')
    savetxt('C:/Users/Rolan/1_GIT/DNN/DNN_data/who1.csv', n.wih1, delimiter=',')
    savetxt('C:/Users/Rolan/1_GIT/DNN/DNN_data/who2.csv', m.wih2, delimiter=',') 
    savetxt('C:/Users/Rolan/1_GIT/DNN/DNN_data/who3.csv', o.wih3, delimiter=',')    
    savetxt('C:/Users/Rolan/1_GIT/DNN/DNN_data/dihh1.csv', n.dihh1, delimiter=',')
    savetxt('C:/Users/Rolan/1_GIT/DNN/DNN_data/dihh2.csv', m.dihh2, delimiter=',')
    savetxt('C:/Users/Rolan/1_GIT/DNN/DNN_data/dihh3.csv', o.dihh3, delimiter=',')
    savetxt('C:/Users/Rolan/1_GIT/DNN/DNN_data/dohh1.csv', n.dohh1, delimiter=',') 
    savetxt('C:/Users/Rolan/1_GIT/DNN/DNN_data/dohh2.csv', m.dohh2, delimiter=',') 
    savetxt('C:/Users/Rolan/1_GIT/DNN/DNN_data/dohh3.csv', o.dohh3, delimiter=',') 

    #data = loadtxt('C:/Users/Rolan/1_GIT/DNN/DNN_data/wih1.csv', delimiter=',')
    # print the array
    #print(data)

    #DNN_data = open('C:\\Users\\Rolan\\1_GIT\\DNN_csv.txt','a') # Speicherung auf PC ("\\" = "/")
    #DNN_data = open('C:/Users/Rolan/1_GIT/DNN_csv.txt','a') # Speicherung auf PC
    #DNN_data = open('H:/DNN_data/DNN_DATA.txt','w') # Speicherung auf Stick: "NEUES_LEBEN/DNN_data/DNN_DATA.txt"
    #DNN_data.write(Datum + ' ' + Uhrzeit + str(n.wih1) + str(m.wih2) + "\n")
    #DNN_data.write(Datum + ' ' + Uhrzeit + str( [n.wih1] )+ str( [m.wih2] ) + str(o.wih3) + "\n")
    #DNN_data.close()
    pass



    # query dynNN
    
#####################################################################
# Schritt 0: Initialisieren von allen Arealen des DNN (Gewichte, Dämpfung,...  #
input_nodes = 784               # MNIST Datensatz 28x28+1=785 Pixel (1 Pixel für das target)
hidden_nodes = 199  
output_nodes = 10               # Detektion von 10 Ziffern aus dem MNIST Datensatz
stufen_wert = 1                 # Stufenwert der Sprungfunktion
alpha_takt = 30                 # Anzahl der Takte (=Schwingungsdauer); der Alpha-Welle = 30; Betha-Welle = 90Takte
alpha_laenge = 70               # Länge der alpha-Welle (Anzahl der Lerndurchläufe)
alpha_step = 0                  # der Startwert der Alpha-Welle
alpha_bias = 0                  # Verschiebung der sin-Welle in den Plus-Bereich (1) + etwas höher (0.2)
Ergebnis_sum = 0                # Anmeldung der Gesamtsumme aller Flanken
generation_max = 200000              # Anzahl der Generationen für die Evolution
Ergebnis_sum_alt = 100000          # Gegriffener Anfangswert für das allererste Ergebnis_sum
evo_anzahl = 10                 # Anzahl der Werte welche pro Generation in den Gewichtsmatrizen und Dämpf. vektoren geändert werden
evo_breite = 10                 # Standardabweichung der gaußschen Normalverteilung der Zufallswerte

# create instance of 3 dynneural network areals (n,m,o)
n = dynNN1(input_nodes, hidden_nodes, output_nodes)
m = dynNN2(input_nodes, hidden_nodes, output_nodes)
o = dynNN3(input_nodes, hidden_nodes, output_nodes)
n.status1()
#m.status2()
#o.status3()

print("Anzahl input nodes= ", input_nodes+1)
print("Anzahl hidden nodes= ", hidden_nodes+1)
print("Anzahl output nodes= ", output_nodes+1)
print("alpha_takt= ", alpha_takt)
print("alpha_laenge= ", alpha_laenge)
print("alpha_step= ", alpha_step)
print("alpha_bias= ", alpha_bias)

# load MNIST data csv-file from ".../1_GIT/DNN/mnist_dataset/mnist_train_100.csv" into a list
training_data_file = open("mnist_dataset/mnist_train_100.csv", 'r')
#training_data_file = open("mnist_dataset/mnist_test_10.csv", 'r') ### erst einmal mit weniger (10) Datensätzen trainieren
training_data_list = training_data_file.readlines()
training_data_file.close()
print()

###############################################
######HAUPTPROGRAMM SCHLEIFE 1#################
###############################################
# Schritt 1: Schleife über Anlegen aller Zahlen aus der MNIST Trainingsdatei

for generation in range(generation_max):
    Ergebnis_sum = 0
    for record in training_data_list:
        
        # Vektor: C und B für das numpy array definieren
        C1 = numpy.zeros( [output_nodes,alpha_laenge,] )
        C2 = numpy.zeros( [output_nodes,alpha_laenge,] )
        C3 = numpy.zeros( [output_nodes,alpha_laenge,] )
        C1_edge = numpy.zeros( [output_nodes,alpha_laenge,] )
        C2_edge = numpy.zeros( [output_nodes,alpha_laenge,] )
        C3_edge = numpy.zeros( [output_nodes,alpha_laenge,] )
        B1 = numpy.zeros( [output_nodes,alpha_laenge,] )
        B2 = numpy.zeros( [output_nodes,alpha_laenge,] )
        B3 = numpy.zeros( [output_nodes,alpha_laenge,] )
        DNN = numpy.zeros( [3, output_nodes, alpha_takt] )
        Ergebnis = numpy.zeros( [output_nodes] )

        n.hidden_outputs1 = n.hidden_outputs1_stable    # zurücksetzen der Vektoren hidden_nodes und output_nodes auf Anfangswerte
        m.hidden_outputs2 = m.hidden_outputs2_stable
        o.hidden_outputs3 = o.hidden_outputs3_stable
        n.output_outputs1 = n.output_outputs1_stable
        m.output_outputs2 = m.output_outputs2_stable
        o.output_outputs3 = o.output_outputs3_stable
            
        #print("Angelegtes Trainingsmuster: ", record)
        #print(training_data_list[1])   # Ausdruck beginnend ab 2tem Zeichen, an erster Stelle steht die korrekte Zahl!
        # split the records by ',' commas
        all_values = record.split(",")
        # scale and shift the inputs statt Null -> 0.01, statt 255 -> 0.99
        INPUT_akt = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # create the target output values (all 0.01, except the desired label which is 0.99)
        targets = numpy.zeros(output_nodes) +0.01
        # all values[0] is the target label for this record
        targets[int(all_values[0])] = 0.99
        #print("targets= ", targets)
                
        # Schritt 2: Schleife Antworten über die alpha-WellenLänge (Lernzyklen)
        for zykl in range(alpha_laenge): # input_times = Anzahl der Durchläufe

            # Schritt 3: Vektor hi berechnen: hi = wih @ INPUT_akt[i]-Vektor + dihh @ self.hidden_outputs
            n.vektor_hi1(INPUT_akt)
            m.vektor_hi2(INPUT_akt)
            o.vektor_hi3(INPUT_akt)

            # Schritt 4: Sprungantwort der hidden Neuronen entspr. dem Stufenwer (stufen_wert) berechnen
            n.sprung_antwort_hidden1(stufen_wert)
            m.sprung_antwort_hidden2(stufen_wert)
            o.sprung_antwort_hidden3(stufen_wert)

            # Schritt 5: dynamische Anpassung der Dämpfungsgewichte di
        
            # Schritt 6: Vektor ho berechnen: ho = who @ self.hidden_outputs-Vektor + dohh @ self.output_outputs
            n.vektor_ho1()
            m.vektor_ho2()
            o.vektor_ho3()
            for outpn in range(output_nodes):
                B1[outpn,zykl]=n.output_outputs1[outpn]
                B2[outpn,zykl]=m.output_outputs2[outpn]
                B3[outpn,zykl]=o.output_outputs3[outpn]
                


            # Schritt 7: Sprungantwort der output Neuronen o1 - on entspr. dem Stufenwert berechnen
            n.sprung_antwort_output1(stufen_wert)
            m.sprung_antwort_output2(stufen_wert)
            o.sprung_antwort_output3(stufen_wert)
         
            # Schritt 8: Alpha Modellierung des Stufenwertes
            # Refraktionszeit = 3ms; alpha-Welle ca 100ms (=10Hz); ß-Welle ca. 30Hz => Lernschritte zwischen 30 - 90 Takten (alpha - ß Welle)
            # Erklärung: z%alpha_takt = z modulo alpha_takt => gibt immer nur den Rest der Division zurück
            stufen_wert = numpy.sin(2*math.pi/alpha_takt*(zykl%alpha_takt))+alpha_bias # z%alpha_takt = z modulo alpha_takt

            for outpn in range(output_nodes):
                C1[outpn,zykl]=n.output_outputs1[outpn]
                C2[outpn,zykl]=m.output_outputs2[outpn]
                C3[outpn,zykl]=o.output_outputs3[outpn]
                


    # Schritt 9: Synchronisation pro Buchstabe über alle Areale berechnen und wenn:
    # a) Synchron.Ergebnis DNNneu besser als DNN alt dann DNNneu behalten sonst
    # b) DNNalt behalten und die Gewichte der Verknüpfungen hi und oi per Zufall und die Dämpfungen di und do neu setzen (Gaußsche Normalvert.).

        synchron_calc()
        #print(" Ergebnis= ", Ergebnis.round(2)," Summe= ", SUM.round(3))
        #print()
        #print("targets jetztttt= ", targets)
        #print("Ergebnis jetzttt= ", Ergebnis.round(2))
        Ergebnis = abs(Ergebnis - targets)                  # "Abstand" (abs) des "normierten" Ergebnises vom gewünschten <targets>-Ergebnis berechnen 
        #print("Ergebnis   minus= ", Ergebnis.round(2))
        Ergebnis_sum = Ergebnis_sum + numpy.sum(Ergebnis)   # Summe aller Abstände = Synchronisations-Güte für die aktuelle Zahl (=record) des DNNs 
    #print("Ergebnis_sum= ", Ergebnis_sum.round(3))          # Gesamtsumme aller Abstände = Synchronisations-Güte für alle Zahlen (=alle records) des DNNs 

    #zeichne_output()  # graphische Ausgabe der Ergebnisse

    if Ergebnis_sum < Ergebnis_sum_alt:   #Wenn das neue DNN bessser ist als das alte, dann Speicherung der neuen Gewichtsmatrizen (wihx, Dhox)und der Dämpfungsvektoren (Dihhx, Dohhx)
        Ergebnis_sum_alt = Ergebnis_sum
        print("Generation: ", generation, " Ergebnis_sum (neu)= ", Ergebnis_sum_alt.round(4))
        store_new()
        #n.store_new1()
        #m.store_new2()
        #o.store_new3()
        #n.status1()
        #m.status2()
        #o.status3()
    else:
        n.reset_train1()
        m.reset_train2()
        o.reset_train3()
    
    # Schritt 11: Trainieren (=evolutives Ändern) der Gewichtsmatrizen (wihx, Dhox)und der Dämpfungsvektoren (Dihhx, Dohhx)
    n.train1()
    m.train2()
    o.train3()      
     
    
    









    
        
        
    
