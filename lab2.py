# Lab 1 Redes de Computadores
# Marco Hernandez 
# 19.318.862-1

import scipy
import scipy.io
import scipy.misc
from scipy.io import wavfile
from scipy import signal
from scipy.fftpack import fft,fftfreq,ifft
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

import copy

import sys



# Parte 1: Definir funcion de convolucion, en el caso problematico de los bordes, estos valores seran recortados 

def convolve(matrix, kernel):
    convoluted = []

    rowlen = len(matrix[0])
    collen = len(matrix)
    kerrow = len(kernel[0])
    kercol = len(kernel[0])

    for b in range(kercol -1, collen):
        auxrow = []
        for a in range(kerrow -1, rowlen):
            aux = []
            for n in range(0,kercol):
                for m in range(0,kerrow):
                    pix = kernel[n][m] * matrix[b-n][a-n]
                    aux.append(pix)
            aux = np.array(aux)
            value = sum(aux)
            auxrow.append(value)
        convoluted.append(auxrow)
    convoluted = np.array(convoluted)
    return convoluted


def convolutionTest(matrix,kernelCol,kernelRow):
    testKernel = []
    i = 0
    j = 0
    rowCenter = 0
    colCenter = 0
    if(kernelCol % 2 == 0):
        colCenter = kernelCol / 2 - 1
    else:
        colCenter = int(kernelCol / 2) 
    if(kernelRow % 2 == 0):
        rowCenter = kernelRow / 2 - 1
    else:
        rowCenter = int(kernelRow / 2) 

    print(rowCenter)

    while(j < kernelCol):
        testRow = []
        while(i < kernelRow):
            if(i == rowCenter and j == colCenter):
                testRow.append(1)
            else:
                testRow.append(0)
            i+=1
        testRow = np.array(testRow)
        testKernel.append(testRow)
        i = 0
        j+=1
    
    testKernel = np.array(testKernel)
    convoluted = convolve(matrix, testKernel)
    print(convoluted)


    colstart = int(kernelCol/2)
    rowStart = int(kernelRow/2)
    print(len(convoluted[0]))
    for n in range(colstart,len(matrix) - 1):
        for m in range(rowStart,len(matrix[0]) - 1):
            if(matrix[n][m] != convoluted[n-colstart][m-rowStart]):
                print("No se realiza la convolucion adecuadamente")
                return False
    
    print("Se lleva a cabo la convolucion correctamente")
    return True

im = Image.open("Lab2_redes/lena512.bmp")
image_array = np.array(im)

result = convolutionTest(image_array,3,3)
#print(image_array)