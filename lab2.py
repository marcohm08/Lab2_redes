# Lab 1 Redes de Computadores
# Marco Hernandez 
# 19.318.862-1

import scipy
import scipy.io
import scipy.misc
from scipy.io import wavfile
from scipy import signal
from scipy.fftpack import fft2,fftshift
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
                    pix = kernel[n][m] * matrix[b-n][a-m]
                    aux.append(pix)
            aux = np.array(aux)
            value = np.sum(aux)
            if(value > 255):
                value = 255
            elif(value < 0):
                value = 0
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


    colstart = int(kernelCol/2)
    rowStart = int(kernelRow/2)
    for n in range(colstart,len(matrix) - colstart):
        for m in range(rowStart,len(matrix[0]) - rowStart):
            if(matrix[n][m] != convoluted[n-colstart][m-rowStart]):
                print("No se realiza la convolucion adecuadamente")
                return False
    
    print("Se lleva a cabo la convolucion correctamente")
    return True

class ImageObject:
    def __init__(self,path):
        self.name = path
        self.image = Image.open(path)
        self.imageMatrix = np.array(self.image)

    def convolveImage(self,kernel):
        result = convolve(self.imageMatrix,kernel)
        self.imageMatrix = result
    
    def fourierTransform(self, title, saveName):
        fourierData = fft2(self.imageMatrix)
        fourierOriginal = fftshift(fourierData)
        g = plt.figure()
        plt.imshow(np.log(abs(fourierOriginal)))
        if(title == "" or title == None):    
            plt.title("Transformada de Fourier de imagen")
        else:
            plt.title(title)
        if(saveName == "" or saveName == None):
            g.savefig("Espectrograma de imagen")
        else:
            g.savefig(saveName)
               
    def saveImage(self,name):
        self.image = Image.fromarray(self.imageMatrix.astype(np.uint8))
        self.image.save(name)


if __name__ == "__main__":
    

    realImage = ImageObject("lena512.bmp")
    gaussImage = copy.deepcopy(realImage)
    edgeImage = copy.deepcopy(realImage)
    ed2 = ImageObject("im1.png")

    #convolutionTest(realImage.imageMatrix,5,5)

    gaussKer= np.array([[1,4,6,4,1],
                        [4, 16, 24, 16, 4],
                        [6, 24, 36, 24, 6],
                        [4, 16, 24, 16, 4],
                        [1, 4, 6, 4, 1]])/256
    
    edgeKer= np.array([[1,2,0,-2,-1],
                        [1,2,0,-2,-1],
                        [1,2,0,-2,-1],
                        [1,2,0,-2,-1],
                        [1,2,0,-2,-1]])

    edgeKer2= np.array([[1,0,-1],
                        [2,0,-2],
                        [1,0,-1]])

    realImage.fourierTransform("Transformada de Fourier imagen original", "Fourier_Original")

    gaussImage.convolveImage(gaussKer)
    gaussImage.fourierTransform("Transformada de Fourier imagen filtro gaussiano", "Fourier_Gauss")
    gaussImage.saveImage("Gauss_image.png")

    edgeImage.convolveImage(edgeKer)
    edgeImage.fourierTransform("Transformada de Fourier iltro de bordes", "Fourier_Edge")
    edgeImage.saveImage("Edge_image.png")

    #ed2.convolveImage(edgeKer2)
    #ed2.saveImage("ed2.png")

    plt.show()






