# Lab 2 Redes de Computadores
# Marco Hernandez 
# 19.318.862-1

import scipy
import scipy.io
import scipy.misc
from scipy import signal
from scipy.fftpack import fft2,fftshift
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

import copy

import sys


#Function to fill with 0 borders of the convoluted matrix
#inputs: 
#   martrix: convoluted matrix
#   kernel: kernel matrix wich was used to convolute the original image
# output: convoluted matrix with pixel border values equal to 0
def fillWithZeros(matrix, kernel):
    cols = int(len(kernel)/ 2)
    rows = int(len(kernel[0])/ 2)
    for row in matrix:
        for i in range(0,rows):
            row.insert(0,0)
            row.append(0)
    
    zeroRow = [0 for i in range(0,len(matrix[0]))]
    for j in range(0,cols):
        matrix.insert(0,zeroRow)
        matrix.append(zeroRow)
    return matrix


# Convolution function, that makes the convolution between a matrix of values of an 
# image and a kernel, this kernel must have odd dimensions
# inputs
#   matrix: matrix with pixel values if an image
#   kernel: kernel matrix wich dimension must be odd to have a central element
# output: convoluted image matrix 
def convolve(matrix, kernel):
    convoluted = []

    rowlen = len(matrix[0])
    collen = len(matrix)
    kerrow = len(kernel[0])
    kercol = len(kernel)

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
    convoluted = fillWithZeros(convoluted,kernel)
    convoluted = np.array(convoluted)
    return convoluted

# Function to check that the convolution is done in the right way, to do this, we use the convolution with an identity kernel
# this kernel only has the central element of the matrix equal to 1, the rest of the numbers of the matrix is equal to 0
# the expected output is the same matrix of the input except of the bordes in wich case, the pixel values will be 0
# Input:
#   matrix: matrix wich will be convoluted
#   kernelCol: number of columns of the kenrl matrix, must be odd
#   kernelRow: number of rows of the kernel matrix, must be odd
# Output:
#   the convoluted matrix if the convolution is done in the right way
#   False if the convolution is wrong 
def convolutionTest(matrix,kernelCol,kernelRow):
    if(kernelCol % 2 == 0 or kernelRow % 2 == 0):
        print("Kernel dimensions must be odd")
        return False, matrix
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

    # construction of the neutral kernel
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

    # The pixel values that are compared between the original and the convoluted 
    # matrix are the pixels in the convoluted matrix than are not in the borders
    # of the image
    colstart = int(kernelCol/2)
    rowStart = int(kernelRow/2)
    for n in range(colstart,len(matrix) - colstart):
        for m in range(rowStart,len(matrix[0]) - rowStart):
            if(matrix[n][m] != convoluted[n][m]):
                print("The convolution fails")
                return False,matrix
    
    print("The convolution is successfull")
    return True,convoluted

# Class to save all elements of an image and the oprations that are done with it
class ImageObject:
    def __init__(self,path):
        self.name = path
        self.image = Image.open(path)
        self.imageMatrix = np.array(self.image)

    # Method to aplpy the convolution function defined before to the imageMatrix attribute
    def convolveImage(self,kernel):
        result = convolve(self.imageMatrix,kernel)
        self.imageMatrix = result
    
    #Mehod to apply the 2D fourier transform to the imageMatrix attribute
    def fourierTransform(self, title, saveName):
        fourierData = fft2(self.imageMatrix)
        fourierOriginal = fftshift(fourierData)
        g = plt.figure()
        plt.imshow(np.log(abs(fourierOriginal)))
        plt.colorbar()
        if(title == "" or title == None):    
            plt.title("Transformada de Fourier de imagen")
        else:
            plt.title(title)
        if(saveName == "" or saveName == None):
            g.savefig("Espectrograma de imagen")
        else:
            g.savefig(saveName)

    # Method to create an image from the imageMatrix attribute          
    def saveImage(self,name):
        self.image = Image.fromarray(self.imageMatrix.astype(np.uint8))
        self.image.save(name)


if __name__ == "__main__":
    

    realImage = ImageObject("lena512.bmp")
    testImage = copy.deepcopy(realImage)
    gaussImage = copy.deepcopy(realImage)
    edgeImage = copy.deepcopy(realImage)

    isRight,testImage.imageMatrix =  convolutionTest(testImage.imageMatrix,int(sys.argv[1]),int(sys.argv[2]))

    if(isRight == True):
        testImage.saveImage("Test_image.png")

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


    realImage.fourierTransform("Transformada de Fourier imagen original", "Fourier_Original")

    gaussImage.convolveImage(gaussKer)
    gaussImage.fourierTransform("Transformada de Fourier imagen filtro gaussiano", "Fourier_Gauss")
    gaussImage.saveImage("Gauss_image.png")

    edgeImage.convolveImage(edgeKer)
    edgeImage.fourierTransform("Transformada de Fourier Filtro de bordes", "Fourier_Edge")
    edgeImage.saveImage("Edge_image.png")

    plt.show()






