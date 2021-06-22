import numpy
import cv2 as openCV
from sklearn.cluster import DBSCAN

# carregar e preparar imagem
imagem_original = openCV.imread("imagens/parana_clube.jpg")  # padrão BGR
openCV.imshow("Imagem Original", imagem_original)

# vetorizar imagem
vetor_imagem = imagem_original.reshape((-1, 3))
vetor_imagem = numpy.float32(vetor_imagem)

# rodar dbscan
epsilon = 10
numero_minimo_de_amostras = 10
dbscan = DBSCAN(eps=epsilon, min_samples=numero_minimo_de_amostras)
dbscan.fit(vetor_imagem)
categorias = dbscan.labels_

# numero de clusters e pontos de ruido
numero_de_clusters = len(set(categorias)) - (1 if -1 in categorias else 0)
numero_de_ruidos = list(categorias).count(-1)

# converter vetores para formato da imagem original
imagem_ruido_x_categorizado = numpy.uint8(categorias.reshape(imagem_original.shape[:2]))

# mostrar imagem segmentada
openCV.imshow('Imagem Segmentada', imagem_ruido_x_categorizado)

# encerrar
openCV.waitKey(0)
openCV.destroyAllWindows()
