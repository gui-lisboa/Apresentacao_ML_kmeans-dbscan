import numpy
import cv2 as openCV
import matplotlib.pyplot as pyplot

# carregar e preparar imagem
imagem_original = openCV.imread("imagens/parana_clube.jpg")  # padrão BGR
openCV.imshow("Imagem Original", imagem_original)

# vetorizar imagem
vetor_imagem = imagem_original.reshape((-1, 3))
vetor_imagem = numpy.float32(vetor_imagem)

# vizualização dos pontos da imagem_vetorizada
azul = vetor_imagem[:, 0]
verde = vetor_imagem[:, 1]
vermelho = vetor_imagem[:, 2]
figura = pyplot.figure()
grafico = figura.add_subplot(projection='3d')
grafico.set_title("Pontos no espaço BGR")
grafico.set_xlabel('Azul')
grafico.set_ylabel('Verde')
grafico.set_zlabel('Vermelho')
grafico.scatter(azul, verde, vermelho, color="#000000")
pyplot.show()

# definir parametros e rodar K-means
# medida_de_compactacao = ∑i∥samplesi−centerslabelsi∥2
# ref: https://docs.opencv.org/master/d5/d38/group__core__cluster.html#ga9a34dc06c6ec9460e90860f15bcd2f88
numero_de_clusters = 3
criterios = (
    openCV.TermCriteria_EPS + openCV.TermCriteria_MAX_ITER,
    10,
    0.5
)
tentativas = 10
medida_de_compactacao, categorias, centroides = openCV.kmeans(
    vetor_imagem,
    numero_de_clusters,
    None,
    criterios,
    10,
    openCV.KMEANS_RANDOM_CENTERS
)

# converter vetores para formato da imagem original
centroides = numpy.uint8(centroides)
imagem_segmentada = centroides[categorias.flatten()]
imagem_segmentada = imagem_segmentada.reshape((imagem_original.shape))

# mostrar imagem segmentada
openCV.imshow('Imagem Segmentada', imagem_segmentada)

# encerrar
openCV.waitKey(0)
openCV.destroyAllWindows()
