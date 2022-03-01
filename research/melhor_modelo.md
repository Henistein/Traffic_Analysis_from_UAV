# R-CNN

Temos algoritmos de dois estágios representados por arquiteturas como R-CNN, Fast R-CNN e Faster R-CNN. Estes são realizados em duas etapas, primeiro a busca seletiva para gerar a proposta de região e, em seguida a classificação e regressão na proposta de região. Este método tem alta precisão, mas também limita a velocidade de detecção.

O modelo de R-CNN  consiste em:
  - Recebem uma imagem como entrada 
  - Extraem cerca de 2000 propostas de regiões
  - Processam as características de cada proposta usando uma larga rede neuronal convulational (CNN)
  - Classificam cada região usando SVMs.

source(https://arxiv.org/pdf/1311.2524.pdf)
---------------------------------------------

O modelo R-CNN é lento pois executa uma passagem convulacional para cada proposta de objeto, sem partilhar poder computational.
Para resolver este problema foi criado posteriormente um modelo Fast R-CNN este intoduz várias melhorias e inovações na velocidade de treino e teste enquanto aumenta também a taxa de precisão. Fast R-CNN é 9x mais rápida que a R-CNN e alcança uma maior mAP no PASCAL VOC 2012.

source(https://arxiv.org/pdf/1504.08083.pdf)

Resumo comparativo entre R-CNN e Fast R-CNN:
'''
Faster R-CNN is an object detection algorithm proposed by Ren et al. in 2015Error!
Reference source not found., consisting of four parts: feature extraction network,
region proposal network, ROI Pooling, and a fully connected layer. The overall
detection process is shown in Figure 1. Faster R-CNN is a modified version of
R-CNN and Fast R-CNN algorithms. The difference between the two is that the Faster
R-CNN algorithm avoids the computationally expensive selective search algorithm
and uses the RPN to generate candidate regions instead. This algorithm calculates the
features of the whole image at once and thus does not involve repeated calculations,
which greatly improves the detection speed of Faster R-CNN. 
'''

# SSD
Este modelo baseia-se no mecanismo do Faster R-CNN e na estrutura end-to-end do YOLO, na qual a classificação de objetos e regressão de localização são realizadas de forma direta no estágio da convulução.
source(https://assets.researchsquare.com/files/rs-668895/v1_covered.pdf?c=1631875157)
A rede principal do algoritmo de SSD usa uma rede VGG-16 como backbone e modifica-a fazendo uma alteração na ultimas duas fully connected layers com camadas convulacionais, enquanto adiciona também quatro camadas convulacionais depois para finalmente realizar a extração de características, tais como Conv4_3, Conv7, Conv8_2, Conv9_2, Conv10_2, and Conv11_2, whose sizes are (38, 38), (19, 19), (10, 10), (5, 5), (3, 3), and (1, 1), respetivamente. SSD é treinada de forma a obter um conjunto de fixed-sized bounding boxes e a previsão das classes alvo das bounding boxes.
source(https://assets.researchsquare.com/files/rs-668895/v1_covered.pdf?c=1631875157)

# SSD
A arquitetura SSD adopta um algoritmo para a detecção de várias classes de objetos em um imagem, fornecendo pontuações associadas à presença de qualquer categoria de objetos. É indicado para aplicações em tempo real visto que não re-avalia as caixas delimitadoras como o Faster R-CNN. A arquitetura SSD é uma arquitetura baseada em CNN e para detetar classes dos objetos alvos, esta primeiramente extrai as características dos objetos e depois aplica filtros convulacionais para detetar objetos.
source(https://www.ijert.org/research/comparison-of-yolov3-and-ssd-algorithms-IJERTV10IS020077.pdf)

# YOLO
YOLO é uma CNN, este algoritmo aplica a imagem completa para uma rede neuronal solitária e depois isola a imagem em regiões, prevendo a bounding box e a probabilidade para cada região. Estas bounding boxes e as probabilidas apresentam boa acurácia enquanto permitem ser executadas em tempo real.
YOLO funciona por aceitar uma imagem como informação e dividi-la em uma grelha de SxS, tomando m caixas delimitadoras dentro da grelha.
Para cada caixa delimitadora a rede devolve a probalidade de classe e estimativas de contrapeso para cada caixa delimitadora formada. A caixa delimitadora com probabilidade da classe acima do valor limite é usada para encontrar o objeto dentro da imagem.
O YOLO é um algoritmo de deteção de objetos bastante rápido, mas apresenta algumas disvantagens quando este tem que distinguir objetos mais pequenos, devido às restrições espaciais do algoritmo.
source(https://www.ijert.org/research/comparison-of-yolov3-and-ssd-algorithms-IJERTV10IS020077.pdf)



# Single-stage methods vs Two-stage methods
Tipicamente metodos single stage como SSD e YOLO, aplicam sliding default boxes predifinidas de diferentes escalas e tamanhos em uma ou várias características para chegar a uma troca entre velocidade e acurácia. Este tipo de modelos são usualmente mais rápidos que os modelos de dois estagios, mas em contra partida apresentam menos precisão.
source(https://arxiv.org/pdf/1807.11013.pdf)


# YOLO vs SSD
source(https://arxiv.org/pdf/1807.11013.pdf) conclusion

- Explicar Two-stage methods
  - Explicar R-CNN, Faster R-CNN

- Explicar Single-stage methods
  - Explicar SSD
  - Explicar YOLO

- Comparar Single-stage vs Two-stage methods (concluir que Single-stage é melhor)

- Comparar SSD com YOLO (concluir que YOLO é melhor, apresentar dados que comprovem)

- Comparar várias versões do YOLO?


# Deteção de objetos
Um dos desafios da área do machine learning e visão computacional é a de deteção de objetos. Esta permite-nos indentificar e localizar vários objetos numa imagem ou num vídeo. Estas técnicas podem ser usadas para contar objetos em uma cena determinar as suas localizações e até rastreá-los mesmo estando em movimento, sendo aplicadas em áreas como a medicina, na indentificação de doenças através de imagens, na condução autónoma e até na deteção de pessoas no âmbito da segurança.
Geralmente, estes algoritmos podem ser classificados em duas categorias, sendo estas, os modelos de um único estágio e modelos de múltiplos estágios.

# Modelos de múltiplos estágios (dois estágios)
Um dos modelos mais conhecidos e usados de dois estágios são os R-CNN, Fast R-CNN e Faster R-CNN. Nestes, um primeiro estágio é responsável pela extração das regiões dos objetos e um segundo estágio é usado para classificar e aprimorar a localização do objeto na imagem. Estes métodos são mais lentos, mas muito poderosos e precisos.

## Modelos R-CNN
Neste model recebe-se uma imagem como entrada (1), de seguida são extraídas cerca de 2000 propostas de regiões (2), processam-se as características de cada proposta usando uma larga rede neuronal convulational (3), e por fim classifica-se cada região usando SVMs.

(imagem do modelo R-CNN)

O modelo R-CNN é lento pois executa uma passagem convulacional para cada proposta de objeto, sem partilhar poder computational.
Para resolver este problema foi criado posteriormente um modelo Fast R-CNN este intoduz várias melhorias e inovações na velocidade de treino e teste enquanto aumenta também a taxa de precisão. Fast R-CNN é 9x mais rápida que a R-CNN e alcança uma maior mAP no PASCAL VOC 2012.

Faster R-CNN, proposto por Ren et al. em 2015, é uma versão melhorada do R-CNN e do Fast R-CNN. A diferença entre estes dois é que o Faster R-CNN evita a exaustivo algoritmo de procura seletiva e usa o Region Proposal Net (RPN) para gerar regiões candidatas. Desta forma, é possível calcular as características de toda a imagem de uma só vez e, portanto, não envolve cálculos repetidos, o que melhora muito a velocidade de deteção do Faster R-CNN.


