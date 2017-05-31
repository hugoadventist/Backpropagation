# encoding: iso-8859-1

import numpy as np
import matplotlib.pyplot as plt
#import sklearn as skl


def pesos(n,m):
    np.random.seed(1)
    return np.random.rand(n,m)


def bias():
#    np.random.seed(1)
#    return 0.1*np.random.randn(1)
    return -1

def dsigmoid(sigmoid):
    return sigmoid * (1 - sigmoid)

def normalizar(entradas_treino):
    for a in range(len(entradas_treino)):
        for i in range(len(entradas_treino[0,:])):
            entradas_treino[a,i] /= entradas_treino[:,i].max()
    return entradas_treino


class DataSet:
    def __init__(self,caminho_do_arquivo,coluna_separacao):
        self.caminho_do_arquivo = caminho_do_arquivo
        self.coluna_separacao = coluna_separacao

        base_dados = np.loadtxt(self.caminho_do_arquivo, delimiter=",")
        self.entradas = base_dados[:,self.coluna_separacao:]
        self.saida_desejada = base_dados[:,:self.coluna_separacao]

    def mostrar_dados(self):
        print(len(self.entradas))
        print(self.entradas)
        print(self.saida_desejada)

class Neuronio:

    def __init__(self, entradas_treino, pesos, bias):
        self.entradas_treino = entradas_treino
        #self.bias = []
        self.bias = bias
        self.pesos = pesos


#    def vetor_campo_induzido(self, entradas_para_neuronio):
    def vetor_campo_induzido(self):
        #self.campo_induzido = np.dot(self.entradas_treino, self.peso)
        self.campo_induzido = np.dot(self.entradas_treino, self.pesos) + self.bias
        return self.campo_induzido

    def sigmoid(self):
        return 1/(1 + np.exp(self.campo_induzido))




class RedeNeural:

    def __init__(self, taxa_aprendizado, neuronio_entrada, neuronio_escondida, neuronio_saida):
        self.taxa_aprendizado = taxa_aprendizado
        self.neuronio_escondida = neuronio_escondida
        self.neuronio_entrada = neuronio_entrada
        self.neuronio_saida = neuronio_saida

        self.bias_camada_entrada_escondida = bias()
        print("Bias camada escondida",self.bias_camada_entrada_escondida)
        self.bias_camada_escondida_saida = bias()
        print("Bias camada saida",self.bias_camada_escondida_saida)

        self.erro_quadratico_medio = []
        self.erro_saida = []

#        self.pesos_camada_entrada_p_escondida = np.array(self.pesos_iniciais_camada_entrada_p_escondida())
#        self.pesos_camada_escondida_p_saida = np.array(self.pesos_iniciais_camada_escondida_p_saida())
        self.pesos_camada_entrada_p_escondida = self.pesos_iniciais_camada_entrada_p_escondida()
        self.pesos_camada_escondida_p_saida = self.pesos_iniciais_camada_escondida_p_saida()



    def pesos_iniciais_camada_entrada_p_escondida(self):
        pesos_iniciais = pesos(self.neuronio_entrada, self.neuronio_escondida)
        print("W_h = ", pesos_iniciais)
        return pesos_iniciais

    def pesos_iniciais_camada_escondida_p_saida(self):
        pesos_iniciais = pesos(self.neuronio_escondida,self.neuronio_saida)
        print("W_o = ", pesos_iniciais)
        print('\n')
        return pesos_iniciais

    def treino(self, numero_epocas, entradas_treino, saida_treino):
        self.entradas_treino = normalizar(entradas_treino)
        self.num_epocas = numero_epocas
        self.saida_treino = saida_treino

        for i in range(self.num_epocas):
            erro_total = 0
            for j in range(len(self.entradas_treino)):
                # Inseri para percorrer cada exemplo de treinamento, utilizados como entrada para os neuronios de entrada
                camada_escondida = Neuronio(self.entradas_treino[j],self.pesos_camada_entrada_p_escondida,
                                          self.bias_camada_entrada_escondida)

                # Inseri para percorrer cada exemplo de treinamento, utilizados como entrada para os neuronios de entrada
                #camada_escondida.vetor_campo_induzido(self.entradas_treino[j])*/
                camada_escondida.vetor_campo_induzido()
                sig0 = camada_escondida.sigmoid()
                camada_saida = Neuronio(sig0,self.pesos_camada_escondida_p_saida,
                                            self.bias_camada_escondida_saida)

                #aqui inseri para percorrer o resultado da sigmoid, utilizado como entrada para o neuronio de saída
                camada_saida.vetor_campo_induzido()
                sig1 = camada_saida.sigmoid()
                erro_camada_saida = self.saida_treino[j] - sig1
                self.erro_saida.append(erro_camada_saida)

                #  inseri para pegar somente o valor da saída de treino
                erro_total += 0.5 * erro_camada_saida ** 2
                #derivada parcial
                delta_camada_saida = erro_camada_saida * dsigmoid(sig1)

                # delta_camada_escondida = (delta_camada_saida.dot(self.pesos_camada_escondida_p_saida.T))*sig0*(1-sig0)
                # inseri para somente pegar o valor do delta da camada de saída
                delta_vs_pesos = np.dot(delta_camada_saida, self.pesos_camada_escondida_p_saida.T)
                delta_camada_escondida = np.dot(delta_vs_pesos,dsigmoid(sig0))

                #self.pesos_camada_escondida_p_saida = self.taxa_aprendizado * (sig1.T.dot(delta_camada_saida))
                #self.pesos_camada_entrada_p_escondida = self.taxa_aprendizado * (sig0.T.dot(delta_camada_escondida))
                # O resultado do aprendizado deve ser somado aos pesos, correto?
                self.pesos_camada_escondida_p_saida += self.taxa_aprendizado * (sig1.T.dot(delta_camada_saida))
                self.pesos_camada_entrada_p_escondida += self.taxa_aprendizado * (sig0.T.dot(delta_camada_escondida))


            self.erro_quadratico_medio.append(erro_total/len(self.entradas_treino))
            #erro_quadratico_medio = np.mean(np.abs(self.erro_saida))


dados = DataSet("dados_wine.csv", 1)
dados.mostrar_dados()
rede1 = RedeNeural(0.01,13,10,1)
rede1.treino(40000,dados.entradas,dados.saida_desejada)

#Saida dos erros quadráticos

print("Vetor de erro quadrático médio:",rede1.erro_quadratico_medio)
#print("Vetor de erro na saída",rede1.erro_saida)
print("\n Valores dos pesos após o treinamento \n")
print("W_o:",rede1.pesos_camada_escondida_p_saida)
print("W_h:",rede1.pesos_camada_entrada_p_escondida)
print("\n Último valor de erro:",rede1.erro_quadratico_medio[len(rede1.erro_quadratico_medio)- 1])
# GRAFICOS DA REDE NEURAL
#Gráfico saida desejada vs saída da rede neural
plt.title('Saída desejada versus saída da rede neural')
plt.xlabel('Número de amostras')
plt.ylabel('Saída desejada e saída da rede neural')
plt.xlim(0,180)
plt.plot(rede1.erro_saida, 'ro',rede1.saida_treino,'bo')
plt.savefig('Grafico_saidas.png')
plt.show()

#Erros ao longo das epocas

plt.title('Evolucao do erro ao longo das epocas')
plt.xlabel('Epocas')
plt.ylabel('Media dos erros das saidas')
plt.xlim((0,rede1.num_epocas))
plt.plot(rede1.erro_quadratico_medio)
plt.savefig("Gráfico de evolução do erro")
plt.show()


