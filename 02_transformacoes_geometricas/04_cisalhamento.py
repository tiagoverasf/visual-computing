from PIL import Image
import numpy as np

# 1.Carregar a imagem e transformar em array NumPy
caminho_entrada = r"assets/img/cats 2.jpg"
imagem = Image.open(caminho_entrada)
arr = np.array(imagem)

altura, largura = arr.shape[:2] #trabalhar somente com 2 dimensões (largura, altura)
print(f"Original: shape NumPy = {arr.shape} (altura,largura," f"canais)")

# 2. Definir o fator de cisalhamento horizontal.
#    shx = 0.3 significa: cada linha é deslocada 0.3 * y pixels para a direita.
#    Alinha do topo (y=0) fica parada; a de baixo (y=altura-1) é a mais deslocada

shx = 0.3
# 3. Calcular a nova largura para que a imagem cisalhada caiba inteira no canvas
deslocamento_max = int(np.ceil(shx * (altura -1)))
nova_largura = largura + deslocamento_max

# 4. Cisalhamento em NumPy usando MAPEAMENTO INVERSO:
#    Para cada pixel de SAÍDA   (x_out, y_out), calculamos de qual pixel
#    da Entrada ele veio, aplicando a inversa da transformação: x_in = x_out -
#    shx * y_out.
#    Isso evita "buracos" que apareceriam se fizéssemos o mapeamento direto.

y_out, x_out = np.indices((altura, nova_largura)) #malhas de coordenadas
x_in = (x_out - shx * y_out).astype(int) #vizinho mais próximo
y_in = y_out # y não muda

# 5. Máscara: pixels cuja coordenadas de origem cai dentro da imagem original
#    Os que ficam fora viram fundo (preto por padrão, pois o canvas
#    começa zerado).

valido = (x_in >= 0) & (x_in < largura)

# 6. Montar a imagem de saída usando indexação avançada do NumPy (sem loops).
arr_cisalhado = np.zeros((altura, nova_largura, arr.shape[2]),dtype=arr.dtype)
arr_cisalhado[valido] = arr[y_in[valido], x_in[valido]]

print(f"Cisalhado: shape NumPy = {arr_cisalhado.shape} (largura aumenta em {deslocamento_max}px)")

# 7. Voltar para PIL só para salvar em disco
caminho_saida = r"assets/img/imagem_cisalhada.png"
Image.fromarray(arr_cisalhado).save(caminho_saida)

print("\033[30;47mImagem cisalhada e salva com sucesso!\033[m")