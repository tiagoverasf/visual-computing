# REPRESENTAÇÃO
# -------------
# 1. A população é uma grade L x L; cada célula é uma pessoa.
# 2. Cada pessoa é um vetor de 5 características (sexo, período, curso,
#    religião, origem). Cada característica é um número inteiro que aponta
#    para uma opção da lista CARACTERISTICAS.
# 3. A cultura é a combinação completa desses valores. Mesma combinação =
#    mesma cultura.
# 4. No início, todos os valores são sorteados aleatoriamente, então a grade
#    começa muito diversa.
# 5. Cada pessoa só interage com os vizinhos de cima, baixo, esquerda e
#    direita (bordas fixas, sem wrap-around).


# SIMILARIDADE
# ------------
# Fração de características iguais entre duas pessoas.
# Exemplo:
#     A = [1, 2, 0, 3, 1]
#     B = [1, 2, 4, 0, 1]   -> 3 iguais de 5 -> similaridade = 0,6


# CONTAMINAÇÃO (PASSO A PASSO)
# ----------------------------
# 1. Sorteia-se uma pessoa e um de seus vizinhos.
# 2. Calcula-se a similaridade s entre os dois.
# 3. Se s = 0 (nada em comum) ou s = 1 (idênticos), nada acontece.
# 4. Se 0 < s < 1, com probabilidade s ocorre a influência: a pessoa escolhe
#    uma característica em que difere do vizinho e adota o valor dele.
# 5. Seleciono a cada passos printa uma amostra para visualização 
# 6. O processo se repete até o sistema estabilizar, ou seja, até todo par
#    de vizinhos ser totalmente igual ou totalmente diferente.

# OBSERVAÇÕES
# ----------------------------
# 1. A estrutura príncipal do código esta feita, porem há muito para melhorar como gráficos, representação ao vivo, entre outros...
# 2. Como foi feito em sala e em pouco tempo a lógica que usei não foi tão apurada, então considerei que todos podem sofrer alterações em todos os atributos(até mesmo sexo) há muito para aprimorar nisso
# 3. O Código esta incompleto, isso é uma versão primitiva, então podem ter erros
# 4. Há bastantes melhorias no sistema de seeds, até o momento só há como colocar manualmente no código ou deixar aleatorio

import numpy as np
import matplotlib.pyplot as plt
import secrets
from matplotlib.colors import ListedColormap


CARACTERISTICAS = {
    "sexo":     ["Feminino", "Masculino"],
    "periodo":  ["1º", "2º", "3º", "4º", "5º", "6º", "7º", "8º"],
    "curso":    ["Medicina", "Direito", "Engenharia", "Ciencia da Computação", "Administração", "Inteligência Artificial"],
    "religiao": ["Católica", "Evangélica", "Candomble", "Sem religião", "Outra"],
    "origem":   ["Norte", "Norteste", "Sudeste", "Sul", "Leste"],

    #LOGICA:
    # A = [1, 2, 0, 3, 1]
    # B = [1, 2, 4, 0, 1]   → 3 iguais de 5 → similaridade = 0,6
}
NOMES = list(CARACTERISTICAS.keys())
F = len(NOMES)                                   # nº de características
Q = [len(v) for v in CARACTERISTICAS.values()]   # nº de valores de cada uma


def gerar_matriz(L, rng):
    """Matriz L x L x F; cada pessoa é um vetor com F características."""
    pessoas = np.zeros((L, L, F), dtype=int)
    for f in range(F):
        pessoas[:, :, f] = rng.integers(0, Q[f], size=(L, L))
    return pessoas


def descrever(pessoa):
    """Converte o vetor numérico em um dicionário legível."""
    return {nome: CARACTERISTICAS[nome][v] for nome, v in zip(NOMES, pessoa)}


VIZINHOS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def vizinhos_de(i, j, L):
    """Vizinhos de Von Neumann (bordas fixas, sem wrap-around)."""
    return [(i + di, j + dj) for di, dj in VIZINHOS
            if 0 <= i + di < L and 0 <= j + dj < L]


def similaridade(a, b):
    return np.mean(a == b)


def esta_estavel(pessoas):
    """Estável quando nenhum par de vizinhos tem 0 < similaridade < 1."""
    L = pessoas.shape[0]
    for i in range(L):
        for j in range(L):
            for ni, nj in vizinhos_de(i, j, L):
                s = similaridade(pessoas[i, j], pessoas[ni, nj])
                if 0 < s < 1:
                    return False
    return True


def passo(pessoas, rng):
    """Um passo do modelo. Retorna True se alguma característica mudou."""
    L = pessoas.shape[0]
    i, j = rng.integers(0, L, size=2)
    ni, nj = vizinhos_de(i, j, L)[rng.integers(0, len(vizinhos_de(i, j, L)))]

    a, b = pessoas[i, j], pessoas[ni, nj]
    s = similaridade(a, b)

    if 0 < s < 1 and rng.random() < s:
        diferentes = np.flatnonzero(a != b)
        f = rng.choice(diferentes)
        a[f] = b[f]          # a pessoa adota a característica do vizinho
        return True
    return False


def simular(L=15, max_passos=2_000_000, checar_a_cada=5000, semente=None):
    if semente is None:
        semente = secrets.randbits(32)
    print(f"Semente usada: {semente}")
    rng = np.random.default_rng(semente)

    pessoas = gerar_matriz(L, rng)
    fotos = [(0, pessoas.copy())]          # lista de (passo, matriz)

    for t in range(1, max_passos + 1):
        passo(pessoas, rng)
        if t % checar_a_cada == 0:
            fotos.append((t, pessoas.copy()))
            if esta_estavel(pessoas):
                print(f"Estado estável atingido após {t} passos.")
                break
    else:
        print("Limite de passos atingido (pode não ter estabilizado).")

    if fotos[-1][0] != t:                  # garante o estado final nas fotos
        fotos.append((t, pessoas.copy()))
    return fotos


def codigo_cultura(pessoas, tabela):
    """
    Dá um ID inteiro para cada combinação única de características.
    `tabela` é compartilhada entre todas as imagens, então a mesma cultura
    mantém a mesma cor do início ao fim.
    """
    L = pessoas.shape[0]
    ids = np.zeros((L, L), dtype=int)
    for i in range(L):
        for j in range(L):
            ids[i, j] = tabela.setdefault(tuple(pessoas[i, j]), len(tabela))
    return ids


def plotar_processo(fotos, n_imagens=8, colunas=4):
    """Mostra `n_imagens` fotos igualmente espaçadas ao longo do processo."""
    n_imagens = min(n_imagens, len(fotos))
    indices = np.linspace(0, len(fotos) - 1, n_imagens).round().astype(int)
    escolhidas = [fotos[k] for k in indices]

    tabela = {}
    matrizes_ids = [codigo_cultura(m, tabela) for _, m in escolhidas]
    cmap = ListedColormap(np.random.default_rng(0).random((len(tabela), 3)))

    linhas = int(np.ceil(n_imagens / colunas))
    fig, eixos = plt.subplots(linhas, colunas, figsize=(3.2 * colunas, 3.4 * linhas))
    eixos = np.atleast_1d(eixos).ravel()

    for ax, (t, _), ids in zip(eixos, escolhidas, matrizes_ids):
        ax.imshow(ids, cmap=cmap, vmin=0, vmax=len(tabela) - 1, interpolation="nearest")
        ax.set_title(f"Passo {t}\n{len(np.unique(ids))} culturas", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in eixos[n_imagens:]:           # esconde os eixos vazios
        ax.axis("off")

    plt.suptitle("Disseminação cultural de Axelrod: cada cor = uma cultura")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    N_IMAGENS = 8        # quantas imagens mostrar do processo
    COLUNAS = 4          # quantas imagens por linha

    fotos = simular(L=15, checar_a_cada=5000)
    inicial, final = fotos[0][1], fotos[-1][1]

    print("\nExemplo de pessoa na posição (0, 0):")
    print("  Inicial:", descrever(inicial[0, 0]))
    print("  Final:  ", descrever(final[0, 0]))

    # Distribuição final de cada característica (característica: nº de pessoas)
    print("\nDistribuição final:")
    for f, nome in enumerate(NOMES):
        valores, contagens = np.unique(final[:, :, f], return_counts=True)
        resumo = {CARACTERISTICAS[nome][v]: int(c) for v, c in zip(valores, contagens)}
        print(f"  {nome}: {resumo}")

    plotar_processo(fotos, n_imagens=N_IMAGENS, colunas=COLUNAS)