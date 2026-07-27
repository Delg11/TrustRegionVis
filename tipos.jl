# ==============================================================================
# TIPOS FUNDAMENTAIS
# ==============================================================================
#
# Este arquivo define os dois tipos em torno dos quais todo o framework é
# construído: `Problema` (a função a ser visualizada) e `PassoRC` (o registro
# de uma iteração de um método de Região de Confiança). Qualquer otimizador,
# desde que produza um vetor de `PassoRC`, pode ter sua trajetória visualizada
# pelo restante do pacote — o framework não sabe nada sobre como os passos
# foram calculados, apenas como desenhá-los.
# ==============================================================================

"""
    Problema{F,G}

Representa um problema de otimização irrestrita em duas variáveis, pronto
para ser visualizado.

# Campos
- `f::F`: função objetivo. Deve aceitar um vetor `x::Vector{Float64}` de
  tamanho 2 e retornar um escalar `Float64`.
- `∇f::G`: função gradiente de `f`. Deve aceitar `x` e retornar um vetor de
  tamanho 2 com o gradiente naquele ponto.
- `nome::String`: nome descritivo do problema, usado apenas em relatórios e
  títulos (não afeta o cálculo).

# Exemplo
```julia
rosenbrock(x) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
grad_rosenbrock(x) = [
    -2*(1 - x[1]) - 400*x[1]*(x[2] - x[1]^2),
    200*(x[2] - x[1]^2),
]

prob = Problema(rosenbrock, grad_rosenbrock, "Rosenbrock")
```
"""
struct Problema{F,G}
    f::F
    ∇f::G
    nome::String
end

# Construtor de conveniência: nome é opcional
Problema(f, ∇f) = Problema(f, ∇f, "Problema sem nome")

"""
    PassoRC

Registra uma única iteração de um método de Região de Confiança (*Trust
Region*), contendo tudo o que é necessário para reconstruir visualmente a
decisão tomada pelo otimizador naquele passo.

Este tipo é o "contrato" entre um otimizador qualquer e as funções de
visualização deste pacote: qualquer algoritmo capaz de preencher um vetor de
`PassoRC` pode ser visualizado por `visualizar_frames_dinamicos`.

# Campos
- `iter::Int`: número da iteração (apenas para exibição).
- `x_from::Vector{Float64}`: ponto de partida da iteração.
- `x_to::Vector{Float64}`: ponto candidato gerado nessa iteração (aceito ou não).
- `delta::Float64`: raio da região de confiança usado para gerar este passo.
- `status::Symbol`: `:aceito` se o passo foi aceito pelo otimizador, ou
  `:rejeitado` caso contrário.
- `ared::Float64`: redução real (*actual reduction*) obtida em `f`.
- `pred::Float64`: redução prevista pelo modelo quadrático (*predicted
  reduction*) usado internamente pelo otimizador.
"""
struct PassoRC
    iter::Int
    x_from::Vector{Float64}
    x_to::Vector{Float64}
    delta::Float64
    status::Symbol
    ared::Float64
    pred::Float64
end