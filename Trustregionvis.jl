"""
    TrustRegionVis

Framework leve para visualizar, quadro a quadro, o comportamento de métodos
de otimização por Região de Confiança (*Trust Region*) em problemas 2D.

Dado o histórico de iterações de um otimizador — pontos visitados, raio da
região de confiança em cada passo, status de aceitação/rejeição, redução
real e prevista — este pacote gera uma sequência de imagens PNG com:

- zoom dinâmico da câmera, que acompanha o raio da região de confiança;
- níveis de contorno adaptados automaticamente à função objetivo, garantindo
  que o vale (ou qualquer região de interesse) permaneça visível mesmo em
  funções com paredes muito íngremes, como a de Rosenbrock;
- indicadores visuais de gradiente, passo tentado e decisão do algoritmo.

O framework é agnóstico em relação ao otimizador: qualquer algoritmo capaz de
produzir um `Vector{PassoRC}` pode ter sua trajetória visualizada.

# Uso básico
```julia
using TrustRegionVis

prob = Problema(f, gradiente_f, "Meu problema")
historico = PassoRC[
    PassoRC(1, x0, x1, delta, :aceito, ared, pred),
    # ...
]

visualizar_frames_dinamicos(prob, historico)
```

Veja `examples/exemplo_minimo.jl` para um exemplo completo e executável,
incluindo um otimizador de Região de Confiança (método do dogleg) aplicado
à função de Rosenbrock, do início ao fim.
"""
module TrustRegionVis

using LinearAlgebra
using Plots
using Printf

include("tipos.jl")
include("niveis.jl")
include("visualizacao.jl")

export Problema, PassoRC
export gerar_niveis_inteligentes, visualizar_frames_dinamicos, gerar_gif

end # module TrustRegionVis