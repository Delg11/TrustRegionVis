# TrustRegionVis

Framework leve para visualizar, quadro a quadro, o comportamento de métodos de otimização por Região de Confiança (*Trust Region*) em problemas 2D.

## Funcionalidades

* O pacote gera uma sequência de imagens PNG com zoom dinâmico da câmera, recalculado a partir do raio da região de confiança e do tamanho do passo tentado.
* Os níveis de contorno são gerados com interpolação logarítmica para preencher espaços vazios, garantindo que vales e paredes íngremes permaneçam visíveis no mapa.
* A renderização apresenta indicadores visuais do vetor gradiente atual, do gradiente da iteração aceita anterior, da região de confiança delimitada e da decisão de aceitação ou rejeição do passo.
* O framework é agnóstico em relação ao otimizador, exigindo estritamente que as iterações sejam registradas e fornecidas em um vetor do tipo `PassoRC`.
* Os quadros gerados pela função `visualizar_frames_dinamicos` podem ser compilados em um arquivo GIF através da função `gerar_gif`, a qual exige a disponibilidade do executável `ffmpeg` no sistema.

## Uso Básico

A estrutura do pacote é fundamentada nos tipos `Problema`, que armazena a função objetivo e seu gradiente, e `PassoRC`, que encapsula as métricas e decisões de uma única iteração do otimizador.

```julia
using TrustRegionVis

prob = Problema(f, gradiente_f, "Meu problema")
historico = PassoRC[
    PassoRC(1, x0, x1, delta, :aceito, ared, pred),
    # ...
]

visualizar_frames_dinamicos(prob, historico)

```

Um script executável completo encontra-se no pacote. Para executar uma otimização com o método do dogleg na função de Rosenbrock e gerar as visualizações, utilize o comando abaixo na raiz do diretório:

```bash
julia --project=. examples/exemplo_minimo.jl

```

## Licença

Este software é provido sob a licença MIT. Copyright (c) 2026 Delg11.
