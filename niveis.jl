# ==============================================================================
# GERAÇÃO DE NÍVEIS DE CONTORNO (INTERPOLAÇÃO LOGARÍTMICA)
# ==============================================================================
#
# Funções com vales estreitos e paredes íngremes (como a de Rosenbrock) são
# difíceis de desenhar com `contour`: níveis igualmente espaçados deixam o
# vale "escondido" atrás das paredes, enquanto níveis puramente logarítmicos
# e uniformes podem não ter nenhum nível próximo dos valores de f(x) que o
# otimizador realmente visitou — deixando a trajetória sem contraste visual.
#
# A ideia aqui é simples: sempre incluir os valores de f(x) efetivamente
# visitados no histórico, e só então preencher os espaços vazios entre eles
# com níveis intermediários em escala log.
# ==============================================================================

"""
    gerar_niveis_inteligentes(prob::Problema, historico::Vector{PassoRC}; densidade::Int=2)

Gera um vetor de níveis de contorno (*levels*) que inclui obrigatoriamente os
valores de `f(x)` visitados ao longo da otimização, preenchendo os espaços
vazios entre eles com níveis intermediários em escala logarítmica.

# Argumentos
- `prob`: o problema de otimização (usado para reavaliar `f` em cada ponto do histórico).
- `historico`: vetor de passos (`PassoRC`) já executados pelo otimizador.

# Argumentos nomeados
- `densidade::Int=2`: quantos níveis intermediários inserir em cada "buraco"
  grande (fator maior que 2×) entre dois níveis consecutivos.

# Retorno
Um `Vector{Float64}` ordenado e sem duplicatas próximas, pronto para ser
passado ao argumento `levels` de `Plots.contour`.
"""
function gerar_niveis_inteligentes(prob::Problema, historico::Vector{PassoRC}; densidade::Int=2)
    # 1. Coleta todos os valores de f(x) visitados (origem e destino de cada passo)
    valores_z = Float64[]
    push!(valores_z, 0.1)  # "quase zero", garante que o fundo do vale apareça no mapa

    for passo in historico
        push!(valores_z, prob.f(passo.x_from))
        push!(valores_z, prob.f(passo.x_to))
    end

    # Valores altos fixos, para garantir que as paredes íngremes fiquem visíveis
    # mesmo que o otimizador nunca as tenha visitado diretamente
    append!(valores_z, [1e4, 1e5, 1e6])

    # 2. Ordena e remove duplicatas muito próximas entre si
    sort!(valores_z)
    z_unicos = Float64[valores_z[1]]
    for v in valores_z[2:end]
        # só adiciona se for pelo menos 1% maior que o anterior (evita linhas grudadas)
        if v > z_unicos[end] * 1.01
            push!(z_unicos, v)
        end
    end

    # 3. Interpolação logarítmica entre níveis consecutivos muito distantes
    niveis_finais = Float64[]
    for i in 1:(length(z_unicos) - 1)
        a, b = z_unicos[i], z_unicos[i + 1]
        push!(niveis_finais, a)

        # se o "buraco" entre a e b for grande (fator > 2), insere intermediários
        if b > a * 2.0 && a > 1e-6
            log_a, log_b = log10(a), log10(b)
            passos_log = range(log_a, log_b, length = densidade + 2)

            for k in 2:(length(passos_log) - 1)  # pula os extremos, já incluídos acima
                push!(niveis_finais, 10.0 ^ passos_log[k])
            end
        end
    end
    push!(niveis_finais, z_unicos[end])  # adiciona o último nível

    return niveis_finais
end