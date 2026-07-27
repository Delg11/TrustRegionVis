# ==============================================================================
# VISUALIZAÇÃO COM ZOOM DINÂMICO SOBRE A REGIÃO DE CONFIANÇA
# ==============================================================================
#
# A câmera de cada quadro é recalculada a partir do raio da região de
# confiança (`delta`) e do tamanho do passo tentado naquela iteração, de modo
# que o zoom acompanha naturalmente o otimizador: quando `delta` encolhe (por
# exemplo, após um passo rejeitado), a câmera se aproxima; quando `delta`
# cresce, a câmera se afasta.
# ==============================================================================

"""
    visualizar_frames_dinamicos(prob::Problema, historico::Vector{PassoRC}; kwargs...)

Renderiza, quadro a quadro, a trajetória de um método de Região de Confiança
sobre as curvas de nível de `prob.f`, com zoom dinâmico que acompanha o raio
da região de confiança a cada iteração.

Cada quadro mostra:
- as curvas de nível preenchidas de `prob.f` na vizinhança do ponto atual;
- o quadrado da região de confiança (linha tracejada);
- uma seta indicando o passo tentado (verde se aceito, vermelho se rejeitado);
- os pontos "atual" e "candidato";
- o vetor gradiente atual (magenta) e, quando disponível, o gradiente da
  última iteração aceita (laranja, tracejado), para visualizar a mudança de
  direção de busca ao longo do tempo.

# Argumentos
- `prob`: problema de otimização a ser visualizado.
- `historico`: vetor de passos (`PassoRC`) a serem renderizados, em ordem.

# Argumentos nomeados
- `output_dir::String="frames_saida"`: diretório onde os PNGs serão salvos.
  É recriado do zero a cada chamada (o conteúdo anterior é apagado).
- `densidade_niveis::Int=3`: repassado a `gerar_niveis_inteligentes`.
- `fator_zoom::Float64=1.2`: margem extra aplicada ao raio de visão da câmera.
- `raio_minimo::Float64=0.05`: impede zoom microscópico em passos muito pequenos.
- `resolucao_grade::Int=50`: número aproximado de células de grade por eixo
  usadas para desenhar o contorno (maior = mais suave, porém mais lento).
- `mostrar_progresso::Bool=true`: imprime o andamento da renderização.

# Retorno
O caminho (`String`) do diretório onde os quadros foram salvos.

# Exemplo
Veja `examples/exemplo_minimo.jl` para um exemplo completo e executável,
incluindo um otimizador de Região de Confiança aplicado à função de Rosenbrock.
"""
function visualizar_frames_dinamicos(
    prob::Problema,
    historico::Vector{PassoRC};
    output_dir::String = "frames_saida",
    densidade_niveis::Int = 3,
    fator_zoom::Float64 = 1.2,
    raio_minimo::Float64 = 0.05,
    resolucao_grade::Int = 50,
    mostrar_progresso::Bool = true,
)
    # Prepara o diretório de saída (limpo a cada execução, para não misturar
    # quadros de execuções antigas com a atual)
    isdir(output_dir) && rm(output_dir, recursive = true, force = true)
    mkpath(output_dir)

    mostrar_progresso && println("\n🎥 Renderizando com zoom dinâmico na região de confiança...")
    mostrar_progresso && println("   Calculando níveis de contorno inteligentes...")

    niveis = gerar_niveis_inteligentes(prob, historico; densidade = densidade_niveis)

    gradiente_anterior = nothing

    for (i, passo) in enumerate(historico)
        # --- 1. Definição da câmera (zoom dinâmico) ---
        cx, cy = passo.x_from[1], passo.x_from[2]
        delta = passo.delta

        dist_passo = norm(passo.x_to - passo.x_from, Inf)
        raio_visao = max(delta, dist_passo) * fator_zoom
        raio_visao = max(raio_visao, raio_minimo)  # impede zoom excessivo (microscópico)

        xlims = (cx - raio_visao, cx + raio_visao)
        ylims = (cy - raio_visao, cy + raio_visao)
        passo_grade = raio_visao / resolucao_grade  # densidade de grade adaptativa

        # --- 2. Plotagem do contorno de fundo ---
        p = contour(
            xlims[1]:passo_grade:xlims[2],
            ylims[1]:passo_grade:ylims[2],
            (x, y) -> prob.f([x, y]);
            levels = niveis,
            fill = true, c = :viridis, alpha = 0.5, legend = :topright,
            xlims = xlims, ylims = ylims,
            framestyle = :box, size = (800, 600), dpi = 150,
            aspect_ratio = :equal,
        )

        # --- 3. Região de confiança (quadrado tracejado) ---
        plot!(
            p,
            [cx - delta, cx + delta, cx + delta, cx - delta, cx - delta],
            [cy - delta, cy - delta, cy + delta, cy + delta, cy - delta];
            color = :black, style = :dash, linewidth = 2.0, label = "Região de Confiança",
        )

        # --- 4. Passo tentado ---
        aceito = (passo.status == :aceito)
        cor_passo = aceito ? :green : :red
        rotulo_passo = aceito ? "Passo ACEITO" : "Passo REJEITADO"

        plot!(
            p, [cx, passo.x_to[1]], [cy, passo.x_to[2]];
            color = cor_passo, arrow = true, linewidth = 3, label = rotulo_passo,
        )

        scatter!(
            p, [passo.x_to[1]], [passo.x_to[2]];
            color = cor_passo, shape = (aceito ? :circle : :xcross), markersize = 8, label = "",
        )
        scatter!(p, [cx], [cy]; color = :blue, markersize = 6, label = "Atual")

        # --- 5. Gradientes (atual e anterior, escala adaptativa ao zoom) ---
        gradiente_atual = prob.∇f(passo.x_from)
        escala_grad = raio_visao * 0.4

        v_atual = normalize(gradiente_atual) * escala_grad
        plot!(
            p, [cx, cx + v_atual[1]], [cy, cy + v_atual[2]];
            color = :magenta, arrow = true, linewidth = 2, label = "∇f(k)",
        )

        if !isnothing(gradiente_anterior) && norm(gradiente_atual - gradiente_anterior) > 1e-5
            v_anterior = normalize(gradiente_anterior) * escala_grad
            plot!(
                p, [cx, cx + v_anterior[1]], [cy, cy + v_anterior[2]];
                color = :orange, arrow = true, style = :dot, linewidth = 2, label = "∇f(k-1)",
            )
        end
        aceito && (gradiente_anterior = gradiente_atual)

        # --- 6. Título com as métricas da iteração ---
        status_str = uppercase(string(passo.status))
        titulo = @sprintf(
            "Iter %d | %s\nAred: %.2e | Pred: %.2e | δ: %.2e",
            passo.iter, status_str, passo.ared, passo.pred, delta,
        )
        title!(p, titulo)

        savefig(p, joinpath(output_dir, @sprintf("quadro_%04d.png", i)))

        mostrar_progresso && i % 10 == 0 && print(".")
    end

    mostrar_progresso && println("\n✅ Concluído! Quadros salvos em: ", output_dir)
    return output_dir
end

"""
    gerar_gif(output_dir::String; nome_arquivo::String="animacao.gif", fps::Int=6)

Compila os quadros PNG salvos por `visualizar_frames_dinamicos` em um GIF
animado. Requer `ffmpeg` disponível no sistema, já que o Plots.jl delega a
codificação do GIF a ele internamente.

# Argumentos
- `output_dir`: diretório contendo os arquivos `quadro_XXXX.png`.
- `nome_arquivo`: caminho do GIF de saída.
- `fps`: quadros por segundo do GIF final.

# Retorno
O caminho (`String`) do GIF gerado.
"""
function gerar_gif(output_dir::String; nome_arquivo::String = "animacao.gif", fps::Int = 6)
    arquivos = sort(filter(f -> endswith(f, ".png"), readdir(output_dir, join = true)))
    isempty(arquivos) && error("Nenhum quadro .png encontrado em $output_dir")

    anim = Animation(output_dir, basename.(arquivos))
    gif(anim, nome_arquivo, fps = fps)
    return nome_arquivo
end