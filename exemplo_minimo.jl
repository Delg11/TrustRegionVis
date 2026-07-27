# ==============================================================================
# EXEMPLO MÍNIMO DE USO — TrustRegionVis.jl
# ==============================================================================
#
# Este exemplo implementa, do zero, um otimizador simples de Região de
# Confiança (método do "dogleg") aplicado à clássica função de Rosenbrock, e
# usa o pacote TrustRegionVis apenas para gerar a visualização do processo.
#
# O otimizador em si é puramente didático (poucas dezenas de linhas) — o
# objetivo é mostrar o contrato mínimo que qualquer algoritmo precisa
# cumprir para ser visualizado: produzir um `Vector{PassoRC}`.
#
# Para rodar (a partir da raiz do pacote):
#     julia --project=. examples/exemplo_minimo.jl
# ==============================================================================

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using LinearAlgebra
using. TrustRegionVis

# ------------------------------------------------------------------------------
# 1. Definição do problema: função de Rosenbrock
# ------------------------------------------------------------------------------

"""
    rosenbrock(x; a=1.0, b=100.0)

Clássica função de Rosenbrock, com um vale estreito e curvo — ideal para
testar visualmente métodos de otimização, pois força o algoritmo a "seguir a
curva" do vale em vez de seguir direto rumo ao mínimo.

Mínimo global em `x = [a, a^2]`, onde `f(x) = 0`.
"""
function rosenbrock(x; a = 1.0, b = 100.0)
    return (a - x[1])^2 + b * (x[2] - x[1]^2)^2
end

"""
    grad_rosenbrock(x; a=1.0, b=100.0)

Gradiente analítico da função de Rosenbrock.
"""
function grad_rosenbrock(x; a = 1.0, b = 100.0)
    dx = -2 * (a - x[1]) - 4 * b * x[1] * (x[2] - x[1]^2)
    dy = 2 * b * (x[2] - x[1]^2)
    return [dx, dy]
end

"""
    hess_rosenbrock(x; a=1.0, b=100.0)

Hessiana analítica da função de Rosenbrock. Usada apenas internamente pelo
otimizador deste exemplo — o framework TrustRegionVis não exige uma
Hessiana, apenas `f` e `∇f`.
"""
function hess_rosenbrock(x; a = 1.0, b = 100.0)
    h11 = 2 - 4 * b * x[2] + 12 * b * x[1]^2
    h12 = -4 * b * x[1]
    h22 = 2 * b
    return [h11 h12; h12 h22]
end

# ------------------------------------------------------------------------------
# 2. Otimizador de Região de Confiança (método do dogleg de Powell)
# ------------------------------------------------------------------------------

"""
    passo_dogleg(g, B, delta)

Calcula o passo de dogleg — combinação do ponto de Cauchy (direção de máxima
descida) com o passo de Newton — restrito ao raio `delta` da região de
confiança. É a heurística clássica de Powell para resolver aproximadamente o
subproblema da região de confiança sem precisar de um solver iterativo.
"""
function passo_dogleg(g::Vector{Float64}, B::Matrix{Float64}, delta::Float64)
    # Ponto de Cauchy: melhor passo ao longo de -g, dado o modelo quadrático
    gBg = dot(g, B * g)
    p_cauchy = gBg > 0 ? -(dot(g, g) / gBg) * g : -(delta / norm(g)) * g

    # Passo de Newton: mínimo exato do modelo quadrático (se B for bem-condicionada)
    p_newton = try
        -(B \ g)
    catch
        p_cauchy
    end

    if norm(p_newton) <= delta
        return p_newton
    elseif norm(p_cauchy) >= delta
        return (delta / norm(p_cauchy)) * p_cauchy
    else
        # Interpola entre Cauchy e Newton até tocar a borda da região de confiança
        d = p_newton - p_cauchy
        coef_a = dot(d, d)
        coef_b = 2 * dot(p_cauchy, d)
        coef_c = dot(p_cauchy, p_cauchy) - delta^2
        tau = (-coef_b + sqrt(coef_b^2 - 4 * coef_a * coef_c)) / (2 * coef_a)
        return p_cauchy + tau * d
    end
end

"""
    otimizar_regiao_confianca(f, ∇f, hess, x0; delta0=0.5, delta_max=2.0, max_iter=50, tol=1e-8)

Implementação enxuta (fins didáticos) de um método de Região de Confiança com
passo de dogleg, que registra cada iteração num `Vector{PassoRC}` — pronto
para ser consumido por `visualizar_frames_dinamicos`.

Retorna `(x_final, historico)`.
"""
function otimizar_regiao_confianca(
    f, ∇f, hess, x0::Vector{Float64};
    delta0::Float64 = 0.5, delta_max::Float64 = 2.0,
    max_iter::Int = 50, tol::Float64 = 1e-8,
)
    x = copy(x0)
    delta = delta0
    historico = PassoRC[]

    for iter in 1:max_iter
        g = ∇f(x)
        norm(g) < tol && break

        B = hess(x)
        p = passo_dogleg(g, B, delta)
        x_candidato = x + p

        # Redução real (ared) vs. redução prevista pelo modelo quadrático (pred)
        ared = f(x) - f(x_candidato)
        pred = -(dot(g, p) + 0.5 * dot(p, B * p))
        razao = pred > 1e-14 ? ared / pred : -Inf

        aceito = razao > 0.1
        status = aceito ? :aceito : :rejeitado

        push!(historico, PassoRC(iter, copy(x), x_candidato, delta, status, ared, pred))

        # Atualização clássica do raio da região de confiança
        if razao < 0.25
            delta *= 0.25
        elseif razao > 0.75 && norm(p) >= 0.9 * delta
            delta = min(2 * delta, delta_max)
        end

        aceito && (x = x_candidato)
    end

    return x, historico
end

# ------------------------------------------------------------------------------
# 3. Execução do exemplo, ponta a ponta
# ------------------------------------------------------------------------------

x0 = [-1.2, 1.0]  # ponto inicial clássico para testar a Rosenbrock

x_final, historico = otimizar_regiao_confianca(
    rosenbrock, grad_rosenbrock, hess_rosenbrock, x0,
)

println("Ponto inicial:                    ", x0)
println("Ponto final:                      ", x_final)
println("f(x final):                       ", rosenbrock(x_final))
println("Total de iterações registradas:   ", length(historico))

prob = Problema(rosenbrock, grad_rosenbrock, "Rosenbrock (a=1, b=100)")

visualizar_frames_dinamicos(
    prob, historico;
    output_dir = joinpath(@__DIR__, "frames_exemplo"),
)

# Descomente a linha abaixo para gerar um GIF a partir dos quadros
# (requer o ffmpeg instalado no sistema):
#
# gerar_gif(joinpath(@__DIR__, "frames_exemplo");
#           nome_arquivo = joinpath(@__DIR__, "animacao.gif"))
