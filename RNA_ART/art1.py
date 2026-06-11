"""
Rede ART-1 (Adaptive Resonance Theory 1)
Classificação de situações de um processo industrial.
"""

import numpy as np
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')


def art1_train(patterns, rho, L=2.0):
    """
    Treina uma rede ART-1.

    Parâmetros:
        patterns: lista de vetores binários (numpy arrays)
        rho: parâmetro de vigilância (0 < rho <= 1)
        L: parâmetro L > 1 (default=2.0)

    Retorna:
        clusters: dicionário {cluster_id: [índices das situações]}
        weights_top_down: pesos top-down (Tj→i) para cada cluster
        weights_bottom_up: pesos bottom-up (Bi→j) para cada cluster
    """
    n = len(patterns[0])  # dimensão dos padrões (16)
    
    # Inicialização
    # Pesos top-down (T): inicializados com 1s
    # Pesos bottom-up (B): inicializados com L / (L - 1 + n)
    top_down = []  # lista de vetores de pesos top-down para cada neurônio de saída
    bottom_up = []  # lista de vetores de pesos bottom-up para cada neurônio de saída
    
    clusters = {}  # cluster_id -> lista de índices de padrões
    num_clusters = 0
    
    # Registrar histórico detalhado
    history = []
    
    for idx, pattern in enumerate(patterns):
        sit_name = f"Situação {idx + 1}"
        step_log = {"situacao": sit_name, "pattern": pattern.tolist(), "steps": []}
        
        assigned = False
        
        # Calcular ativação para cada neurônio de saída existente
        # Ordenar por ativação (maior primeiro) para testar na ordem
        if num_clusters > 0:
            activations = []
            for j in range(num_clusters):
                # Ativação = B_j · I (produto escalar bottom-up com entrada)
                activation = np.dot(bottom_up[j], pattern)
                activations.append((j, activation))
            
            # Ordenar por ativação decrescente
            activations.sort(key=lambda x: x[1], reverse=True)
            
            inhibited = set()
            
            for j, activation in activations:
                if j in inhibited:
                    continue
                
                # Teste de vigilância
                # Calcular |T_j ∩ I| / |I|
                intersection = np.logical_and(top_down[j], pattern).astype(float)
                match_ratio = np.sum(intersection) / np.sum(pattern)
                
                step_log["steps"].append({
                    "cluster_testado": j + 1,
                    "ativacao": float(activation),
                    "intersecao_T_I": int(np.sum(intersection)),
                    "norma_I": int(np.sum(pattern)),
                    "match_ratio": float(match_ratio),
                    "rho": float(rho),
                    "resultado": ""
                })
                
                if match_ratio >= rho:
                    # Ressonância! Atualizar pesos
                    # T_j(novo) = T_j(antigo) ∩ I
                    top_down[j] = intersection
                    # B_j(novo) = L * (T_j ∩ I) / (L - 1 + |T_j ∩ I|)
                    norm_intersection = np.sum(intersection)
                    bottom_up[j] = (L * intersection) / (L - 1 + norm_intersection)
                    
                    clusters[j].append(idx)
                    assigned = True
                    step_log["steps"][-1]["resultado"] = f"RESSONÂNCIA → atribuído ao Cluster {j + 1}"
                    break
                else:
                    # Reset: inibir este neurônio e tentar o próximo
                    inhibited.add(j)
                    step_log["steps"][-1]["resultado"] = f"RESET (match_ratio={match_ratio:.4f} < rho={rho})"
        
        if not assigned:
            # Criar novo cluster
            # T_j = I (entrada como pesos top-down)
            top_down.append(pattern.copy().astype(float))
            # B_j = L * I / (L - 1 + |I|)
            norm_I = np.sum(pattern)
            bottom_up.append((L * pattern.copy().astype(float)) / (L - 1 + norm_I))
            
            clusters[num_clusters] = [idx]
            step_log["steps"].append({
                "resultado": f"NOVO Cluster {num_clusters + 1} criado"
            })
            num_clusters += 1
        
        history.append(step_log)
    
    return clusters, top_down, bottom_up, history


def format_results(clusters, history, rho, patterns):
    """Formata os resultados para exibição."""
    lines = []
    lines.append(f"## Parâmetro de Vigilância ρ = {rho}")
    lines.append("")
    
    # Número de classes ativas
    num_classes = len(clusters)
    lines.append(f"**Número de classes ativas:** {num_classes}")
    lines.append("")
    
    # Tabela de agrupamentos
    lines.append("### Agrupamentos")
    lines.append("")
    lines.append("| Classe | Situações Agrupadas | Padrão Representativo (pesos top-down) |")
    lines.append("|--------|---------------------|----------------------------------------|")
    
    for cluster_id in sorted(clusters.keys()):
        sit_list = [f"Situação {i + 1}" for i in clusters[cluster_id]]
        sits_str = ", ".join(sit_list)
        lines.append(f"| Classe {cluster_id + 1} | {sits_str} | (ver detalhes abaixo) |")
    
    lines.append("")
    
    # Detalhes passo a passo
    lines.append("### Detalhamento do Treinamento (passo a passo)")
    lines.append("")
    
    for step in history:
        lines.append(f"**{step['situacao']}** — Entrada: `{step['pattern']}`")
        lines.append("")
        for s in step["steps"]:
            if "cluster_testado" in s:
                lines.append(f"- Testou Classe {s['cluster_testado']}: "
                           f"|T∩I|={s['intersecao_T_I']}, |I|={s['norma_I']}, "
                           f"match_ratio={s['match_ratio']:.4f}, ρ={s['rho']} → **{s['resultado']}**")
            else:
                lines.append(f"- **{s['resultado']}**")
        lines.append("")
    
    lines.append("---")
    lines.append("")
    
    return "\n".join(lines)


def main():
    # Definir os 10 padrões (situações) com 16 variáveis binárias
    patterns = np.array([
        [0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1],  # Situação 1
        [1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0],  # Situação 2
        [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1],  # Situação 3
        [1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0],  # Situação 4
        [0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1],  # Situação 5
        [1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1],  # Situação 6
        [1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0],  # Situação 7
        [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1],  # Situação 8
        [0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],  # Situação 9
        [0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1],  # Situação 10
    ], dtype=float)
    
    vigilance_values = [0.5, 0.8, 0.9, 0.99]
    
    all_results = []
    
    # Cabeçalho do markdown
    md = []
    md.append("# Trabalho — Rede ART-1: Classificação de Situações de Processo Industrial")
    md.append("")
    md.append("**Disciplina:** Lab. Inteligência Artificial")
    md.append("")
    md.append("**Professor:** Lázaro Eduardo da Silva")
    md.append("")
    md.append("**Data:** 11/06/2026")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Descrição do Problema")
    md.append("")
    md.append("O comportamento de um processo industrial é analisado considerando 16 variáveis de status "
              "binárias ao longo de 10 situações. A rede ART-1 é utilizada para classificar e agrupar "
              "as situações \"parecidas\", de modo a obter um provável diagnóstico para eventual manutenção.")
    md.append("")
    md.append("## Dados de Entrada")
    md.append("")
    md.append("| Situação | x1 | x2 | x3 | x4 | x5 | x6 | x7 | x8 | x9 | x10 | x11 | x12 | x13 | x14 | x15 | x16 |")
    md.append("|----------|----|----|----|----|----|----|----|----|----|----|-----|-----|-----|-----|-----|-----|")
    
    for i, p in enumerate(patterns):
        vals = " | ".join([str(int(v)) for v in p])
        md.append(f"| Situação {i+1} | {vals} |")
    
    md.append("")
    md.append("## Parâmetros da Rede ART-1")
    md.append("")
    md.append("- **Dimensão de entrada (n):** 16")
    md.append("- **Parâmetro L:** 2.0")
    md.append("- **Inicialização dos pesos top-down (T):** todos 1s (ou igual à primeira entrada do cluster)")
    md.append("- **Inicialização dos pesos bottom-up (B):** L / (L - 1 + n)")
    md.append("- **Critério de vigilância:** |T_j ∩ I| / |I| ≥ ρ")
    md.append("")
    md.append("---")
    md.append("")
    
    for rho in vigilance_values:
        clusters, top_down, bottom_up, history = art1_train(patterns, rho)
        result_text = format_results(clusters, history, rho, patterns)
        md.append(result_text)
    
    # Análise comparativa
    md.append("## Análise Comparativa")
    md.append("")
    md.append("| ρ (Vigilância) | Nº de Classes Ativas | Observação |")
    md.append("|----------------|----------------------|------------|")
    
    summaries = []
    for rho in vigilance_values:
        clusters, _, _, _ = art1_train(patterns, rho)
        n_classes = len(clusters)
        
        if rho <= 0.5:
            obs = "Baixa vigilância — agrupamentos mais amplos (generalização)"
        elif rho <= 0.8:
            obs = "Vigilância moderada — equilíbrio entre generalização e especificidade"
        elif rho <= 0.9:
            obs = "Alta vigilância — classes mais específicas"
        else:
            obs = "Vigilância muito alta — quase cada padrão é uma classe separada"
        
        md.append(f"| {rho} | {n_classes} | {obs} |")
        summaries.append((rho, n_classes))
    
    md.append("")
    md.append("### Conclusão")
    md.append("")
    md.append("À medida que o parâmetro de vigilância **ρ** aumenta, a rede ART-1 se torna mais exigente "
              "quanto à similaridade necessária para que um padrão seja incorporado a um cluster existente. "
              "Isso resulta em:")
    md.append("")
    md.append("- **ρ baixo (0.5):** Poucos clusters, agrupamentos amplos. A rede generaliza bastante, "
              "agrupando situações que têm pelo menos 50% de semelhança.")
    md.append("- **ρ moderado (0.8):** Mais clusters surgem, com agrupamentos mais coerentes.")
    md.append("- **ρ alto (0.9):** Clusters muito específicos, apenas situações muito similares ficam juntas.")
    md.append("- **ρ muito alto (0.99):** Quase cada situação forma seu próprio cluster, "
              "exceto pares praticamente idênticos (como Situações 3 e 8, e Situações 5 e 10).")
    md.append("")
    md.append("Para fins de **diagnóstico de manutenção**, um valor intermediário de ρ (como 0.8) "
              "pode oferecer o melhor equilíbrio entre agrupar situações suficientemente parecidas "
              "sem perder informação relevante de diferenças.")
    md.append("")
    
    # Salvar arquivo markdown
    output = "\n".join(md)
    with open("respostas_ART1.md", "w", encoding="utf-8") as f:
        f.write(output)
    
    print(output)
    print("\n\nArquivo 'respostas_ART1.md' gerado com sucesso!")


if __name__ == "__main__":
    main()
