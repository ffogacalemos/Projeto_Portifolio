"""Function-calling / tool-use — registro de tools usadas pelo agente."""

from __future__ import annotations

import json
from typing import Any, Callable


# ============================================================================
# TODO 4 — Tool especifica do dominio: cite_article
# Retorna o texto integral de artigos da LGPD para evitar alucinacoes do LLM.
# ============================================================================


def cite_article(article_number: int) -> str:
    """
    Retorna o texto integral de artigos especificos da LGPD.

    CORRECAO: versao original cobria apenas arts. 1, 2, 7, 11, 18.
    Expandido para cobrir os artigos mais consultados em cenarios de
    compliance: definicoes (5), principios (6), consentimento (8, 9),
    dados de criancas (14), direitos (17, 18, 20), seguranca (46, 48)
    e sancoes (52).
    """

    lgpd_db: dict[int, str] = {
        1: (
            "Art. 1º Esta Lei dispoe sobre o tratamento de dados pessoais, inclusive nos meios digitais, "
            "por pessoa natural ou por pessoa juridica de direito publico ou privado, com o objetivo de "
            "proteger os direitos fundamentais de liberdade e de privacidade e o livre desenvolvimento "
            "da personalidade da pessoa natural."
        ),
        2: (
            "Art. 2º A disciplina da protecao de dados pessoais tem como fundamentos: "
            "I - o respeito a privacidade; II - a autodeterminacao informativa; "
            "III - a liberdade de expressao, de informacao, de comunicacao e de opiniao; "
            "IV - a inviolabilidade da intimidade, da honra e da imagem; "
            "V - o desenvolvimento economico e tecnologico e a inovacao; "
            "VI - a livre iniciativa, a livre concorrencia e a defesa do consumidor; e "
            "VII - os direitos humanos, o livre desenvolvimento da personalidade, a dignidade e o "
            "exercicio da cidadania pelas pessoas naturais."
        ),
        5: (
            "Art. 5º Para os fins desta Lei, considera-se: "
            "I - dado pessoal: informacao relacionada a pessoa natural identificada ou identificavel; "
            "II - dado pessoal sensivel: dado pessoal sobre origem racial ou etnica, conviccao religiosa, "
            "opiniao politica, filiacao a sindicato, dado referente a saude ou a vida sexual, dado "
            "genetico ou biometrico, quando vinculado a uma pessoa natural; "
            "III - dado anonimizado: dado relativo a titular que nao possa ser identificado; "
            "V - titular: pessoa natural a quem se referem os dados pessoais que sao objeto de tratamento; "
            "VI - controlador: pessoa natural ou juridica a quem competem as decisoes referentes ao "
            "tratamento de dados pessoais; "
            "VII - operador: pessoa natural ou juridica que realiza o tratamento de dados pessoais em "
            "nome do controlador; "
            "X - tratamento: toda operacao realizada com dados pessoais, como coleta, producao, recepcao, "
            "classificacao, utilizacao, acesso, reproducao, transmissao, distribuicao, processamento, "
            "arquivamento, armazenamento, eliminacao, avaliacao ou controle da informacao, modificacao, "
            "comunicacao, transferencia, difusao ou extracao; "
            "XII - consentimento: manifestacao livre, informada e inequivoca pela qual o titular concorda "
            "com o tratamento de seus dados pessoais para uma finalidade determinada."
        ),
        6: (
            "Art. 6º As atividades de tratamento de dados pessoais deverao observar a boa-fe e os seguintes principios: "
            "I - finalidade: realizacao do tratamento para propositos legitimos, especificos, explicitos e informados ao titular; "
            "II - adequacao: compatibilidade do tratamento com as finalidades informadas ao titular; "
            "III - necessidade: limitacao do tratamento ao minimo necessario para a realizacao de suas finalidades; "
            "IV - livre acesso: garantia aos titulares de consulta facilitada e gratuita sobre a forma e duracao do tratamento; "
            "V - qualidade dos dados: garantia de exatidao, clareza, relevancia e atualizacao dos dados; "
            "VI - transparencia: garantia de informacoes claras, precisas e facilmente acessiveis sobre o tratamento; "
            "VII - seguranca: utilizacao de medidas tecnicas e administrativas aptas a proteger os dados pessoais; "
            "VIII - prevencao: adocao de medidas para prevenir a ocorrencia de danos; "
            "IX - nao discriminacao: impossibilidade de realizacao do tratamento para fins discriminatorios; "
            "X - responsabilizacao e prestacao de contas: demonstracao da adocao de medidas eficazes."
        ),
        7: (
            "Art. 7º O tratamento de dados pessoais somente podera ser realizado nas seguintes hipoteses: "
            "I - mediante o fornecimento de consentimento pelo titular; "
            "II - para o cumprimento de obrigacao legal ou regulatoria pelo controlador; "
            "III - pela administracao publica, para o tratamento e uso compartilhado de dados necessarios "
            "a execucao de politicas publicas previstas em leis e regulamentos; "
            "IV - para a realizacao de estudos por orgao de pesquisa, garantida a anonimizacao sempre que possivel; "
            "V - quando necessario para a execucao de contrato do qual seja parte o titular; "
            "VI - para o exercicio regular de direitos em processo judicial, administrativo ou arbitral; "
            "VII - para a protecao da vida ou da incolumidade fisica do titular ou de terceiro; "
            "VIII - para a tutela da saude, em procedimento realizado por profissionais de saude; "
            "IX - quando necessario para atender aos interesses legitimos do controlador ou de terceiro; "
            "X - para a protecao do credito."
        ),
        8: (
            "Art. 8º O consentimento previsto no inciso I do art. 7º devera ser fornecido por escrito ou "
            "por outro meio que demonstre a manifestacao de vontade do titular. "
            "§ 1º Caso o consentimento seja fornecido por escrito, esse devera constar de clausula destacada. "
            "§ 2º Cabe ao controlador o onus da prova de que o consentimento foi obtido em conformidade com esta Lei. "
            "§ 3º E vedado o tratamento de dados pessoais mediante vicio de consentimento. "
            "§ 4º O consentimento devera referir-se a finalidades determinadas; autorizacoes genericas serao nulas. "
            "§ 5º O consentimento pode ser revogado a qualquer momento mediante manifestacao expressa do titular, "
            "por procedimento gratuito e facilitado."
        ),
        11: (
            "Art. 11. O tratamento de dados pessoais sensiveis somente podera ocorrer nas seguintes hipoteses: "
            "I - quando o titular ou seu responsavel legal consentir, de forma especifica e destacada, para finalidades especificas; "
            "II - sem fornecimento de consentimento do titular, nas hipoteses em que for indispensavel para: "
            "a) cumprimento de obrigacao legal ou regulatoria pelo controlador; "
            "b) tratamento compartilhado de dados necessarios a execucao de politicas publicas; "
            "c) realizacao de estudos por orgao de pesquisa, garantida a anonimizacao sempre que possivel; "
            "d) exercicio regular de direitos, inclusive em contrato e em processo judicial; "
            "e) protecao da vida ou da incolumidade fisica do titular ou de terceiro; "
            "f) tutela da saude, em procedimento realizado por profissionais de saude; "
            "g) garantia da prevencao a fraude e a seguranca do titular nos processos de identificacao e autenticacao."
        ),
        14: (
            "Art. 14. O tratamento de dados pessoais de criancas e de adolescentes devera ser realizado em seu "
            "melhor interesse. "
            "§ 1º O tratamento de dados pessoais de criancas devera ser realizado com o consentimento especifico "
            "e em destaque dado por pelo menos um dos pais ou pelo responsavel legal. "
            "§ 4º Os controladores nao deverao condicionar a participacao de criancas em jogos ou aplicacoes de "
            "internet ao fornecimento de informacoes pessoais alem das estritamente necessarias."
        ),
        17: (
            "Art. 17. Toda pessoa natural tem assegurada a titularidade de seus dados pessoais e garantidos os "
            "direitos fundamentais de liberdade, de intimidade e de privacidade, nos termos desta Lei."
        ),
        18: (
            "Art. 18. O titular dos dados pessoais tem direito a obter do controlador, em relacao aos dados do "
            "titular por ele tratados, a qualquer momento e mediante requisicao: "
            "I - confirmacao da existencia de tratamento; "
            "II - acesso aos dados; "
            "III - correcao de dados incompletos, inexatos ou desatualizados; "
            "IV - anonimizacao, bloqueio ou eliminacao de dados desnecessarios, excessivos ou tratados em desconformidade; "
            "V - portabilidade dos dados a outro fornecedor de servico ou produto; "
            "VI - eliminacao dos dados pessoais tratados com o consentimento do titular; "
            "VII - informacao das entidades publicas e privadas com as quais o controlador realizou uso compartilhado de dados; "
            "VIII - informacao sobre a possibilidade de nao fornecer consentimento e sobre as consequencias da negativa; "
            "IX - revogacao do consentimento."
        ),
        20: (
            "Art. 20. O titular dos dados tem direito a solicitar a revisao de decisoes tomadas unicamente com base "
            "em tratamento automatizado de dados pessoais que afetem seus interesses, incluidas as decisoes destinadas "
            "a definir o seu perfil pessoal, profissional, de consumo e de credito ou os aspectos de sua personalidade. "
            "§ 1º O controlador devera fornecer, sempre que solicitadas, informacoes claras e adequadas a respeito dos "
            "criterios e dos procedimentos utilizados para a decisao automatizada."
        ),
        46: (
            "Art. 46. Os agentes de tratamento devem adotar medidas de seguranca, tecnicas e administrativas aptas a "
            "proteger os dados pessoais de acessos nao autorizados e de situacoes acidentais ou ilicitas de destruicao, "
            "perda, alteracao, comunicacao ou qualquer forma de tratamento inadequado ou ilicito. "
            "§ 2º As medidas deverao ser observadas desde a fase de concepcao do produto ou do servico ate a sua execucao."
        ),
        48: (
            "Art. 48. O controlador devera comunicar a autoridade nacional e ao titular a ocorrencia de incidente de "
            "seguranca que possa acarretar risco ou dano relevante aos titulares. "
            "§ 1º A comunicacao sera feita em prazo razoavel e devera mencionar, no minimo: "
            "I - a descricao da natureza dos dados pessoais afetados; "
            "II - as informacoes sobre os titulares envolvidos; "
            "III - as medidas tecnicas e de seguranca utilizadas para a protecao dos dados; "
            "IV - os riscos relacionados ao incidente; "
            "VI - as medidas que foram ou que serao adotadas para reverter ou mitigar os efeitos do prejuizo."
        ),
        52: (
            "Art. 52. Os agentes de tratamento de dados, em razao das infracoes cometidas as normas previstas nesta Lei, "
            "ficam sujeitos as seguintes sancoes administrativas: "
            "I - advertencia, com indicacao de prazo para adocao de medidas corretivas; "
            "II - multa simples, de ate 2% do faturamento da pessoa juridica no Brasil no seu ultimo exercicio, "
            "limitada, no total, a R$ 50.000.000,00 por infracao; "
            "III - multa diaria, observado o limite total; "
            "IV - publicizacao da infracao apos devidamente apurada e confirmada; "
            "V - bloqueio dos dados pessoais a que se refere a infracao ate a sua regularizacao; "
            "VI - eliminacao dos dados pessoais a que se refere a infracao; "
            "X - suspensao parcial do funcionamento do banco de dados pelo periodo maximo de 6 meses; "
            "XI - suspensao do exercicio da atividade de tratamento dos dados pelo periodo maximo de 6 meses; "
            "XII - proibicao parcial ou total do exercicio de atividades relacionadas a tratamento de dados."
        ),
    }

    texto = lgpd_db.get(article_number)

    if texto:
        return texto
    else:
        return (
            f"O Artigo {article_number} nao esta no cache rapido desta tool. "
            "Analise os chunks retornados pelo RAG no corpus principal para responder ao usuario. "
            "Artigos disponiveis nesta tool: "
            + ", ".join(str(n) for n in sorted(lgpd_db.keys()))
            + "."
        )


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "cite_article",
            "description": (
                "Retorna o texto integral de artigos estruturais da LGPD (Lei 13.709/2018). "
                "Acione esta tool se o usuario pedir explicitamente o conteudo exato de um artigo "
                "ou quando precisar confirmar a base legal de uma hipotese de tratamento. "
                "Artigos disponiveis: 1, 2, 5, 6, 7, 8, 11, 14, 17, 18, 20, 46, 48, 52."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "article_number": {
                        "type": "integer",
                        "description": "O numero do artigo da LGPD a ser consultado (ex: 7, 18, 52).",
                    },
                },
                "required": ["article_number"],
            },
        },
    },
]

TOOL_REGISTRY: dict[str, Callable[..., str]] = {
    "cite_article": cite_article,
}


def run_tool_call(name: str, arguments_json: str) -> str:
    """Executa uma tool call e retorna o resultado como string."""
    if name not in TOOL_REGISTRY:
        return f"ERROR: tool '{name}' nao registrada"
    try:
        kwargs = json.loads(arguments_json)
        return TOOL_REGISTRY[name](**kwargs)
    except Exception as e:
        return f"ERROR ao executar {name}: {e}"
