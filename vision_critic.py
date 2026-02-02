"""
vision_critic.py - Analisador de Outputs Visuais com Contexto de Intencao

Este modulo analisa graficos e outputs visuais usando Vision AI,
injetando contexto de intencao para analises mais precisas.

O segredo: Nao perguntar "o que voce ve?" mas sim
"O autor queria verificar X. A imagem confirma isso?"
"""

import os
import base64
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class IntentContext:
    """Contexto de intencao para analise de grafico."""
    intent: str           # O que estamos verificando
    expected: str         # O que esperamos ver
    alerts: List[str]     # O que indicaria problema
    decision_needed: str  # Que decisao deve ser tomada
    business_context: str = ""  # Contexto de negocio adicional


class VisionCritic:
    """
    Analisador de outputs visuais com Vision AI.
    
    Usa intent injection para obter analises mais uteis:
    - Nao: "Descreva este grafico"
    - Sim: "O autor queria verificar outliers. Ha outliers criticos?"
    """
    
    # Templates de intent por tipo de grafico
    INTENT_TEMPLATES = {
        "histogram": IntentContext(
            intent="Analisar distribuicao da feature",
            expected="Distribuicao aproximadamente normal ou log-normal",
            alerts=[
                "Outliers extremos (pontos isolados)",
                "Bimodalidade (dois picos) - pode indicar grupos distintos",
                "Concentracao em zero - feature sparse",
                "Assimetria forte (skewness > 2)"
            ],
            decision_needed="Aplicar transformacao (log, sqrt, binning) ou remover outliers?"
        ),
        
        "roc_curve": IntentContext(
            intent="Avaliar poder discriminativo do modelo",
            expected="Curvas bem acima da diagonal, gap treino-teste < 8%",
            alerts=[
                "Gap > 8% entre treino e teste = OVERFITTING",
                "Curvas proximas da diagonal = modelo fraco",
                "AUC < 0.75 = revisar features",
                "Cruzamento de curvas = instabilidade"
            ],
            decision_needed="Aceitar modelo, aumentar regularizacao, ou revisar features?"
        ),
        
        "null_barplot": IntentContext(
            intent="Identificar features com dados faltantes criticos",
            expected="Maioria das features com menos de 5% de nulos",
            alerts=[
                ">50% nulos = remover feature",
                "5-50% nulos = investigar imputacao",
                "Padrao de nulos correlacionado = MNAR",
                "Feature critica com muitos nulos = imputacao especial"
            ],
            decision_needed="Lista de features para remover ou imputar"
        ),
        
        "correlation_matrix": IntentContext(
            intent="Identificar multicolinearidade e redundancia",
            expected="Correlacoes < 0.9 entre features, algumas com target",
            alerts=[
                "Correlacao > 0.95 = features redundantes, remover uma",
                "Clusters de features = criar feature agregada",
                "Nenhuma correlacao com target = features fracas",
                "Correlacao perfeita (1.0) = possivel leakage"
            ],
            decision_needed="Lista de features redundantes para remover"
        ),
        
        "profit_curve": IntentContext(
            intent="Encontrar threshold otimo de lucro",
            expected="Pico de lucro claro, eficiencia > 75%",
            alerts=[
                "Curva plana = modelo nao discrimina bem",
                "Multiplos picos = instabilidade",
                "Lucro negativo em todo range = problema grave",
                "Eficiencia < 75% = revisar calibracao"
            ],
            decision_needed="Threshold otimo e se modelo e viavel financeiramente"
        ),
        
        "learning_curve": IntentContext(
            intent="Diagnosticar bias vs variance",
            expected="Curvas convergindo, gap pequeno no final",
            alerts=[
                "Gap grande persistente = overfitting",
                "Ambas curvas baixas = underfitting",
                "Treino perfeito (1.0) = overfitting severo",
                "Curvas nao convergem = precisa mais dados"
            ],
            decision_needed="Ajustar complexidade do modelo ou coletar mais dados?"
        ),
        
        "feature_importance": IntentContext(
            intent="Identificar features mais relevantes",
            expected="Top 10-20 features com importancia significativa",
            alerts=[
                "Uma feature dominante = possivel leakage",
                "Importancias muito uniformes = modelo incerto",
                "Features inesperadas no top = investigar",
                "Features de negocio ausentes = revisar engenharia"
            ],
            decision_needed="Validar se top features fazem sentido de negocio"
        ),
        
        "shap_beeswarm": IntentContext(
            intent="Entender direcao do impacto das features",
            expected="Cores separadas (vermelho/azul) indicando relacao clara",
            alerts=[
                "Cores misturadas = relacao nao-linear ou ruido",
                "Feature com impacto apenas em uma direcao = verificar",
                "Separacao perfeita de cores = possivel leakage",
                "Impacto muito concentrado = feature dominante"
            ],
            decision_needed="Validar se relacoes feature-target fazem sentido"
        ),
        
        "psi_chart": IntentContext(
            intent="Detectar drift na distribuicao de scores",
            expected="PSI < 0.1 (excelente) ou < 0.2 (aceitavel)",
            alerts=[
                "PSI > 0.2 = drift significativo",
                "PSI > 0.25 = acao urgente necessaria",
                "Bins extremos com alto PSI = caudas mudaram",
                "PSI crescente ao longo do tempo = degradacao"
            ],
            decision_needed="Retreinar modelo ou ajustar features?"
        ),
        
        "confusion_matrix": IntentContext(
            intent="Avaliar erros do modelo por classe",
            expected="Diagonal forte, poucos falsos positivos/negativos",
            alerts=[
                "Muitos falsos negativos = modelo conservador demais",
                "Muitos falsos positivos = modelo agressivo demais",
                "Classe desbalanceada dominando = ajustar threshold",
                "Erros concentrados = revisar features dessa classe"
            ],
            decision_needed="Ajustar threshold ou custo de erros?"
        )
    }
    
    def __init__(self):
        from dotenv import load_dotenv
        load_dotenv()
        
        self.gemini_key = os.getenv("GEMINI_KEY")
        self.model_name = os.getenv("MODEL_NAME", "gemini-1.5-flash")  # Usar MODEL_NAME do .env
        self.openai_key = os.getenv("OPENAI_API_KEY")
        
        if self.gemini_key:
            self.provider = "gemini"
            print(f"[VISION] Usando Gemini: {self.model_name}")
        elif self.openai_key:
            self.provider = "openai"
        else:
            self.provider = "mock"
            print("[VISION] AVISO: Nenhuma API key. Modo simulado.")
    
    def _encode_image(self, image_path: str) -> str:
        """Codifica imagem em base64."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    def _detect_plot_type(self, image_path: str, code_context: str = "") -> str:
        """Detecta tipo de grafico baseado no nome/codigo."""
        path_lower = image_path.lower()
        code_lower = code_context.lower()
        
        if "roc" in path_lower or "roc_curve" in code_lower:
            return "roc_curve"
        elif "null" in path_lower or "missing" in path_lower or "isnull" in code_lower:
            return "null_barplot"
        elif "corr" in path_lower or "correlation" in code_lower or "heatmap" in code_lower:
            return "correlation_matrix"
        elif "profit" in path_lower or "lucro" in path_lower:
            return "profit_curve"
        elif "learning" in path_lower:
            return "learning_curve"
        elif "importance" in path_lower or "feature_import" in code_lower:
            return "feature_importance"
        elif "shap" in path_lower or "beeswarm" in code_lower:
            return "shap_beeswarm"
        elif "psi" in path_lower or "stability" in path_lower:
            return "psi_chart"
        elif "confusion" in path_lower or "confusion_matrix" in code_lower:
            return "confusion_matrix"
        elif "hist" in path_lower or "histplot" in code_lower or "distplot" in code_lower:
            return "histogram"
        else:
            return "histogram"  # Default
    
    def _build_intent_prompt(self, intent: IntentContext, custom_context: str = "") -> str:
        """Constroi prompt com contexto de intencao."""
        
        prompt = f"""Voce e um Senior Data Scientist analisando um grafico de um pipeline de Credit Scoring.

## CONTEXTO DE INTENCAO
O autor deste grafico queria: **{intent.intent}**

## O QUE ERA ESPERADO
{intent.expected}

## ALERTAS A PROCURAR
"""
        for alert in intent.alerts:
            prompt += f"- {alert}\n"
        
        prompt += f"""
## DECISAO NECESSARIA
{intent.decision_needed}

"""
        if custom_context:
            prompt += f"""## CONTEXTO ADICIONAL DO CODIGO
{custom_context}

"""
        
        prompt += """## SUA TAREFA
Analise a imagem e responda:

1. **OBSERVACAO**: O que voce ve no grafico? (2-3 frases objetivas)
2. **VALIDACAO**: O resultado confirma o esperado ou ha alertas?
3. **ALERTAS**: Liste qualquer alerta detectado
4. **DECISAO**: Qual acao recomendada baseada na analise?
5. **CONFIANCA**: Qual sua confianca na analise? (Alta/Media/Baixa)

Seja direto e tecnico. Foque em insights acionaveis.
"""
        return prompt
    
    def analyze_plot(self, image_path: str, custom_intent: str = "", code_context: str = "") -> Dict[str, Any]:
        """
        Analisa um grafico com Vision AI.
        
        Args:
            image_path: Caminho para a imagem
            custom_intent: Intent customizado (sobrescreve deteccao automatica)
            code_context: Codigo que gerou o plot (para contexto)
        
        Returns:
            Dict com analise estruturada
        """
        if not Path(image_path).exists():
            return {
                "error": f"Imagem nao encontrada: {image_path}",
                "success": False
            }
        
        # Detectar tipo de grafico
        plot_type = self._detect_plot_type(image_path, code_context)
        
        # Pegar template de intent ou usar customizado
        if custom_intent:
            intent = IntentContext(
                intent=custom_intent,
                expected="A ser determinado pela analise",
                alerts=["Qualquer anomalia ou padrao inesperado"],
                decision_needed="Determinar se resultado e aceitavel"
            )
        else:
            intent = self.INTENT_TEMPLATES.get(plot_type, self.INTENT_TEMPLATES["histogram"])
        
        # Extrair intent do codigo se disponivel
        if code_context:
            # Buscar comentarios [INTENT], [ESPERADO], [ALERTA]
            intent_match = re.search(r'\[INTENT\]\s*(.+?)(?=\n|\[|$)', code_context)
            expected_match = re.search(r'\[ESPERADO\]\s*(.+?)(?=\n|\[|$)', code_context)
            alert_match = re.search(r'\[ALERTA\]\s*(.+?)(?=\n|\[|$)', code_context)
            
            if intent_match:
                intent.intent = intent_match.group(1).strip()
            if expected_match:
                intent.expected = expected_match.group(1).strip()
            if alert_match:
                intent.alerts.insert(0, alert_match.group(1).strip())
        
        # Construir prompt
        prompt = self._build_intent_prompt(intent, code_context)
        
        # Chamar Vision API
        if self.provider == "gemini":
            response = self._call_gemini_vision(image_path, prompt)
        elif self.provider == "openai":
            response = self._call_openai_vision(image_path, prompt)
        else:
            response = self._mock_vision_response(plot_type)
        
        return {
            "success": True,
            "plot_type": plot_type,
            "intent": intent.intent,
            "analysis": response,
            "image_path": image_path
        }
    
    def _call_gemini_vision(self, image_path: str, prompt: str) -> str:
        """Chama Gemini Vision API."""
        import google.generativeai as genai
        
        genai.configure(api_key=self.gemini_key)
        model = genai.GenerativeModel(self.model_name)  # Usar MODEL_NAME do .env
        
        # Carregar imagem
        import PIL.Image
        image = PIL.Image.open(image_path)
        
        response = model.generate_content([prompt, image])
        return response.text
    
    def _call_openai_vision(self, image_path: str, prompt: str) -> str:
        """Chama OpenAI Vision API."""
        from openai import OpenAI
        
        client = OpenAI(api_key=self.openai_key)
        
        # Codificar imagem
        base64_image = self._encode_image(image_path)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    
    def _mock_vision_response(self, plot_type: str) -> str:
        """Resposta mock para testes sem API."""
        return f"""## Analise (Modo Simulado)

**OBSERVACAO**: Grafico do tipo {plot_type} detectado. Analise visual nao disponivel sem API key.

**VALIDACAO**: Nao foi possivel validar automaticamente.

**ALERTAS**: Nenhum (modo simulado)

**DECISAO**: Configure GEMINI_KEY ou OPENAI_API_KEY para analise real.

**CONFIANCA**: Baixa (simulado)
"""
    
    def analyze_multiple(self, image_paths: List[str], code_context: str = "") -> List[Dict[str, Any]]:
        """Analisa multiplos graficos."""
        results = []
        for path in image_paths:
            result = self.analyze_plot(path, code_context=code_context)
            results.append(result)
        return results
    
    def summarize_analyses(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Resume multiplas analises em decisoes acionaveis."""
        
        all_alerts = []
        all_decisions = []
        
        for analysis in analyses:
            if analysis.get("success"):
                text = analysis.get("analysis", "")
                
                # Extrair alertas
                alert_section = re.search(r'\*\*ALERTAS?\*\*:?\s*(.+?)(?=\*\*|$)', text, re.DOTALL)
                if alert_section:
                    all_alerts.append(alert_section.group(1).strip())
                
                # Extrair decisoes
                decision_section = re.search(r'\*\*DECIS[AÃ]O\*\*:?\s*(.+?)(?=\*\*|$)', text, re.DOTALL)
                if decision_section:
                    all_decisions.append(decision_section.group(1).strip())
        
        return {
            "total_plots": len(analyses),
            "successful": sum(1 for a in analyses if a.get("success")),
            "alerts": all_alerts,
            "decisions": all_decisions,
            "action_required": len(all_alerts) > 0
        }


def analyze_plot_with_intent(image_path: str, intent: str) -> str:
    """
    Funcao de conveniencia para analisar um plot com intent customizado.
    
    Args:
        image_path: Caminho da imagem
        intent: Descricao do que procurar
    
    Returns:
        Texto da analise
    """
    critic = VisionCritic()
    result = critic.analyze_plot(image_path, custom_intent=intent)
    return result.get("analysis", "Analise nao disponivel")


# CLI para testes
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Vision Critic")
    parser.add_argument("--image", required=True, help="Path to image")
    parser.add_argument("--intent", default="", help="Custom intent")
    parser.add_argument("--code", default="", help="Code context file")
    
    args = parser.parse_args()
    
    critic = VisionCritic()
    
    code_context = ""
    if args.code and Path(args.code).exists():
        code_context = Path(args.code).read_text(encoding='utf-8')
    
    result = critic.analyze_plot(args.image, args.intent, code_context)
    
    analysis = result.get('analysis') or ""
    if len(analysis) > 1500:
        analysis = analysis[:1500].rstrip() + "\n... (truncado)"
    print(f"\nPlot Type: {result.get('plot_type')}")
    print(f"Intent: {result.get('intent')}")
    print(f"\n{analysis}")
