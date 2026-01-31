#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AGENT CONTROLLER: O Cientista de Dados Autonomo (Versao Unificada)
---------------------------------------------------
Este script implementa o loop OODA (Observe, Orient, Decide, Act).
Combina a robustez de setup do agent_controller com a inteligencia de metricas do meta_controller.
"""

import os
import sys
import json
import yaml
import time
import subprocess
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Configurar encoding UTF-8 para stdout no Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Tenta importar load_env_file, fallback se falhar
try:
    from markdown_logger import load_env_file
except ImportError:
    def load_env_file(p): return {}

class AutonomousDataScientist:
    def __init__(self, goals_file="GOALS.md", config_file="config.yaml", history_file="CHANGELOG.md"):
        self.base_path = Path(__file__).parent
        self.goals_path = self.base_path / goals_file
        self.config_path = self.base_path / config_file
        self.history_path = self.base_path / history_file
        self.reports_dir = self.base_path / "reports"
        
        # Histórico de métricas para detectar estagnação
        self.profit_history = []  # Fallback para lucro absoluto
        self.efficiency_history = []  # ✅ Prioridade: eficiência financeira percentual
        
        # Carregar API Key
        self._setup_llm()
        
        # Estado inicial
        self.iteration = 0
        # ✅ Ajustar max_iterations baseado no modo (DEV = 3, PROD = 10)
        self.max_iterations = 10  # Default, será ajustado abaixo
        self._detect_mode_and_set_iterations()

    def _setup_llm(self):
        """Configura conexão com LLM (OpenAI/Gemini/Claude)."""
        env_vars = load_env_file(self.base_path / ".env")
        self.api_key = os.getenv("OPENAI_API_KEY") or env_vars.get("OPENAI_API_KEY")
        self.provider = "openai" # Default para raciocínio complexo
        self.model_name = None  # Será definido abaixo
        
        if not self.api_key:
            # Tentar Gemini se OpenAI não estiver disponível
            self.api_key = os.getenv("GEMINI_API_KEY") or env_vars.get("GEMINI_KEY")
            self.provider = "gemini" if self.api_key else None
            
        if not self.api_key:
            raise ValueError("[ERRO] Nenhuma API Key encontrada no .env ou ambiente. O Agente nao pode pensar.")
        
        # ✅ Ler MODEL_NAME do .env se disponível
        if self.provider == "gemini":
            self.model_name = env_vars.get("MODEL_NAME") or os.getenv("MODEL_NAME") or "gemini-1.5-pro"
            print(f"[INFO] Cerebro ativado: {self.provider.upper()} (Modelo: {self.model_name})")
        else:
            print(f"[INFO] Cerebro ativado: {self.provider.upper()}")
    
    def _detect_mode_and_set_iterations(self):
        """Detecta o modo (DEV/PROD) do config.yaml e ajusta max_iterations."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    mode = config.get('pipeline', {}).get('mode', 'DEV')
                    if mode == 'DEV':
                        self.max_iterations = 2
                        print(f"[INFO] Modo DEV detectado: max_iterations ajustado para {self.max_iterations}")
                    else:
                        self.max_iterations = 10
                        print(f"[INFO] Modo PROD detectado: max_iterations mantido em {self.max_iterations}")
        except Exception as e:
            print(f"[WARN] Erro ao detectar modo do config.yaml: {e}. Usando default (10 iterações)")
            self.max_iterations = 10

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Chamada agnóstica para a LLM de raciocínio."""
        try:
            if self.provider == "openai":
                from openai import OpenAI
                client = OpenAI(api_key=self.api_key)
                response = client.chat.completions.create(
                    model="gpt-4o", # Modelo forte para raciocínio
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
                return response.choices[0].message.content
                
            elif self.provider == "gemini":
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                # ✅ Usar model_name do .env ou fallback
                model_to_use = self.model_name or "gemini-1.5-pro"
                print(f"[DEBUG] Usando modelo Gemini: {model_to_use}")
                model = genai.GenerativeModel(model_to_use)
                response = model.generate_content(
                    f"{system_prompt}\n\nUSER INPUT:\n{user_prompt}",
                    generation_config={"response_mime_type": "application/json"}
                )
                return response.text
                
        except Exception as e:
            print(f"[ERRO] Erro na chamada LLM: {e}")
            # ✅ Retornar um JSON válido com status FAIL para que o agente possa processar o erro
            error_response = {
                "status": "FAIL",
                "analysis": f"Erro na chamada LLM: {str(e)}",
                "changes": {},
                "reasoning": "Erro ao comunicar com LLM. Verifique API key e modelo configurado."
            }
            return json.dumps(error_response)

    def run_pipeline(self) -> bool:
        """Executa o script do pipeline e retorna True se sucesso."""
        print(f"\n[EXEC] Executando Pipeline (Iteracao {self.iteration})...")
        start = time.time()
        
        pipeline_script = "credit_scoring_pipeline.py"
        if not (self.base_path / pipeline_script).exists():
             print(f"[ERRO] Erro: {pipeline_script} nao encontrado!")
             return False

        # Executa o script python como subprocesso
        # ✅ Configurar encoding UTF-8 para subprocess no Windows
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        try:
            result = subprocess.run(
                ["python", pipeline_script],
                capture_output=True,
                text=True,
                cwd=str(self.base_path),
                timeout=3600,  # 1 hora timeout
                env=env,
                encoding='utf-8',
                errors='replace'  # Substitui caracteres inválidos ao invés de falhar
            )
            
            duration = time.time() - start
            
            if result.returncode == 0:
                print(f"[OK] Pipeline concluido em {duration:.1f}s.")
                return True
            else:
                print(f"[ERRO] Falha no Pipeline:\n{result.stderr[-2000:]}") # Mostra os ultimos 2000 chars do erro
                return False
        except Exception as e:
            print(f"[ERRO] Erro critico ao chamar subprocesso: {e}")
            return False

    def get_latest_report(self) -> Optional[str]:
        """Lê o conteúdo do relatório markdown mais recente."""
        if not self.reports_dir.exists():
            return None
            
        reports = sorted(self.reports_dir.glob("*.md"), key=os.path.getmtime, reverse=True)
        if not reports:
            return None
        
        latest_report = reports[0]
        print(f"[INFO] Lendo relatorio: {latest_report.name}")
        return latest_report.read_text(encoding='utf-8')

    def extract_metrics_from_report(self, report_content: str) -> dict:
        """
        ✅ 2. EXTRAÇÃO DE MÉTRICAS ROBUSTA: Regex melhorado com múltiplos padrões.
        """
        metrics = {}
        try:
            # AUC Validação - Múltiplos padrões
            patterns_auc = [
                r'AUC Validação[:\s]+(\d+\.\d+)',
                r'AUC.*?Validação[:\s]+(\d+\.\d+)',
                r'Validação.*?AUC[:\s]+(\d+\.\d+)',
                r'- \*\*AUC Validação\*\*:.*?`(\d+\.\d+)`',
            ]
            for pattern in patterns_auc:
                auc_match = re.search(pattern, report_content, re.IGNORECASE)
                if auc_match:
                    metrics['auc_val'] = float(auc_match.group(1))
                    break
            
            # AUC Treino
            patterns_auc_train = [
                r'AUC Treino[:\s]+(\d+\.\d+)',
                r'AUC.*?Treino[:\s]+(\d+\.\d+)',
                r'- \*\*AUC Treino\*\*:.*?`(\d+\.\d+)`',
            ]
            for pattern in patterns_auc_train:
                auc_train_match = re.search(pattern, report_content, re.IGNORECASE)
                if auc_train_match:
                    metrics['auc_train'] = float(auc_train_match.group(1))
                    break
            
            # Gap (Overfitting) - Múltiplos padrões
            patterns_gap = [
                r'Gap \(Overfitting\)[:\s]+(\d+\.\d+)%',
                r'Gap.*?Overfitting[:\s]+(\d+\.\d+)%',
                r'Overfitting.*?Gap[:\s]+(\d+\.\d+)%',
                r'- \*\*Gap \(Overfitting\)\*\*:.*?`(\d+\.\d+)%`',
                r'Gap.*?(\d+\.\d+)\s*%',  # Padrão mais genérico
            ]
            for pattern in patterns_gap:
                gap_match = re.search(pattern, report_content, re.IGNORECASE)
                if gap_match:
                    metrics['gap_percent'] = float(gap_match.group(1))
                    break
            
            # PSI - Múltiplos padrões
            patterns_psi = [
                r'PSI.*?(\d+\.\d+)',
                r'Population Stability Index.*?(\d+\.\d+)',
                r'- \*\*PSI\*\*:.*?`(\d+\.\d+)`',
            ]
            for pattern in patterns_psi:
                psi_match = re.search(pattern, report_content, re.IGNORECASE)
                if psi_match:
                    metrics['psi'] = float(psi_match.group(1))
                    break

            # Lucro Real - Múltiplos padrões e formatos
            patterns_profit = [
                r'Lucro Real.*?R\$\s*([\d,\.]+)',
                r'Lucro.*?Esperado.*?Máximo.*?R\$\s*([\d,\.]+)',
                r'Lucro.*?Total.*?R\$\s*([\d,\.]+)',
                r'Lucro.*?Líquido.*?R\$\s*([\d,\.]+)',
                r'Lucro.*?R\$\s*([\d,\.]+)',
                r'Profit.*?R\$\s*([\d,\.]+)',
                r'- \*\*Lucro.*?\*\*:.*?R\$\s*([\d,\.]+)',
            ]
            for pattern in patterns_profit:
                profit_match = re.search(pattern, report_content, re.IGNORECASE)
                if profit_match:
                    profit_str = profit_match.group(1).replace(',', '').replace('.', '')
                    # Detectar se é milhões (tem 6+ dígitos) ou milhares
                    if len(profit_str) >= 6:
                        # Provavelmente está em milhões, converter
                        metrics['profit'] = float(profit_str)
                    else:
                        metrics['profit'] = float(profit_str)
                    break
            
            # ✅ NOVO: Eficiência Financeira (% do Potencial)
            # Padrões flexíveis para capturar diferentes formatos
            patterns_efficiency = [
                r'Eficiência Financeira.*?\(% do Potencial\)[:\s]+(\d+\.\d+)%',  # Formato completo
                r'Eficiência Financeira.*?% do Potencial[:\s]+(\d+\.\d+)%',  # Sem parênteses
                r'Eficiência Financeira[:\s]+(\d+\.\d+)%',  # Formato simples
                r'Eficiência Financeira.*?(\d+\.\d+)\s*%',  # Mais genérico
            ]
            for pattern in patterns_efficiency:
                efficiency_match = re.search(pattern, report_content, re.IGNORECASE | re.MULTILINE)
                if efficiency_match:
                    metrics['financial_efficiency'] = float(efficiency_match.group(1))
                    break
            
            # ✅ NOVO: Lucro Potencial Máximo
            potential_match = re.search(r'Lucro Potencial Máximo.*?R\$\s*([\d,\.]+)', report_content, re.IGNORECASE)
            if potential_match:
                potential_str = potential_match.group(1).replace(',', '').replace('.', '')
                metrics['profit_potential_max'] = float(potential_str)
            
            # Taxa de Aprovação
            approval_match = re.search(r'Taxa.*?Aprovação.*?(\d+\.\d+)%', report_content, re.IGNORECASE)
            if approval_match:
                metrics['approval_rate'] = float(approval_match.group(1))
            
            # Brier Score (calibração)
            brier_match = re.search(r'Brier Score.*?\(Depois.*?\)[:\s]+(\d+\.\d+)', report_content, re.IGNORECASE)
            if brier_match:
                metrics['brier_score'] = float(brier_match.group(1))
            
            # Threshold Ótimo
            threshold_match = re.search(r'Threshold.*?Ótimo[:\s]+(\d+\.\d+)', report_content, re.IGNORECASE)
            if threshold_match:
                metrics['optimal_threshold'] = float(threshold_match.group(1))

        except Exception as e:
            print(f"[WARN] Erro ao extrair metricas via regex: {e}")
            import traceback
            print(f"[DEBUG] Traceback: {traceback.format_exc()}")
        
        return metrics

    def check_stagnation(self, current_efficiency: float = None, current_profit: float = None, patience: int = 3) -> bool:
        """
        Verifica se a performance parou de melhorar.
        Prioriza eficiência financeira (percentual) se disponível, senão usa lucro absoluto.
        """
        # ✅ Priorizar eficiência financeira (agnóstica a amostra)
        if current_efficiency is not None:
            if not hasattr(self, 'efficiency_history'):
                self.efficiency_history = []
            self.efficiency_history.append(current_efficiency)
            
            if len(self.efficiency_history) < patience + 1:
                return False
            
            recent = self.efficiency_history[-(patience+1):]
            # Se a eficiência atual não melhorou significativamente
            if recent[-1] <= max(recent[:-1]) * 1.001:  # Margem de 0.1%
                return True
            return False
        
        # Fallback para lucro absoluto (se eficiência não disponível)
        if current_profit:
            if not hasattr(self, 'profit_history'):
                self.profit_history = []
            self.profit_history.append(current_profit)
            if len(self.profit_history) < patience + 1:
                return False
            
            recent = self.profit_history[-(patience+1):]
            if recent[-1] <= max(recent[:-1]) * 1.001:  # Margem de 0.1%
                return True
        
        return False

    def get_pipeline_code(self) -> str:
        """Lê o código fonte do pipeline para dar contexto à LLM."""
        pipeline_path = self.base_path / "credit_scoring_pipeline.py"
        if pipeline_path.exists():
            code = pipeline_path.read_text(encoding='utf-8')
            # Retornar apenas as partes mais relevantes (funções principais e estrutura)
            # Limitar a ~15000 caracteres para não estourar contexto
            return code[:15000] + "\n... (código truncado para economia de contexto) ..."
        return "Código do pipeline não encontrado."
    
    def get_changelog_summary(self) -> str:
        """Lê o changelog para dar contexto histórico ao agente."""
        if self.history_path.exists():
            changelog = self.history_path.read_text(encoding='utf-8')
            # Retornar apenas as últimas 3 rodadas
            lines = changelog.split('\n')
            # Encontrar os headers de rodadas
            rodada_lines = [i for i, line in enumerate(lines) if line.startswith('## Rodada')]
            if len(rodada_lines) >= 3:
                # Pegar últimas 3 rodadas
                start_idx = rodada_lines[-3] if len(rodada_lines) >= 3 else 0
                return '\n'.join(lines[start_idx:])
            return changelog[:3000]  # Limitar tamanho
        return "Nenhum histórico disponível."
    
    def analyze_and_decide(self, report_content: str, current_config: Dict, metrics: Dict) -> Dict:
        """O Núcleo Cognitivo: Lê o relatório e decide mudanças com CONTEXTO COMPLETO."""
        print("[ANALISE] Agente analisando resultados com Contexto Total...")
        
        goals = self.goals_path.read_text(encoding='utf-8')
        config_str = yaml.dump(current_config, default_flow_style=False)
        
        # ✅ NOVO: Contexto Completo
        pipeline_code = self.get_pipeline_code()  # O Agente agora lê o código!
        changelog_summary = self.get_changelog_summary()  # Histórico de tentativas
        
        # Prepara resumo de métricas para ajudar a LLM
        metrics_str = json.dumps(metrics, indent=2) if metrics else "Não extraídas automaticamente."

        system_prompt = (
            "Você é um AGENTE CIENTISTA DE DADOS AUTÔNOMO SÊNIOR (Principal Data Scientist).\n"
            "Você tem controle total sobre os PARÂMETROS (`config.yaml`) de um pipeline de Risco de Crédito.\n"
            "Seu objetivo é satisfazer `GOALS.md`.\n\n"
            "--- SEU PODER (Via config.yaml) ---\n"
            "Você NÃO pode reescrever o código Python diretamente, mas pode controlar o comportamento dele via config:\n"
            "1. **Overfitting (Gap > 8%)?** Ajuste `xgboost_params.max_depth` (reduzir), `xgboost_params.gamma` (aumentar), "
            "`xgboost_params.min_child_weight` (aumentar), `xgboost_params.subsample` (reduzir).\n"
            "2. **Drift/Leakage detectado?** Se uma feature específica mostrar KS > 0.5 no relatório de Drift, "
            "adicione o nome da coluna em `feature_selection.force_drop_features` (ex: ['feature_757']). "
            "O pipeline removerá essa feature automaticamente.\n"
            "3. **Underfitting (AUC baixo)?** Aumente `xgboost_params.learning_rate` ou `xgboost_params.n_estimators_dev/prod`.\n"
            "4. **Eficiência Financeira < 75%?** A eficiência é calculada como % do lucro potencial máximo. "
            "Se estiver baixa, pode indicar que o modelo não está capturando bem os bons pagadores. "
            "Ajuste hiperparâmetros para melhorar AUC ou verifique se há features ruins sendo usadas.\n\n"
            "--- MÉTRICAS IMPORTANTES (Prioridade) ---\n"
            "1. `financial_efficiency`: % do lucro potencial máximo capturado (meta: > 75%) - AGNÓSTICA A AMOSTRA\n"
            "2. `gap_percent`: Gap entre AUC treino e validação (meta: < 8%) - CRÍTICO\n"
            "3. `auc_val`: AUC de validação (meta: > 0.81)\n"
            "4. `psi`: Population Stability Index (meta: < 0.2)\n"
            "5. Com `psi` < 0.2, priorize maior lucro no teste em caso de empate\n\n"
            "--- CONTEXTO DISPONÍVEL ---\n"
            "Você recebe: (1) Código fonte do pipeline (estrutura e funções principais), "
            "(2) Histórico de tentativas anteriores (CHANGELOG), (3) Relatório markdown completo da última execução, "
            "(4) Métricas extraídas automaticamente, (5) Configuração atual.\n"
            "Use esse contexto para entender como o código funciona e tomar decisões informadas.\n\n"
            "--- FORMATO DE RESPOSTA JSON (OBRIGATÓRIO) ---\n"
            "{\n"
            "  \"analysis\": \"Diagnóstico técnico profundo baseado no código, relatório e histórico.\",\n"
            "  \"status\": \"CONTINUE\" (precisa melhorar) | \"SUCCESS\" (metas atingidas) | \"FAIL\" (estagnou/erro),\n"
            "  \"changes\": {\"feature_selection\": {\"force_drop_features\": [\"feature_757\"]}, \"xgboost_params\": {\"max_depth\": 4}},\n"
            "  \"reasoning\": \"Por que mudou X para Y baseado no gráfico Z e no código do pipeline.\"\n"
            "}"
        )
        
        user_prompt = (
            f"--- OBJETIVOS (GOALS.md) ---\n{goals}\n\n"
            f"--- CONFIG ATUAL (config.yaml) ---\n{config_str}\n\n"
            f"--- MÉTRICAS EXTRAÍDAS ---\n{metrics_str}\n\n"
            f"--- ESTRUTURA DO CÓDIGO (Para Contexto) ---\n{pipeline_code}\n\n"
            f"--- HISTÓRICO (CHANGELOG - Últimas Rodadas) ---\n{changelog_summary}\n\n"
            f"--- RELATÓRIO RECENTE (Markdown Completo) ---\n{report_content[:40000]}"
        )
        
        response_json = self._call_llm(system_prompt, user_prompt)
        
        try:
            cleaned_json = re.sub(r"```json|```", "", response_json).strip()
            decision = json.loads(cleaned_json)
            
            # ✅ Validar que a decisão tem os campos necessários
            if not decision.get('status'):
                decision['status'] = 'CONTINUE'
            if not decision.get('analysis'):
                decision['analysis'] = 'Análise não disponível'
            if not decision.get('changes'):
                decision['changes'] = {}
            if not decision.get('reasoning'):
                decision['reasoning'] = 'Sem justificativa fornecida'
                
            return decision
        except json.JSONDecodeError:
            print(f"[ERRO] Erro no JSON da LLM. Raw: {response_json[:200]}...")
            # ✅ Retornar decisão válida mesmo com erro de JSON
            return {
                "status": "FAIL", 
                "analysis": f"Erro ao decodificar JSON da LLM. Resposta: {response_json[:100]}...", 
                "changes": {},
                "reasoning": "A LLM retornou resposta inválida. Verifique configuração do modelo."
            }

    def apply_changes(self, current_config: Dict, changes: Dict) -> Dict:
        """Aplica mudanças (Deep Merge) e salva."""
        def update_recursive(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = update_recursive(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
            
        new_config = update_recursive(current_config.copy(), changes)
        new_config['last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(new_config, f, sort_keys=False, default_flow_style=False)
            
        return new_config

    def update_history(self, decision: Dict, metrics: Dict):
        """Registra no CHANGELOG.md."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        metrics_line = f"- **Métricas:** {json.dumps(metrics)}\n" if metrics else ""
        
        entry = f"""## Rodada {self.iteration} - {timestamp}
- **Status:** {decision.get('status')}
- **Analise:** {decision.get('analysis')}
{metrics_line}- **Mudancas:** ```json
{json.dumps(decision.get('changes', {}), indent=2) if decision.get('changes') else '{}'}
```
- **Racional:** {decision.get('reasoning')}

"""
        if self.history_path.exists():
            current = self.history_path.read_text(encoding='utf-8')
            self.history_path.write_text(entry + "\n" + current, encoding='utf-8')
        else:
            self.history_path.write_text("# Histórico de Otimização\n\n" + entry, encoding='utf-8')

    def start(self):
        """Loop principal."""
        print(f"[INICIO] INICIANDO AGENTE CIENTISTA DE DADOS")
        print(f"[META] Metas: {self.goals_path}")
        
        for i in range(1, self.max_iterations + 1):
            self.iteration = i
            print(f"\n{'='*60}\n[RODADA] RODADA {i}/{self.max_iterations}\n{'='*60}")
            
            # 1. Carregar Config
            with open(self.config_path, 'r', encoding='utf-8') as f:
                current_config = yaml.safe_load(f)
            
            # 2. Executar Pipeline
            if not self.run_pipeline():
                print("[PARADA] Parando por falha no pipeline.")
                break
            
            # 3. Ler Relatório e Extrair Métricas
            report = self.get_latest_report()
            if not report: break
            
            metrics = self.extract_metrics_from_report(report)
            if metrics: print(f"[METRICAS] Metricas detectadas: {metrics}")
            
            # 4. Decidir
            decision = self.analyze_and_decide(report, current_config, metrics)
            print(f"\n[DECISAO] DECISAO: {decision.get('status')}")
            print(f"[ANALISE] Analise: {decision.get('analysis')}")
            
            # 5. Atualizar Histórico
            self.update_history(decision, metrics)
            
            # 6. Agir
            status = decision.get('status')
            if status == 'SUCCESS':
                print("\n[SUCESSO] METAS ATINGIDAS! Otimizacao concluida.")
                break
            elif status == 'FAIL':
                # ✅ Melhorar tratamento de erros: tentar continuar se for erro de LLM, parar se for estagnação real
                analysis = decision.get('analysis', '')
                if 'Erro na chamada LLM' in analysis or 'Erro ao decodificar JSON' in analysis:
                    print(f"\n[ERRO LLM] Erro na comunicação com LLM: {analysis}")
                    print("[TENTATIVA] Tentando continuar com configuração atual ou ajuste manual...")
                    # Se houver mudanças sugeridas antes do erro, aplicá-las
                    changes = decision.get('changes', {})
                    if changes:
                        print(f"[CONFIG] Aplicando alteracoes sugeridas antes do erro...")
                        self.apply_changes(current_config, changes)
                        continue  # Continua para próxima iteração
                    else:
                        print("[PARADA] Nenhuma mudança disponível após erro LLM. Parando.")
                        break
                else:
                    print(f"\n[PARADA] Agente desistiu: {analysis}")
                    break
            
            # Checar Estagnação (priorizar eficiência financeira se disponível)
            efficiency = metrics.get('financial_efficiency')
            profit = metrics.get('profit')
            if efficiency is not None:
                if self.check_stagnation(current_efficiency=efficiency):
                    print(f"\n[PARADA] Eficiência Financeira estagnou em {efficiency:.2f}%. Parando otimização.")
                    break
            elif profit:
                if self.check_stagnation(current_profit=profit):
                    print(f"\n[PARADA] Lucro estagnou em R$ {profit:,.2f}. Parando otimização.")
                    break

            # Aplicar mudanças
            changes = decision.get('changes', {})
            if changes:
                print(f"[CONFIG] Aplicando alteracoes...")
                self.apply_changes(current_config, changes)
            else:
                # ✅ Se não há mudanças mas status é CONTINUE, continuar de qualquer forma (pode ser que já esteja otimizado)
                if status == 'CONTINUE':
                    print("⚠️ Nenhuma mudança sugerida mas status é CONTINUE. Continuando...")
                    continue
                else:
                    print("⚠️ Nenhuma mudança sugerida. Parando.")
                    break

if __name__ == "__main__":
    # Setup inicial de arquivos se não existirem
    base_dir = Path(__file__).parent
    
    if not (base_dir / "GOALS.md").exists():
        (base_dir / "GOALS.md").write_text("""# Objetivos
1. Gap (Treino - Validação) < 0.08 (Prioridade Alta)
2. AUC Validação > 0.81
3. Maximizar Lucro no Teste (se PSI < 0.2)
4. PSI < 0.2
""", encoding='utf-8')
        print("✅ GOALS.md criado.")

    if not (base_dir / "config.yaml").exists():
        default_config = {
            "pipeline": {"mode": "DEV"},
            "xgboost_params": {
                "objective": "binary:logistic", "eval_metric": "auc",
                "max_depth": 6, "learning_rate": 0.05,
                "n_estimators_dev": 100, "n_estimators_prod": 500,
                "gamma": 0, "min_child_weight": 1
            },
            "business_params": {"ticket_medio": 10000, "ganho_tp": 1500, "perda_fp": -10000}
        }
        with open(base_dir / "config.yaml", 'w') as f:
            yaml.dump(default_config, f)
        print("✅ config.yaml criado.")

    agent = AutonomousDataScientist()
    agent.start()