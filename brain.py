"""
brain.py - Cerebro Cognitivo do Agente Cientista de Dados (Ralph DS v2.0)

Agente AGNÓSTICO que resolve qualquer problema de Data Science:
- Detecta automaticamente o tipo de problema (classificação, regressão, etc.)
- Começa sempre com EDA obrigatória
- Planeja dinamicamente baseado nos insights descobertos
- Documenta continuamente via STATE.md (memória resumida)

O cerebro mantém memoria do projeto e decide qual acao tomar:
- Planejar proximos passos
- Escrever/editar codigo
- Analisar outputs
- Decidir se aceita, corrige ou reescreve

Usa OODA Loop: Observe, Orient, Decide, Act
"""

import os
import json
import re
import yaml
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
from dataclasses import dataclass, field

# Imports locais
from executor import CodeExecutor, ExecutionResult, format_execution_report
from vision_critic import VisionCritic, analyze_plot_with_intent


def _details_for_display(details: Dict[str, Any]) -> Dict[str, Any]:
    """Monta uma versao dos detalhes para exibicao no terminal, sem colar codigo integral."""
    if not details:
        return details
    display = {}
    for k, v in details.items():
        if k in ("changes", "code_changes") and isinstance(v, str):
            n_chars = len(v)
            n_lines = v.count("\n") + 1
            display[k] = f"<edicao no script: {n_chars} caracteres, {n_lines} linhas (codigo omitido no log)>"
        elif isinstance(v, str) and len(v) > 400:
            display[k] = v[:400].rstrip() + "\n... (truncado)"
        else:
            display[k] = v
    return display


class Action(Enum):
    """Acoes que o agente pode tomar."""
    PLAN = "plan"                    # Planejar proximos passos
    WRITE_CODE = "write_code"        # Escrever novo script
    EDIT_CODE = "edit_code"          # Editar script existente
    RUN_STEP = "run_step"            # Executar um script
    ANALYZE = "analyze"              # Analisar output/plot
    ROLLBACK = "rollback"            # Reverter para estado anterior
    UPDATE_CONFIG = "update_config"  # Atualizar config.yaml
    UPDATE_TASK_LIST = "update_task_list"  # Replanejar: add/remove/edit etapas (TASK_LIST + scripts)
    RUN_FROM_STEP = "run_from_step"  # Alterar passado: re-executar a partir de um step
    STOP = "stop"                    # Parar execucao


@dataclass
class AgentState:
    """Estado atual do agente."""
    current_phase: str = "planning"
    current_step: Optional[str] = None
    last_result: Optional[ExecutionResult] = None
    goals_achieved: bool = False
    iteration: int = 0
    errors_in_row: int = 0
    stagnation_count: int = 0
    decisions_log: List[Dict] = field(default_factory=list)


class AgentBrain:
    """
    Cerebro cognitivo do agente autonomo.
    
    Implementa o loop OODA:
    - Observe: Ler estado atual, outputs, graficos
    - Orient: Comparar com objetivos
    - Decide: Escolher proxima acao
    - Act: Executar acao
    """
    
    MAX_ITERATIONS = 50
    MAX_ERRORS_IN_ROW = 3
    MAX_STAGNATION = 5

    # Propósito de cada step (contexto para análise Vision: o que o step faz e para que serve a imagem)
    # Genérico: adapta-se ao tipo de problema detectado
    STEP_PURPOSES = {
        # EDA Obrigatória (sempre executa)
        "01_load_data": "Carregar dados brutos e gerar metadata inicial (tamanhos, tipos, target).",
        "02_eda_overview": "Visão geral dos dados: shape, tipos, memória, primeiras linhas.",
        "02_eda_nulls": "Analisar valores faltantes por feature; decidir remoção ou imputação.",
        "03_eda_target": "Analisar distribuição do target e detectar tipo de problema.",
        "03_eda_nulls": "Analisar valores faltantes por feature; decidir remoção ou imputação.",
        "04_eda_distributions": "Plotar distribuições das features numéricas (skewness, outliers).",
        "04_eda_target": "Analisar distribuição do target e detectar tipo de problema.",
        "05_eda_correlations": "Analisar correlações e multicolinearidade (heatmap, redundâncias).",
        "05_eda_distributions": "Plotar distribuições das features numéricas (skewness, outliers).",
        "06_eda_drift": "Comparar treino vs teste para detectar drift (adversarial validation).",
        "06_eda_correlations": "Analisar correlações e multicolinearidade (heatmap, redundâncias).",
        "07_eda_drift": "Comparar treino vs teste para detectar drift (adversarial validation).",
        # Feature Engineering (dinâmico)
        "07_feature_cleanup": "Remover features críticas (>50% nulos, redundantes).",
        "08_feature_cleanup": "Remover features críticas (alta nulidade, redundantes).",
        "08_feature_transform": "Aplicar transformações (log, binning) em features assimétricas.",
        "09_feature_transform": "Aplicar transformações (log, binning) em features assimétricas.",
        "09_feature_select": "Seleção final de features (importância, correlação).",
        "10_feature_select": "Seleção final de features (importância, correlação).",
        # Modelagem (adapta ao tipo de problema)
        "10_train_baseline": "Treinar modelo baseline; curvas de aprendizado.",
        "11_train_baseline": "Treinar modelo baseline; curvas de aprendizado.",
        "11_evaluate_model": "Avaliar métricas e gap treino-teste.",
        "12_evaluate_model": "Avaliar métricas e gap treino-teste.",
        "12_tune_hyperparams": "Otimização de hiperparâmetros (Optuna/Bayesian).",
        "13_tune_hyperparams": "Otimização de hiperparâmetros (Optuna/Bayesian).",
        "13_cross_validate": "Validação cruzada para estimar variância.",
        "14_cross_validate": "Validação cruzada para estimar variância.",
        # Business Layer (opcional, depende do problema)
        "14_find_threshold": "Encontrar threshold ótimo (classificação).",
        "15_find_threshold": "Encontrar threshold ótimo (classificação).",
        "15_business_metrics": "Calcular métricas de negócio (lucro, eficiência).",
        "16_business_metrics": "Calcular métricas de negócio (lucro, eficiência).",
        "16_stability_analysis": "Calcular PSI e verificar estabilidade (drift).",
        "17_stability_analysis": "Calcular PSI e verificar estabilidade (drift).",
        "17_final_report": "Gerar relatório final consolidado.",
        "18_final_report": "Gerar relatório final consolidado.",
        "18_export": "Exportar modelo, submissão e artefatos.",
        "19_export": "Exportar modelo, submissão e artefatos.",
    }
    
    # Tipos de problema suportados
    PROBLEM_TYPES = {
        "binary_classification": {
            "metrics": ["auc", "f1", "precision", "recall"],
            "default_model": "xgboost",
            "eda_steps": ["01_load_data", "02_eda_overview", "03_eda_nulls", "04_eda_target", 
                         "05_eda_distributions", "06_eda_correlations", "07_eda_drift"],
            "model_steps": ["08_feature_cleanup", "09_feature_select", "10_train_baseline",
                          "11_evaluate_model", "12_tune_hyperparams", "13_find_threshold"],
        },
        "multiclass_classification": {
            "metrics": ["f1_macro", "accuracy", "confusion_matrix"],
            "default_model": "xgboost",
            "eda_steps": ["01_load_data", "02_eda_overview", "03_eda_nulls", "04_eda_target",
                         "05_eda_distributions", "06_eda_correlations"],
            "model_steps": ["07_feature_cleanup", "08_feature_select", "09_train_baseline",
                          "10_evaluate_model", "11_tune_hyperparams"],
        },
        "regression": {
            "metrics": ["rmse", "mae", "r2"],
            "default_model": "xgboost",
            "eda_steps": ["01_load_data", "02_eda_overview", "03_eda_nulls", "04_eda_target",
                         "05_eda_distributions", "06_eda_correlations"],
            "model_steps": ["07_feature_cleanup", "08_feature_select", "09_train_baseline",
                          "10_evaluate_model", "11_residual_analysis"],
        },
        "unknown": {
            "metrics": [],
            "default_model": None,
            "eda_steps": ["01_load_data", "02_eda_overview", "03_eda_nulls"],
            "model_steps": [],
        }
    }
    
    # EDA obrigatória: estes steps sempre executam primeiro
    EDA_REQUIRED_STEPS = ["01_load_data", "02_eda_overview", "03_eda_nulls", "04_eda_target"]

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent
        
        # Componentes
        self.executor = CodeExecutor(self.base_dir)
        self.vision_critic = VisionCritic()
        
        # Paths importantes
        self.goals_path = self.base_dir / "GOALS.md"
        self.config_path = self.base_dir / "config.yaml"
        self.changelog_path = self.base_dir / "CHANGELOG.md"
        self.task_list_path = self.base_dir / "TASK_LIST.md"
        self.state_md_path = self.base_dir / "STATE.md"  # Memória resumida
        self.src_dir = self.base_dir / "src"
        self.notebooks_dir = self.base_dir / "notebooks"
        self.backups_dir = self.base_dir / "backups"
        self.context_dir = self.base_dir / "context"  # Contexto do projeto (inclui data/)
        
        # Estado do agente
        self.state = AgentState()
        
        # Tipo de problema detectado (atualizado após EDA)
        self.problem_type: str = "unknown"

        # Pasta da execução atual (runs/YYYYMMDD_HHMMSS); preenchida ao iniciar run()
        self.current_run_dir: Optional[Path] = None

        # Carregar configuracoes
        self.config = self._load_config()
        self.goals = self._load_goals()
        self.tasks = self._load_tasks()
        
        # Carregar STATE.md se existir
        self._load_state_md()
        
        # LLM client (configurar via .env)
        self._setup_llm()
    
    def _setup_llm(self):
        """Configura cliente LLM."""
        from dotenv import load_dotenv
        load_dotenv()
        
        self.gemini_key = os.getenv("GEMINI_KEY")
        self.model_name = os.getenv("MODEL_NAME", "gemini-1.5-flash")  # Usar MODEL_NAME do .env
        self.openai_key = os.getenv("OPENAI_API_KEY")
        
        if self.gemini_key:
            self.llm_provider = "gemini"
            print(f"[BRAIN] LLM: Gemini ({self.model_name})")
        elif self.openai_key:
            self.llm_provider = "openai"
            print("[BRAIN] LLM: OpenAI")
        else:
            print("[BRAIN] AVISO: Nenhuma API key encontrada. Modo simulado.")
            self.llm_provider = "mock"
    
    def _load_state_md(self) -> None:
        """Carrega STATE.md e extrai informações importantes."""
        if not self.state_md_path.exists():
            return
        try:
            content = self.state_md_path.read_text(encoding='utf-8')
            # Extrair tipo de problema se já detectado
            if "**Detectado:**" in content:
                match = re.search(r'\*\*Detectado:\*\*\s*(\w+)', content)
                if match:
                    detected = match.group(1).lower()
                    if "binár" in detected or "binary" in detected:
                        self.problem_type = "binary_classification"
                    elif "multi" in detected:
                        self.problem_type = "multiclass_classification"
                    elif "regress" in detected:
                        self.problem_type = "regression"
        except Exception:
            pass
    
    def _update_state_md(self, action: Optional["Action"] = None, result: Optional["ExecutionResult"] = None) -> None:
        """Atualiza STATE.md com estado atual do projeto (memória resumida)."""
        meta = self.executor.metadata
        decisions = meta.get("decisions", {})
        metrics = meta.get("metrics", {})
        data_info = meta.get("data", {})
        
        # Detectar tipo de problema baseado nos dados
        problem_type_display = {
            "binary_classification": "Classificação Binária",
            "multiclass_classification": "Classificação Multiclasse",
            "regression": "Regressão",
            "unknown": "Não detectado"
        }.get(self.problem_type, "Não detectado")
        
        # Contar features
        n_features = 0
        n_rows_train = 0
        n_rows_test = 0
        for name, info in data_info.items():
            if "train" in name.lower():
                n_features = info.get("columns", 0)
                n_rows_train = info.get("rows", 0)
            elif "test" in name.lower():
                n_rows_test = info.get("rows", 0)
        
        # Métricas formatadas
        metrics_table = "| Métrica | Valor | Status |\n|---------|-------|--------|\n"
        for key, val in metrics.items():
            if isinstance(val, float):
                status = "✅" if val > 0.8 else "⚠️" if val > 0.7 else "❌"
                metrics_table += f"| {key} | {val:.4f} | {status} |\n"
            else:
                metrics_table += f"| {key} | {val} | - |\n"
        if not metrics:
            metrics_table += "| (nenhuma ainda) | - | - |\n"
        
        # Decisões tomadas
        decisions_text = ""
        if decisions.get("features_to_drop"):
            n_drop = len(decisions["features_to_drop"])
            decisions_text += f"- **Features removidas:** {n_drop}\n"
        if decisions.get("features_to_transform"):
            n_transform = len(decisions["features_to_transform"])
            decisions_text += f"- **Transformações:** {n_transform}\n"
        if decisions.get("safe_features"):
            n_safe = len(decisions["safe_features"])
            decisions_text += f"- **Safe features:** {n_safe}\n"
        if not decisions_text:
            decisions_text = "- (nenhuma ainda)\n"
        
        # Próximos passos
        pending = [t for t in self.tasks if not t.get("done")]
        next_steps = ""
        for i, task in enumerate(pending[:5], 1):
            next_steps += f"{i}. {task.get('name', 'N/A')}: {task.get('description', '')}\n"
        if not next_steps:
            next_steps = "- Pipeline concluído\n"
        
        # Alertas
        warnings = meta.get("warnings", [])
        alerts_text = ""
        for w in warnings[-5:]:
            alerts_text += f"- ⚠️ {w}\n"
        if not alerts_text:
            alerts_text = "- Nenhum alerta\n"
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        run_name = self.current_run_dir.name if self.current_run_dir else "N/A"
        
        content = f"""# Estado Atual do Projeto

**Última Atualização:** {timestamp}
**Step Atual:** {self.state.current_step or 'N/A'}
**Run Ativa:** runs/{run_name}/

## Tipo de Problema
- **Detectado:** {problem_type_display}
- **Target:** {meta.get('target_column', 'N/A')}

## Dados
- **Treino:** {n_rows_train:,} linhas × {n_features} features
- **Teste:** {n_rows_test:,} linhas (blind)

## Decisões Tomadas
{decisions_text}

## Métricas Atuais
{metrics_table}

## Próximos Passos
{next_steps}

## Alertas
{alerts_text}

---
*Atualizado automaticamente pelo agente Ralph DS*
"""
        self.state_md_path.write_text(content, encoding='utf-8')
    
    def _detect_problem_type(self) -> str:
        """Detecta tipo de problema baseado nos dados carregados."""
        meta = self.executor.metadata
        
        # Verificar se já foi detectado
        if self.problem_type != "unknown":
            return self.problem_type
        
        # Tentar detectar pelo target
        target_info = meta.get("target_info", {})
        n_unique = target_info.get("n_unique", 0)
        dtype = target_info.get("dtype", "")
        
        if n_unique == 2:
            self.problem_type = "binary_classification"
        elif 2 < n_unique <= 10:
            self.problem_type = "multiclass_classification"
        elif n_unique > 10 or "float" in str(dtype).lower():
            self.problem_type = "regression"
        
        print(f"[BRAIN] Tipo de problema detectado: {self.problem_type}")
        return self.problem_type
    
    def _generate_initial_task_list(self) -> None:
        """Gera TASK_LIST inicial baseada no tipo de problema."""
        problem_config = self.PROBLEM_TYPES.get(self.problem_type, self.PROBLEM_TYPES["unknown"])
        
        eda_steps = problem_config["eda_steps"]
        model_steps = problem_config["model_steps"]
        
        all_steps = eda_steps + model_steps
        
        # Criar tarefas
        new_tasks = []
        for step_name in all_steps:
            desc = self.STEP_PURPOSES.get(step_name, step_name)
            new_tasks.append({
                "name": step_name,
                "description": desc,
                "done": False,
                "current": False
            })
        
        if new_tasks:
            new_tasks[0]["current"] = True
            self.tasks = new_tasks
            self._save_tasks()
            print(f"[BRAIN] TASK_LIST gerada: {len(new_tasks)} tarefas para {self.problem_type}")
    
    def _find_data_paths(self) -> Dict[str, str]:
        """
        Encontra os paths dos arquivos de dados.
        
        Procura em ordem:
        1. context/data/ (preferido)
        2. Raiz do projeto (fallback)
        """
        data_paths = {}
        
        # Extensões de dados suportadas
        data_extensions = [".parquet", ".csv", ".json", ".pkl"]
        
        # 1. Procurar em context/data/
        context_data = self.context_dir / "data"
        if context_data.exists():
            for ext in data_extensions:
                for f in context_data.glob(f"*{ext}"):
                    key = f.stem.lower()
                    if "train" in key:
                        data_paths["train"] = str(f)
                    elif "test" in key:
                        data_paths["test"] = str(f)
                    else:
                        data_paths[key] = str(f)
        
        # 2. Fallback: procurar na raiz
        if "train" not in data_paths:
            for ext in data_extensions:
                train_file = self.base_dir / f"train{ext}"
                if train_file.exists():
                    data_paths["train"] = str(train_file)
                    break
        
        if "test" not in data_paths:
            for ext in data_extensions:
                test_file = self.base_dir / f"test{ext}"
                if test_file.exists():
                    data_paths["test"] = str(test_file)
                    break
        
        return data_paths
    
    def _load_config(self) -> Dict[str, Any]:
        """Carrega config.yaml."""
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return {}
    
    def _load_goals(self) -> str:
        """Carrega GOALS.md."""
        if self.goals_path.exists():
            return self.goals_path.read_text(encoding='utf-8')
        return "Nenhum objetivo definido."
    
    def _load_tasks(self) -> List[Dict[str, Any]]:
        """Carrega e parseia TASK_LIST.md."""
        tasks = []
        marker = " **<-- ATUAL**"
        current_found = False
        if self.task_list_path.exists():
            content = self.task_list_path.read_text(encoding='utf-8')
            for line in content.split('\n'):
                match = re.match(r'- \[([ x])\] (\w+): (.+)', line)
                if match:
                    done = match.group(1) == 'x'
                    name = match.group(2)
                    desc = match.group(3)
                    # Remover marcador "ATUAL" da descrição para não acumular ao salvar
                    while marker.strip() in desc or marker in desc:
                        desc = desc.replace(marker, "").replace("**<-- ATUAL**", "").strip()
                    # Apenas a primeira task com marcador é "current" (evita várias ATUAL)
                    is_current = "**<-- ATUAL**" in line and not current_found
                    if is_current:
                        current_found = True
                    tasks.append({
                        "name": name,
                        "description": desc,
                        "done": done,
                        "current": is_current
                    })
        return tasks

    def _save_tasks(self):
        """Salva TASK_LIST.md atualizado (marcador ATUAL só uma vez por task atual)."""
        lines = ["## Tarefas do Pipeline\n"]
        marker = " **<-- ATUAL**"
        for task in self.tasks:
            check = "x" if task["done"] else " "
            desc = (task.get("description") or "").replace(marker, "").replace("**<-- ATUAL**", "").strip()
            current = marker if task.get("current") else ""
            lines.append(f"- [{check}] {task['name']}: {desc}{current}")
        self.task_list_path.write_text('\n'.join(lines), encoding='utf-8')

    def _load_context_folder(self, max_total_chars: int = 60000) -> str:
        """
        Carrega todo o conteúdo da pasta context/ (documentação, exemplos, código de referência).
        O agente usa isso para planejamento, fluxo e geração de código (não parte do zero).
        
        Estrutura esperada em context/:
        - Documentação (.md, .txt)
        - Exemplos de código (.py, .ipynb)
        - Configurações (.yaml, .json)
        - data/ (subpasta com dados do projeto: .parquet, .csv, .json)
        """
        context_dir = self.base_dir / "context"
        if not context_dir.exists():
            return ""
        
        exts_text = {".md", ".py", ".txt", ".yaml", ".yml", ".json"}
        exts_data = {".parquet", ".csv"}
        parts = []
        total = 0
        
        # Função para processar arquivo
        def process_file(path: Path, prefix: str = "") -> bool:
            nonlocal total
            if path.suffix.lower() in exts_text:
                try:
                    text = path.read_text(encoding="utf-8", errors="replace")
                    head = f"\n--- CONTEXTO: {prefix}{path.name} ---\n"
                    if total + len(head) + len(text) > max_total_chars:
                        text = text[: max_total_chars - total - len(head) - 200] + "\n... (truncado)\n"
                    parts.append(head + text)
                    total += len(head) + len(text)
                    return total >= max_total_chars
                except Exception:
                    pass
            elif path.suffix.lower() in exts_data:
                # Para dados, apenas registrar existência
                try:
                    import os
                    size_mb = os.path.getsize(path) / 1024 / 1024
                    info = f"\n--- DADOS: {prefix}{path.name} ({size_mb:.1f} MB) ---\n"
                    parts.append(info)
                    total += len(info)
                except Exception:
                    pass
            return False
        
        # Processar arquivos na raiz de context/
        for path in sorted(context_dir.iterdir()):
            if path.is_file():
                if process_file(path):
                    break
            elif path.is_dir() and path.name == "data":
                # Processar subpasta data/
                for data_file in sorted(path.iterdir()):
                    if data_file.is_file():
                        if process_file(data_file, "data/"):
                            break
        
        if not parts:
            return ""
        return "CONTEXTO DO PROJETO (pasta context/ - documentação, exemplos e dados):\n" + "\n".join(parts)

    def _load_project_context(self) -> str:
        """
        Carrega contexto do projeto para a LLM.
        
        HIERARQUIA DE CONTEXTO (mais importante primeiro):
        1. STATE.md - Memória resumida do projeto (decisões, métricas, próximos passos)
        2. GOALS.md - Objetivos do projeto
        3. context/ - Documentação, exemplos, dados
        4. report.md da run atual - Outputs recentes
        5. metadata.json - Schema e decisões técnicas
        """
        parts = []
        
        # 1. STATE.md - Memória resumida (contexto principal)
        if self.state_md_path.exists():
            state_text = self.state_md_path.read_text(encoding="utf-8")
            parts.append("ESTADO ATUAL DO PROJETO (STATE.md - memória resumida):\n" + state_text + "\n")
        
        # 2. GOALS.md resumido
        if self.goals_path.exists():
            goals_text = self.goals_path.read_text(encoding="utf-8")[:3000]
            parts.append("OBJETIVOS DO PROJETO (GOALS.md):\n" + goals_text + "\n")
        
        # 3. Tipo de problema detectado
        if self.problem_type != "unknown":
            problem_config = self.PROBLEM_TYPES.get(self.problem_type, {})
            parts.append(
                f"TIPO DE PROBLEMA: {self.problem_type}\n"
                f"Métricas recomendadas: {problem_config.get('metrics', [])}\n"
                f"Modelo padrão: {problem_config.get('default_model', 'N/A')}\n"
            )
        
        # 4. Conteúdo da pasta context/ (documentação, exemplos, dados)
        context_text = self._load_context_folder()
        if context_text:
            parts.append(context_text)
        
        # 5. Metadata técnico
        meta = self.executor.metadata
        
        # Nomes de colunas no projeto
        col_names = []
        for source in ("df_train", "df_test"):
            names = meta.get("data", {}).get(source, {}).get("column_names")
            if isinstance(names, list) and names:
                col_names = names
                break
        if col_names:
            sample = col_names[:30]
            suffix = f" ... e mais {len(col_names) - len(sample)} colunas" if len(col_names) > len(sample) else ""
            parts.append(
                f"COLUNAS DO DATASET:\n"
                f"Exemplo: {sample}{suffix}.\n"
            )
        
        # Decisões já tomadas
        decisions = meta.get("decisions", {})
        if decisions:
            decisions_summary = []
            if decisions.get("features_to_drop"):
                decisions_summary.append(f"features_to_drop: {len(decisions['features_to_drop'])} itens")
            if decisions.get("features_to_transform"):
                decisions_summary.append(f"transformações: {len(decisions['features_to_transform'])} itens")
            if decisions.get("safe_features"):
                decisions_summary.append(f"safe_features: {len(decisions['safe_features'])} itens")
            if decisions_summary:
                parts.append(f"DECISÕES APLICADAS: {', '.join(decisions_summary)}\n")
        
        # 6. Report da run atual (outputs recentes)
        report_md = None
        if self.current_run_dir and self.current_run_dir.exists():
            report_md = self.current_run_dir / "report.md"
        if report_md and report_md.exists():
            text = report_md.read_text(encoding="utf-8")
            # Usar até ~8k chars para não sobrecarregar
            if len(text) > 8000:
                text = text[-8000:]  # Últimas 8k chars (mais recentes)
                text = "... (início truncado)\n" + text
            parts.append(f"REPORT DA RUN ATUAL (runs/{self.current_run_dir.name}/report.md):\n" + text + "\n")
        
        return "\n".join(parts) if parts else ""

    def _is_stub_output(self, stdout: str) -> bool:
        """Detecta se a saída do step indica stub/TODO (não implementado)."""
        if not (stdout or "").strip():
            return False
        s = (stdout or "").strip()
        return "Stub" in s or "TODO pelo agente" in s or "Stub:" in s

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Chama LLM para raciocinio."""
        if self.llm_provider == "gemini":
            return self._call_gemini(system_prompt, user_prompt)
        elif self.llm_provider == "openai":
            return self._call_openai(system_prompt, user_prompt)
        else:
            return self._mock_llm_response(user_prompt)
    
    def _call_gemini(self, system_prompt: str, user_prompt: str) -> str:
        """Chama Gemini API."""
        import google.generativeai as genai
        
        genai.configure(api_key=self.gemini_key)
        model = genai.GenerativeModel(self.model_name)  # Usar MODEL_NAME do .env
        
        full_prompt = f"{system_prompt}\n\n---\n\n{user_prompt}"
        response = model.generate_content(full_prompt)
        
        return response.text
    
    def _call_openai(self, system_prompt: str, user_prompt: str) -> str:
        """Chama OpenAI API."""
        from openai import OpenAI
        
        client = OpenAI(api_key=self.openai_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )
        
        return response.choices[0].message.content
    
    def _mock_llm_response(self, prompt: str) -> str:
        """Resposta mock quando nao ha LLM."""
        return json.dumps({
            "action": "run_step",
            "reasoning": "Modo simulado - executando proximo passo",
            "details": {}
        })
    
    # =========================================================================
    # OODA LOOP
    # =========================================================================
    
    def observe(self) -> Dict[str, Any]:
        """
        OBSERVE: Coleta informacoes do estado atual.
        
        Returns:
            Dicionario com todas as observacoes
        """
        observations = {
            "iteration": self.state.iteration,
            "current_step": self.state.current_step,
            "last_success": self.state.last_result.success if self.state.last_result else None,
            "metadata": self.executor.metadata,
            "config": self.config,
            "tasks": self.tasks,
            "pending_tasks": [t for t in self.tasks if not t["done"]],
            "available_notebooks": self.executor.list_notebooks(),
            "saved_states": self.executor.list_states(),
            "project_context": self._load_project_context(),
        }
        
        # Ler ultimo output se existir
        if self.state.last_result:
            observations["last_stdout"] = self.state.last_result.stdout[:2000]
            observations["last_stderr"] = self.state.last_result.stderr[:1000]
            observations["last_plots"] = self.state.last_result.plots
            observations["last_exception"] = self.state.last_result.exception
            # Stub/TODO = step não implementado; agente DEVE implementar (EDIT_CODE)
            _stub_check = getattr(self, "_is_stub_output", None)
            observations["last_step_is_stub"] = _stub_check(self.state.last_result.stdout or "") if callable(_stub_check) else bool(self.state.last_result.stdout and ("Stub" in self.state.last_result.stdout or "TODO pelo agente" in self.state.last_result.stdout))
        else:
            observations["last_step_is_stub"] = False
        
        return observations
    
    def orient(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        ORIENT: Compara estado atual com objetivos.
        
        Returns:
            Analise da situacao
        """
        analysis = {
            "goals_status": {},
            "blockers": [],
            "opportunities": [],
            "next_priority": None
        }
        
        # Checar metricas vs metas
        metrics = observations["metadata"].get("metrics", {})
        
        if metrics.get("auc_test"):
            if metrics["auc_test"] >= 0.81:
                analysis["goals_status"]["auc"] = "ACHIEVED"
            else:
                analysis["goals_status"]["auc"] = f"PENDING ({metrics['auc_test']:.4f} < 0.81)"
                analysis["blockers"].append("AUC abaixo da meta")
        
        if metrics.get("gap"):
            if metrics["gap"] <= 0.08:
                analysis["goals_status"]["gap"] = "ACHIEVED"
            else:
                analysis["goals_status"]["gap"] = f"OVERFITTING ({metrics['gap']:.2%} > 8%)"
                analysis["blockers"].append("Overfitting detectado")
        
        # Identificar proxima tarefa pendente
        pending = observations["pending_tasks"]
        if pending:
            analysis["next_priority"] = pending[0]
        
        # Checar se houve erro na ultima execucao
        if observations["last_success"] is False:
            analysis["blockers"].append(f"Erro na ultima execucao: {observations['last_exception']}")
        
        # Stub/TODO: step "sucesso" mas não implementado — agente DEVE implementar
        if observations.get("last_step_is_stub"):
            analysis["blockers"].append(
                "Último step é STUB/TODO (não implementado). O agente DEVE usar EDIT_CODE para implementar o script desse step antes de prosseguir."
            )
        
        return analysis
    
    def decide(self, observations: Dict[str, Any], analysis: Dict[str, Any]) -> Tuple[Action, Dict[str, Any]]:
        """
        DECIDE: Escolhe proxima acao baseado na analise.
        
        Returns:
            Tupla (Action, detalhes)
        """
        # Se houve erro, tentar corrigir
        if observations["last_success"] is False and self.state.errors_in_row < self.MAX_ERRORS_IN_ROW:
            action, details = self._decide_error_recovery(observations, analysis)
            # Proativo: se LLM disse stop mas ainda ha tarefas pendentes, tentar proxima em vez de parar
            pending = observations.get("pending_tasks") or []
            if action == Action.STOP and len(pending) > 1:
                next_task = pending[1]
                next_name = next_task.get("name", "")
                if next_name and next_name in observations.get("available_notebooks", []):
                    return Action.RUN_STEP, {"step": next_name}
                if next_name:
                    return Action.WRITE_CODE, {"step": next_name, "description": next_task.get("description", "")}
            return action, details

        # Se muitos erros seguidos, parar
        if self.state.errors_in_row >= self.MAX_ERRORS_IN_ROW:
            return Action.STOP, {"reason": "Muitos erros consecutivos"}
        
        # Stub/TODO: último step "sucesso" mas não implementado — OBRIGATÓRIO implementar antes de prosseguir
        if observations.get("last_step_is_stub") and self.state.current_step:
            step_name = self.state.current_step
            if step_name in (observations.get("available_notebooks") or []):
                return Action.EDIT_CODE, {
                    "step": step_name,
                    "changes": "Implementar a funcionalidade real deste step. Atualmente é um stub que só imprime 'TODO pelo agente'. "
                    "Substituir o stub por código que: (1) carregue estado/metadata do passo anterior, (2) execute a lógica descrita no TASK_LIST para este step, "
                    "(3) salve resultados e atualize metadata. Não deixar como TODO ou Stub."
                }
        
        # Se objetivos atingidos, parar
        if all(v == "ACHIEVED" for v in analysis["goals_status"].values()) and analysis["goals_status"]:
            return Action.STOP, {"reason": "Objetivos atingidos!"}
        
        # Se tem tarefa pendente, executar
        if analysis["next_priority"]:
            task = analysis["next_priority"]
            notebook_name = task["name"]
            
            # Verificar se notebook existe
            if notebook_name in observations["available_notebooks"]:
                return Action.RUN_STEP, {"step": notebook_name}
            else:
                # Precisa criar o notebook
                return Action.WRITE_CODE, {
                    "step": notebook_name,
                    "description": task["description"]
                }
        
        # Se tem plots para analisar
        if observations.get("last_plots"):
            return Action.ANALYZE, {"plots": observations["last_plots"]}
        
        # Nada a fazer
        return Action.STOP, {"reason": "Nenhuma tarefa pendente"}
    
    def _decide_error_recovery(self, observations: Dict[str, Any], analysis: Dict[str, Any]) -> Tuple[Action, Dict[str, Any]]:
        """Decide como recuperar de um erro."""
        error = observations.get("last_exception", "")
        
        # Usar LLM para diagnosticar e sugerir correcao
        system_prompt = """Voce e um debugger de codigo Python especializado em Data Science.
Analise o erro e sugira uma correcao.

REGRAS:
- Use o CONTEXTO DO PROJETO no prompt: neste projeto as colunas sao SEMPRE minusculas (feature_1, feature_2). KeyError 'Feature_1' significa que deve usar 'feature_1'.
- Prefira "edit_code" com codigo Python COMPLETO e valido em code_changes (sem comentarios // ou pseudo-codigo).
- Use "rollback" para voltar ao script anterior; "stop" apenas se erro irreparavel.

Responda em JSON:
{
    "action": "edit_code" | "rollback" | "stop",
    "diagnosis": "O que causou o erro",
    "fix": "Como corrigir",
    "code_changes": "Codigo Python completo corrigido (se action=edit_code). Apenas codigo, sem markdown."
}"""
        
        project_ctx = observations.get("project_context", "")
        user_prompt = f"""
{project_ctx}

---
Erro: {error}

Codigo que falhou (ultimo passo): {self.state.current_step}

Stdout:
{observations.get('last_stdout', '')}

Stderr:
{observations.get('last_stderr', '')}

Use o CONTEXTO DO PROJETO acima: nomes de colunas sao sempre minusculos (feature_1, feature_2). Retorne codigo Python valido em code_changes (sem comentarios // ou pseudo-codigo).
"""
        
        try:
            response = self._call_llm(system_prompt, user_prompt)
            decision = json.loads(re.sub(r'```json|```', '', response).strip())
            
            if decision["action"] == "edit_code":
                return Action.EDIT_CODE, {
                    "step": self.state.current_step,
                    "changes": decision.get("code_changes", ""),
                    "diagnosis": decision["diagnosis"]
                }
            elif decision["action"] == "rollback":
                return Action.ROLLBACK, {"reason": decision["diagnosis"]}
            else:
                return Action.STOP, {"reason": decision["diagnosis"]}
        
        except (json.JSONDecodeError, KeyError):
            return Action.ROLLBACK, {"reason": "Erro nao recuperavel automaticamente"}
    
    def act(self, action: Action, details: Dict[str, Any]) -> ExecutionResult:
        """
        ACT: Executa a acao decidida.
        
        Returns:
            Resultado da execucao
        """
        print(f"\n[BRAIN] Acao: {action.value}")
        details_display = _details_for_display(details)
        details_str = json.dumps(details_display, indent=2, default=str)
        if len(details_str) > 800:
            details_str = details_str[:800].rstrip() + "\n... (truncado)"
        print(f"[BRAIN] Detalhes: {details_str}")
        
        result = None
        
        if action == Action.WRITE_CODE:
            result = self._write_notebook(details["step"], details["description"])
        
        elif action == Action.EDIT_CODE:
            result = self._edit_notebook(details["step"], details["changes"])
            # Após editar, re-executar o step para verificar se ainda é stub
            if result.success and details.get("step"):
                run_result = self._run_step(details["step"])
                self.state.last_result = run_result
                self._update_state_md(action, run_result)
                return run_result
        
        elif action == Action.RUN_STEP:
            result = self._run_step(details["step"])
            # Após EDA do target, tentar detectar tipo de problema
            if result.success and "eda_target" in details.get("step", ""):
                self._detect_problem_type()
        
        elif action == Action.ANALYZE:
            result = self._analyze_outputs(details["plots"])
        
        elif action == Action.ROLLBACK:
            result = self._rollback(details.get("target_step"))
        
        elif action == Action.UPDATE_CONFIG:
            result = self._update_config(details["changes"])

        elif action == Action.UPDATE_TASK_LIST:
            result = self._update_task_list(
                add_steps=details.get("add_steps"),
                remove_steps=details.get("remove_steps"),
                edit_steps=details.get("edit_steps"),
                run_from=details.get("run_from"),
            )

        elif action == Action.RUN_FROM_STEP:
            step = details.get("step") or details.get("run_from")
            if step:
                self.run_from_step(step)
                result = self.state.last_result or ExecutionResult()
            else:
                result = ExecutionResult()
                result.success = False
                result.exception = "RUN_FROM_STEP sem step indicado"
        
        elif action == Action.STOP:
            result = ExecutionResult()
            result.success = True
            result.stdout = f"Agente parou: {details.get('reason', 'Sem motivo')}"
            self.state.goals_achieved = True
        
        else:
            result = ExecutionResult()
            result.success = False
            result.exception = f"Acao desconhecida: {action}"
        
        # Atualizar STATE.md após cada ação
        if result:
            self._update_state_md(action, result)
        
        return result
    
    # =========================================================================
    # ACOES ESPECIFICAS
    # =========================================================================
    
    def _write_notebook(self, step_name: str, description: str) -> ExecutionResult:
        """Escreve um novo notebook usando LLM."""
        
        # Contexto para a LLM
        metadata = json.dumps(self.executor.metadata, indent=2, default=str)
        
        # Determinar onde estão os dados
        data_paths = self._find_data_paths()
        
        system_prompt = f"""Voce e um Senior Data Scientist escrevendo codigo Python para um pipeline de Data Science.

TIPO DE PROBLEMA: {self.problem_type}

REGRAS DE ARQUITETURA:
1. O codigo deve ser STATEFUL: Carregue o pickle do passo anterior se houver.
2. O codigo deve ser CONFIG-DRIVEN: Leia 'state/metadata.json' para saber decisoes anteriores.
   Exemplo de inicio de script:
   ```python
   import json
   with open('state/metadata.json', 'r') as f:
       meta = json.load(f)
   drop_cols = meta['decisions'].get('features_to_drop', [])
   if drop_cols:
       df = df.drop(columns=drop_cols, errors='ignore')
   ```

PATHS DOS DADOS:
{json.dumps(data_paths, indent=2)}

MODO DEV (amostragem):
- As variaveis MODE, IS_DEV, DEV_SAMPLE_FRACTION ja estao no namespace
- Em modo DEV, apos carregar dados use: df = df.sample(frac=DEV_SAMPLE_FRACTION, random_state=42)

REGRAS CRITICAS:
1. Use APENAS: pandas, numpy, matplotlib, seaborn, sklearn, xgboost, scipy
2. Assuma que 'pd', 'np', 'plt', 'sns', 'MODE', 'IS_DEV' ja estao importados no namespace
3. Para carregar dados: use os paths acima (pd.read_parquet ou pd.read_csv)
4. Todo plot DEVE ter comentarios [INTENT], [ESPERADO], [ALERTA] ANTES do plot
5. Salve resultados importantes em variaveis (nao apenas print)
6. Atualize metadata.json com novas decisoes ao final do script

Formato de [INTENT] para plots:
# [INTENT] <O que estamos verificando>
# [ESPERADO] <O que esperamos ver>
# [ALERTA] <O que indicaria problema>
# [DECISAO] <Que acao tomar baseado no resultado>

Retorne APENAS o codigo Python, sem explicacoes ou markdown."""

        project_ctx = self._load_project_context()
        user_prompt = f"""
{project_ctx}

---
Tarefa: {description}

Step Name: {step_name}

Metadata atual dos dados:
{metadata}

Decisoes ja tomadas (aplicar no inicio do script):
{json.dumps(self.executor.metadata.get('decisions', {}), indent=2)}

IMPORTANTE: 
- Nomes de colunas no projeto sao SEMPRE minusculos: feature_1, feature_2, etc. (veja CONTEXTO DO PROJETO acima).
- Leia state/metadata.json no inicio para aplicar decisoes anteriores
- Salve novas decisoes em state/metadata.json ao final
- Dados estao em train.parquet e test.parquet na raiz

Escreva o codigo Python completo para esta tarefa.
"""
        
        try:
            code = self._call_llm(system_prompt, user_prompt)
            
            # Limpar codigo (remover markdown se houver)
            code = re.sub(r'```python|```', '', code).strip()
            
            # Salvar notebook
            notebook_path = self.notebooks_dir / f"{step_name}.py"
            notebook_path.write_text(code, encoding='utf-8')
            
            print(f"[BRAIN] Notebook criado: {notebook_path}")
            
            # Executar o notebook
            return self.executor.run_step(step_name)
        
        except Exception as e:
            result = ExecutionResult()
            result.success = False
            result.exception = f"Erro ao criar notebook: {str(e)}"
            return result
    
    def _edit_notebook(self, step_name: str, changes: str) -> ExecutionResult:
        """Edita um notebook existente."""
        notebook_path = self.notebooks_dir / f"{step_name}.py"
        
        if not notebook_path.exists():
            result = ExecutionResult()
            result.success = False
            result.exception = f"Notebook nao encontrado: {step_name}"
            return result
        
        # Backup antes de editar
        backup_path = self.backups_dir / f"{step_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
        shutil.copy(notebook_path, backup_path)
        
        # Ler codigo atual
        current_code = notebook_path.read_text(encoding='utf-8')
        
        # Usar LLM para aplicar mudancas (com contexto do projeto para evitar Feature_1 vs feature_1)
        project_ctx = self._load_project_context()
        system_prompt = """Voce e um editor de codigo Python. Aplique as mudancas solicitadas ao codigo.
Retorne APENAS o codigo Python corrigido completo, sem explicacoes. Nao use comentarios // (JavaScript); use # em Python.
Se o erro for KeyError de coluna (ex: 'Feature_1'), use o nome correto do CONTEXTO DO PROJETO: feature_1, feature_2 (minusculo)."""
        
        user_prompt = f"""
{project_ctx}

---
Codigo atual:
```python
{current_code}
```

Mudancas a aplicar (traduza para codigo Python valido; use nomes de colunas do contexto acima):
{changes}

Retorne o codigo corrigido completo.
"""
        
        try:
            new_code = self._call_llm(system_prompt, user_prompt)
            new_code = re.sub(r'```python|```', '', new_code).strip()
            
            # Salvar codigo editado
            notebook_path.write_text(new_code, encoding='utf-8')
            
            print(f"[BRAIN] Notebook editado: {notebook_path}")
            print(f"[BRAIN] Backup em: {backup_path}")
            
            # Re-executar
            return self.executor.run_step(step_name)
        
        except Exception as e:
            # Restaurar backup se falhar
            shutil.copy(backup_path, notebook_path)
            result = ExecutionResult()
            result.success = False
            result.exception = f"Erro ao editar notebook: {str(e)}"
            return result
    
    def _run_step(self, step_name: str) -> ExecutionResult:
        """Executa um passo do pipeline."""
        self.state.current_step = step_name

        # Marcar tarefa como atual
        for task in self.tasks:
            task["current"] = (task["name"] == step_name)
        self._save_tasks()

        # Executar
        result = self.executor.run_step(step_name)

        # Se sucesso e não for stub/TODO: marcar tarefa como concluída
        if result.success:
            if not self._is_stub_output(result.stdout or ""):
                for task in self.tasks:
                    if task["name"] == step_name:
                        task["done"] = True
                        task["current"] = False
                self._save_tasks()
            # Análises Vision com contexto rico e append no report
            if self.current_run_dir:
                self._append_vision_analyses_to_report(step_name, result)

        return result

    def _append_vision_analyses_to_report(self, step_name: str, result: ExecutionResult) -> None:
        """Para cada imagem da step, gera contexto rico (o que estamos verificando, para que serve)
        e análise Vision; appenda no report.md da run para o LLM analisar com precisão."""
        report_path = self.current_run_dir / "report.md"
        if not report_path.exists():
            return
        # Lista de imagens: capturadas + PNGs salvos pelo script nesta step
        plot_paths = list(result.plots) if result.plots else []
        reports_dir = self.executor.reports_dir
        for p in reports_dir.glob("*.png"):
            if step_name in p.stem and str(p) not in plot_paths:
                plot_paths.append(str(p))
        if not plot_paths:
            return

        step_purpose = self.STEP_PURPOSES.get(step_name, step_name)
        stdout_snippet = (result.stdout or "").strip()
        if len(stdout_snippet) > 2000:
            stdout_snippet = stdout_snippet[-2000:] + "\n... (truncado)"

        block = "\n\n### Análises das imagens (contexto + Vision)\n\n"
        block += "Para cada gráfico: contexto (o que este step faz e o que estamos verificando) e análise da Vision AI.\n\n"

        for plot_path in plot_paths:
            path = Path(plot_path)
            if not path.exists():
                continue
            name = path.name
            # Contexto rico para a Vision: tipo de gráfico + propósito do step + saída do step
            plot_type = self.vision_critic._detect_plot_type(str(plot_path), "")
            intent_ctx = self.vision_critic.INTENT_TEMPLATES.get(
                plot_type, self.vision_critic.INTENT_TEMPLATES["histogram"]
            )
            code_context = f"""STEP: {step_name}
O que este step faz: {step_purpose}

O que estamos verificando neste gráfico: {intent_ctx.intent}
Esperado: {intent_ctx.expected}
Alertas a procurar: {'; '.join(intent_ctx.alerts[:4])}
Decisão necessária: {intent_ctx.decision_needed}

Saída relevante do step (últimas linhas):
{stdout_snippet or '(nenhuma)'}
"""
            try:
                print(f"   [VISION] Analisando {name} com contexto rico...")
                analysis_result = self.vision_critic.analyze_plot(
                    image_path=str(plot_path),
                    code_context=code_context
                )
                analysis_text = analysis_result.get("analysis", "") if isinstance(analysis_result, dict) else str(analysis_result)
                if analysis_result.get("success") is False:
                    analysis_text = analysis_result.get("error", analysis_text)
            except Exception as e:
                analysis_text = f"Erro na análise Vision: {e}"

            context_for_report = (
                f"**O que este step faz:** {step_purpose}\n\n"
                f"**O que estamos verificando:** {intent_ctx.intent}\n\n"
                f"**Esperado:** {intent_ctx.expected}\n\n"
                f"**Alertas a procurar:** {'; '.join(intent_ctx.alerts[:3])}\n\n"
            )
            block += f"#### {name}\n\n"
            block += f"**Contexto (o que estamos verificando e para que serve):**\n\n{context_for_report}"
            block += f"**Análise (Vision):**\n\n{analysis_text}\n\n"
            block += f"![{name}]({name})\n\n"

        with open(report_path, "a", encoding="utf-8") as f:
            f.write(block)

    def _ordered_steps_from(self, start_step: str) -> List[str]:
        """Lista de nomes de steps em ordem, a partir de start_step (inclusive). start_step pode ser '06' ou '06_feature_cleanup'."""
        available = set(self.executor.list_notebooks())
        ordered = []
        for t in self.tasks:
            name = t.get("name", "")
            if name and name in available:
                ordered.append(name)
        if not ordered:
            return []
        # Resolver start_step: "06" -> "06_feature_cleanup"
        start = start_step.strip()
        if start.isdigit():
            start = next((n for n in ordered if n.startswith(start + "_")), start)
        if start not in ordered:
            idx = next((i for i, n in enumerate(ordered) if n.startswith(start)), 0)
            start = ordered[idx] if idx < len(ordered) else ordered[0]
        try:
            i = ordered.index(start)
            slice_steps = ordered[i:]
        except ValueError:
            slice_steps = ordered
        # Ordenar por número do step (01, 02, ...) para evitar ordem errada vinda do TASK_LIST
        def _step_num(s: str) -> int:
            try:
                return int(s.split("_")[0]) if s.split("_")[0].isdigit() else 999
            except (IndexError, ValueError):
                return 999
        slice_steps.sort(key=_step_num)
        return slice_steps

    def run_from_step(self, start_step: str) -> None:
        """
        Roda o pipeline em ordem a partir do step indicado (ex: 06_feature_cleanup ou 06).
        Cada step lê o state do anterior; útil após editar um step anterior e querer re-executar dali até o fim.
        """
        self.tasks = self._load_tasks()
        # Planejamento dinâmico: TASK_LIST pode ser ajustada antes de rodar (README_RALPH-DS)
        self._planning_phase()
        steps = self._ordered_steps_from(start_step)
        if not steps:
            print(f"[BRAIN] Nenhum step encontrado a partir de '{start_step}'.")
            return
        # Garantir ordem única (evitar duplicatas e reexecução indevida)
        steps = list(dict.fromkeys(steps))
        # Garantir pasta da run (report.md e imagens vão aqui)
        if self.current_run_dir is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.current_run_dir = self.base_dir / "runs" / run_id
            self.current_run_dir.mkdir(parents=True, exist_ok=True)
            self.executor.set_run_dir(self.current_run_dir)
            print(f"[BRAIN] Run desta execução: {self.current_run_dir}")
            print(f"[BRAIN] Report markdown: runs/{self.current_run_dir.name}/report.md")
        print(f"[BRAIN] Run-from: executando {len(steps)} steps em ordem: {steps[0]} .. {steps[-1]}")
        max_recovery = 5  # tentativas de recuperação (edit_code + re-run) antes de desistir
        for i, step_name in enumerate(steps, 1):
            print(f"\n[BRAIN] Run-from step {i}/{len(steps)}: {step_name}")
            result = self._run_step(step_name)
            self.state.last_result = result
            if result.success:
                self.state.errors_in_row = 0
                self._log_to_changelog(Action.RUN_STEP, {"step": step_name}, result)
                # Stub/TODO: step "sucesso" mas não implementado — desmarcar e acionar agente para implementar
                if self._is_stub_output(result.stdout or ""):
                    for task in self.tasks:
                        if task.get("name") == step_name:
                            task["done"] = False
                            break
                    self._save_tasks()
                    print(f"[BRAIN] Step {step_name} é STUB/TODO (não implementado). Desmarcando e acionando agente para implementar...")
                    for attempt in range(5):
                        if not self.run_once():
                            break
                        if self.state.last_result and not self._is_stub_output(self.state.last_result.stdout or ""):
                            print(f"[BRAIN] Step {step_name} implementado com sucesso.")
                            break
                    continue
                continue
            # Step falhou: acionar recuperação OODA (agente analisa erro, edita código, re-executa)
            self.state.errors_in_row += 1
            print(f"[BRAIN] {step_name} falhou. Entrando em modo recuperação (até {max_recovery} tentativas)...")
            for recovery in range(max_recovery):
                print(f"[BRAIN] Recuperação {recovery + 1}/{max_recovery} - agente analisando erro...")
                if not self.run_once():
                    print(f"[BRAIN] Agente encerrou. Run-from parou em {step_name}.")
                    return
                if self.state.last_result and self.state.last_result.success:
                    self.state.errors_in_row = 0
                    self._log_to_changelog(Action.RUN_STEP, {"step": step_name}, self.state.last_result)
                    print(f"[BRAIN] Recuperação ok: {step_name} executado com sucesso.")
                    break
            else:
                print(f"[BRAIN] Step {step_name} falhou após {max_recovery} tentativas; pulando e continuando para o próximo step.")
                continue
        print(f"[BRAIN] Run-from concluído: todos os {len(steps)} steps executados.")
        if self.current_run_dir:
            report_path = self.current_run_dir / "report.md"
            print(f"[BRAIN] Report: {report_path} ({report_path.stat().st_size if report_path.exists() else 0} bytes)")
    
    def _analyze_outputs(self, plots: List[str]) -> ExecutionResult:
        """Analisa outputs visuais usando Vision AI."""
        result = ExecutionResult()
        result.success = True
        analyses = []
        
        # Ler o codigo do passo atual para dar contexto completo
        current_code = ""
        if self.state.current_step:
            script_path = self.notebooks_dir / f"{self.state.current_step}.py"
            if script_path.exists():
                current_code = script_path.read_text(encoding='utf-8')
        
        for plot_path in plots:
            # Tentar extrair intent do codigo que gerou o plot
            intent = self._extract_intent_for_plot(plot_path)
            
            # Analisar com Vision, passando o codigo como contexto
            # Isso permite que o VisionCritic leia as tags [INTENT] do codigo
            analysis = self.vision_critic.analyze_plot(
                image_path=plot_path,
                custom_intent=intent,
                code_context=current_code
            )
            
            analyses.append({
                "plot": plot_path,
                "intent": intent,
                "analysis": analysis
            })
            
            # Atualizar decisoes baseado na analise
            # A analise retornada pelo VisionCritic e um dict com "analysis" (texto)
            if isinstance(analysis, dict):
                text_analysis = analysis.get("analysis", "")
                self._update_decisions_from_analysis(text_analysis)
            else:
                # Fallback se retornar string direta
                self._update_decisions_from_analysis(str(analysis))
        
        result.stdout = json.dumps(analyses, indent=2, default=str)
        return result
    
    def _extract_intent_for_plot(self, plot_path: str) -> str:
        """Extrai [INTENT] do codigo que gerou o plot."""
        # Buscar no ultimo notebook executado
        if self.state.current_step:
            notebook_path = self.notebooks_dir / f"{self.state.current_step}.py"
            if notebook_path.exists():
                code = notebook_path.read_text(encoding='utf-8')
                # Buscar comentarios [INTENT]
                intents = re.findall(r'# \[INTENT\].*?(?=\n[^#]|\Z)', code, re.DOTALL)
                if intents:
                    return intents[-1]  # Ultimo intent antes do plot
        
        return "Analise geral do grafico"
    
    def _update_decisions_from_analysis(self, analysis: str):
        """Atualiza metadata.decisions baseado na analise."""
        # Usar LLM para extrair decisoes da analise
        system_prompt = """Extraia decisoes acionaveis da analise visual.
Retorne JSON:
{
    "features_to_drop": ["lista de features a remover"],
    "features_to_transform": {"feature": "transformacao"},
    "warnings": ["alertas importantes"],
    "action_required": true/false
}"""
        
        try:
            response = self._call_llm(system_prompt, f"Analise: {analysis}")
            decisions = json.loads(re.sub(r'```json|```', '', response).strip())
            
            # Merge com decisoes existentes
            current = self.executor.metadata.get("decisions", {})
            
            if decisions.get("features_to_drop"):
                current.setdefault("features_to_drop", []).extend(decisions["features_to_drop"])
                current["features_to_drop"] = list(set(current["features_to_drop"]))
            
            if decisions.get("features_to_transform"):
                current.setdefault("features_to_transform", {}).update(decisions["features_to_transform"])
            
            if decisions.get("warnings"):
                current.setdefault("warnings", []).extend(decisions["warnings"])
            
            self.executor.metadata["decisions"] = current
            self.executor._save_metadata()
        
        except (json.JSONDecodeError, KeyError):
            pass  # Ignorar se nao conseguir parsear
    
    def _rollback(self, target_step: Optional[str] = None) -> ExecutionResult:
        """Reverte para um estado anterior."""
        result = ExecutionResult()
        
        if target_step:
            success = self.executor.rollback(target_step)
        else:
            # Rollback para ultimo estado bem sucedido
            states = self.executor.list_states()
            if states:
                success = self.executor.rollback(states[-1]["name"])
            else:
                success = False
        
        result.success = success
        result.stdout = f"Rollback {'bem sucedido' if success else 'falhou'}"
        return result

    def _create_stub_script(self, step_name: str, description: str) -> None:
        """Cria script stub em notebooks/ para nova etapa (agente pode preencher depois)."""
        stub = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step: {description}. Stub criado pelo agente; preencher com lógica."""
import os
import json

STEP_NAME = "{step_name}"
METADATA_PATH = os.path.join("state", "metadata.json")
REPORTS_DIR = globals().get("REPORTS_DIR", "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

print(f"[{{STEP_NAME}}] Stub: {{STEP_NAME}} (TODO pelo agente).")
'''
        path = self.notebooks_dir / f"{step_name}.py"
        path.write_text(stub, encoding="utf-8")
        print(f"[BRAIN] Stub criado: {path}")

    def _update_task_list(
        self,
        add_steps: Optional[List[Dict[str, str]]] = None,
        remove_steps: Optional[List[str]] = None,
        edit_steps: Optional[List[Dict[str, str]]] = None,
        run_from: Optional[str] = None,
    ) -> ExecutionResult:
        """
        Replaneja TASK_LIST: adiciona, remove ou edita etapas; cria stubs para novas.
        Persistido em TASK_LIST.md para todas as sessões.
        """
        result = ExecutionResult()
        result.success = True
        add_steps = add_steps or []
        remove_steps = remove_steps or []
        edit_steps = edit_steps or []
        # Normalizar: LLM pode retornar remove_steps como [{"name": "x"}] ou ["x"]
        remove_names = []
        for x in remove_steps:
            if isinstance(x, dict):
                remove_names.append((x.get("name") or "").strip())
            elif isinstance(x, str):
                remove_names.append(x.strip())
        remove_steps = [n for n in remove_names if n]
        self.tasks = self._load_tasks()
        names_set = {t["name"] for t in self.tasks}
        # Remover
        for name in remove_steps:
            self.tasks = [t for t in self.tasks if t["name"] != name]
            names_set.discard(name)
        # Editar descrição (normalizar: ed pode ser dict com name/description)
        for ed in edit_steps:
            if isinstance(ed, dict):
                n, d = ed.get("name"), ed.get("description", "")
            else:
                continue
            n = (n or "").strip()
            d = (d or "").strip()
            for t in self.tasks:
                if t["name"] == n:
                    t["description"] = d
                    break
        # Adicionar (normalizar: item pode ser dict ou string)
        for item in add_steps:
            if isinstance(item, dict):
                n = (item.get("name") or "").strip()
                d = (item.get("description") or "").strip() or n
            elif isinstance(item, str):
                n = item.strip()
                d = n
            else:
                continue
            if n and n not in names_set:
                self.tasks.append({"name": n, "description": d, "done": False, "current": False})
                names_set.add(n)
                if not (self.notebooks_dir / f"{n}.py").exists():
                    self._create_stub_script(n, d)
        # Marcar current: primeiro pendente ou run_from
        current_found = False
        for t in self.tasks:
            if run_from and t["name"] == run_from:
                t["current"] = True
                t["done"] = False
                current_found = True
            elif not run_from and not t["done"] and not current_found:
                t["current"] = True
                current_found = True
            else:
                t["current"] = False
        if run_from and not current_found:
            for t in self.tasks:
                t["current"] = t["name"] == self.tasks[0]["name"] if self.tasks else False
        self._save_tasks()
        result.stdout = f"TASK_LIST atualizado: +{len(add_steps)} -{len(remove_steps)} editados {len(edit_steps)}"
        return result

    def _planning_phase(self) -> None:
        """
        Ciclo de planejamento (README_RALPH-DS): TASK_LIST precisa de ajustes?
        Consulta o LLM com GOALS + contexto + TASK_LIST; se sim, aplica e persiste.
        Disponível em todas as sessões (TASK_LIST.md).
        """
        if self.llm_provider == "mock":
            return
        goals = self._load_goals()[:1500]
        task_list_text = self.task_list_path.read_text(encoding="utf-8") if self.task_list_path.exists() else ""
        context_summary = self._load_project_context()[:2000]
        meta = self.executor.metadata
        current_step = meta.get("current_step", "N/A")
        history = meta.get("history", [])[-5:]
        prompt = f"""GOALS (única referência fixa):
{goals}

TASK_LIST atual:
{task_list_text}

Contexto (resumo): {context_summary[:800]}

Último step em state: {current_step}. Últimos 5 no history: {[h.get('step') for h in history]}.

A TASK_LIST precisa de ajustes estruturais agora? (adicionar etapas que faltam, remover obsoletas, editar descrições, ou indicar run_from para re-executar a partir de um step).
Responda APENAS com um JSON válido, sem markdown:
{{ "add_steps": [{{"name": "NN_nome", "description": "desc"}}], "remove_steps": [], "edit_steps": [{{"name": "NN_nome", "description": "nova desc"}}], "run_from": null }}
Se não precisar de nenhum ajuste, responda: {{}}
"""
        try:
            response = self._call_llm(
                "Você é o planejador do agente. Responda apenas com JSON.",
                prompt,
            )
            raw = re.sub(r"```json|```", "", response).strip()
            decision = json.loads(raw) if raw and raw != "{}" else {}
        except (json.JSONDecodeError, KeyError):
            return
        add_steps = decision.get("add_steps") or []
        remove_steps = decision.get("remove_steps") or []
        edit_steps = decision.get("edit_steps") or []
        run_from = decision.get("run_from")
        if not add_steps and not remove_steps and not edit_steps and not run_from:
            return
        print("[BRAIN] Planejamento: ajustando TASK_LIST conforme prancheta...")
        self._update_task_list(
            add_steps=add_steps,
            remove_steps=remove_steps,
            edit_steps=edit_steps,
            run_from=run_from,
        )
        self.tasks = self._load_tasks()

    def _update_config(self, changes: Dict[str, Any]) -> ExecutionResult:
        """Atualiza config.yaml."""
        result = ExecutionResult()
        
        try:
            # Merge recursivo
            def deep_merge(base, update):
                for k, v in update.items():
                    if isinstance(v, dict) and k in base:
                        deep_merge(base[k], v)
                    else:
                        base[k] = v
            
            deep_merge(self.config, changes)
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            
            result.success = True
            result.stdout = f"Config atualizado: {json.dumps(changes)}"
        
        except Exception as e:
            result.success = False
            result.exception = str(e)
        
        return result
    
    def _log_to_changelog(self, action: Action, details: Dict, result: ExecutionResult):
        """Registra acao no CHANGELOG.md."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        entry = f"""
## Iteracao {self.state.iteration} - {timestamp}
- **Acao:** {action.value}
- **Step:** {self.state.current_step or 'N/A'}
- **Status:** {'SUCCESS' if result.success else 'FAILED'}
- **Detalhes:** {json.dumps(_details_for_display(details), indent=2, default=str)[:500]}
"""
        
        if result.exception:
            entry += f"- **Erro:** {result.exception}\n"
        
        if result.plots:
            entry += f"- **Plots:** {', '.join(result.plots)}\n"
        
        # Prepend ao changelog (mais recente primeiro)
        if self.changelog_path.exists():
            current = self.changelog_path.read_text(encoding='utf-8')
        else:
            current = "# Changelog do Agente\n"
        
        self.changelog_path.write_text(entry + "\n" + current, encoding='utf-8')
    
    def _update_readme(self, metrics: Dict[str, Any] = None):
        """Atualiza README.md com status atual do projeto."""
        readme_path = self.base_dir / "README.md"
        
        # Extrair metricas do metadata
        meta = self.executor.metadata
        decisions = meta.get("decisions", {})
        stored_metrics = meta.get("metrics", {})
        
        if metrics:
            stored_metrics.update(metrics)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Formatar status das metricas
        auc_val = stored_metrics.get("auc_val", "-")
        gap = stored_metrics.get("gap", "-")
        efficiency = stored_metrics.get("efficiency", "-")
        psi = stored_metrics.get("psi", "-")
        profit = stored_metrics.get("profit", "-")
        
        # Determinar status
        def get_status(metric, value, good, bad):
            if value == "-" or value is None:
                return "Pendente"
            try:
                v = float(value)
                if v >= good:
                    return "OK"
                elif v <= bad:
                    return "CRITICO"
                else:
                    return "Atencao"
            except:
                return "Pendente"
        
        # Features removidas
        dropped = decisions.get("features_to_drop", [])
        dropped_str = ", ".join(dropped[:10]) if dropped else "_Nenhuma ainda_"
        if len(dropped) > 10:
            dropped_str += f" ... e mais {len(dropped) - 10}"
        
        # Transformacoes
        transforms = decisions.get("features_to_transform", {})
        transforms_str = "\n".join([f"- {k}: {v}" for k, v in list(transforms.items())[:10]]) if transforms else "_Nenhuma ainda_"
        
        # Warnings
        warnings = meta.get("warnings", [])
        warnings_str = "\n".join([f"- {w}" for w in warnings[:5]]) if warnings else "_Nenhum alerta_"
        
        readme_content = f"""# Credit Scoring Pipeline - Documentacao do Projeto

> **Ultima atualizacao:** {timestamp}
> **Iteracao:** {self.state.iteration}

## Visao Geral

Sistema autonomo de credit scoring que usa um **Agente Cientista de Dados** para:
- Explorar e analisar dados automaticamente (EDA)
- Engenharia de features adaptativa
- Treino e otimizacao de modelos XGBoost
- Analise financeira (lucro, eficiencia, threshold otimo)
- Monitoramento de drift (PSI)
- Documentacao automatica dos resultados

## Status Atual do Pipeline

| Metrica | Valor | Meta | Status |
|---------|-------|------|--------|
| AUC Validacao | {auc_val if auc_val != '-' else '-'} | > 0.81 | {get_status('auc', auc_val, 0.81, 0.75)} |
| Gap Treino-Teste | {f"{gap:.2%}" if isinstance(gap, (int, float)) else gap} | < 8% | {get_status('gap', gap, 0.08, 0.12) if gap != '-' else 'Pendente'} |
| Eficiencia Financeira | {f"{efficiency:.1f}%" if isinstance(efficiency, (int, float)) else efficiency} | > 75% | {get_status('eff', efficiency, 75, 50)} |
| PSI | {f"{psi:.4f}" if isinstance(psi, (int, float)) else psi} | < 0.2 | {get_status('psi', psi, 0.2, 0.25) if psi != '-' else 'Pendente'} |
| Lucro Estimado | {f"R$ {profit:,.2f}" if isinstance(profit, (int, float)) else profit} | Maximizar | - |

## Decisoes Tomadas pelo Agente

### Features Removidas ({len(dropped)} total)
{dropped_str}

### Transformacoes Aplicadas
{transforms_str}

## Alertas e Recomendacoes

{warnings_str}

## Arquitetura do Agente

```
project_root/
|-- brain.py           # Cerebro cognitivo (OODA Loop)
|-- executor.py        # Executor de codigo com estado persistente
|-- vision_critic.py   # Analisador de graficos com Vision AI
|-- config.yaml        # Parametros dinamicos
|-- GOALS.md           # Metas de performance
|-- TASK_LIST.md       # Fila de tarefas
|
|-- src/               # Modulos do pipeline
|-- notebooks/         # Scripts gerados pelo agente
|-- state/             # Estados intermediarios (pickles)
|-- runs/               # Uma pasta por execucao (runs/YYYYMMDD_HHMMSS/) com reports e plots
```

## Como Executar

```bash
# Modo autonomo (loop completo)
python brain.py --mode auto

# Modo passo a passo  
python brain.py --mode step

# Apenas planejar proxima acao
python brain.py --mode plan
```

## Dados

- **Treino:** `train.parquet`
- **Teste:** `test.parquet`
- **Localizacao:** Raiz do projeto

---

_Este README e atualizado automaticamente pelo Agente Cientista de Dados_
"""
        
        readme_path.write_text(readme_content, encoding='utf-8')
        print(f"[BRAIN] README.md atualizado")
    
    # =========================================================================
    # LOOP PRINCIPAL
    # =========================================================================
    
    def run_once(self) -> bool:
        """
        Executa uma iteracao do loop OODA.
        
        Returns:
            True se deve continuar, False se deve parar
        """
        # Se ainda não há pasta da run (ex.: modo step ou primeira iteração), criar uma
        if self.current_run_dir is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.current_run_dir = self.base_dir / "runs" / run_id
            self.current_run_dir.mkdir(parents=True, exist_ok=True)
            self.executor.set_run_dir(self.current_run_dir)
            print(f"[BRAIN] Run desta execução: {self.current_run_dir}")

        self.state.iteration += 1
        print(f"\n{'='*60}")
        print(f"[BRAIN] Iteracao {self.state.iteration}")
        print(f"{'='*60}")
        
        # OBSERVE
        observations = self.observe()
        print(f"[OBSERVE] Tasks pendentes: {len(observations['pending_tasks'])}")
        print(f"[OBSERVE] Ultimo resultado: {observations['last_success']}")
        
        # ORIENT
        analysis = self.orient(observations)
        print(f"[ORIENT] Blockers: {analysis['blockers']}")
        print(f"[ORIENT] Proxima prioridade: {analysis['next_priority']}")
        
        # DECIDE
        action, details = self.decide(observations, analysis)
        print(f"[DECIDE] Acao escolhida: {action.value}")
        
        # ACT
        result = self.act(action, details)
        self.state.last_result = result
        
        # Atualizar contadores
        if result.success:
            self.state.errors_in_row = 0
        else:
            self.state.errors_in_row += 1
        
        # Log
        self._log_to_changelog(action, details, result)
        
        # Atualizar README com status atual
        self._update_readme()
        
        # Verificar se deve parar
        if action == Action.STOP:
            return False
        
        if self.state.iteration >= self.MAX_ITERATIONS:
            print(f"[BRAIN] Limite de iteracoes atingido ({self.MAX_ITERATIONS})")
            return False
        
        return True
    
    def run(self, max_iterations: Optional[int] = None, from_zero: bool = True):
        """
        Executa o loop autonomo ate atingir objetivos ou limite.
        from_zero=True (default): marca todas as tarefas como nao feitas para comecar do step 01.
        """
        if max_iterations:
            self.MAX_ITERATIONS = max_iterations

        # Recarregar tarefas (pode ter sido editado)
        self.tasks = self._load_tasks()
        
        # Se não há tarefas, gerar TASK_LIST inicial baseada no tipo de problema
        if not self.tasks:
            print("[BRAIN] TASK_LIST vazia. Gerando tarefas iniciais...")
            # Se tipo de problema desconhecido, começar com EDA obrigatória
            if self.problem_type == "unknown":
                self._generate_initial_task_list()
            else:
                self._generate_initial_task_list()

        # Default: rodar do zero = primeiro step pendente seja 01_load_data
        if from_zero and self.tasks:
            for task in self.tasks:
                task["done"] = False
                task["current"] = (task["name"] == self.tasks[0]["name"])
            self._save_tasks()
            print("[BRAIN] From-zero: tarefas resetadas; comecando do step 01.")
            # Fase de planejamento (TASK_LIST dinâmica)
            self._planning_phase()

        print("\n" + "="*60)
        print(" RALPH DS v2.0 - AGENTE AUTÔNOMO DE DATA SCIENCE ")
        print("="*60)
        print(f"Objetivos: {self.goals_path}")
        print(f"Config: {self.config_path}")
        print(f"Tipo de Problema: {self.problem_type}")
        print(f"Tarefas: {len(self.tasks)} definidas")
        print("="*60)

        # Nova pasta para esta execução (report.md e imagens ficam aqui)
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_run_dir = self.base_dir / "runs" / run_id
        self.current_run_dir.mkdir(parents=True, exist_ok=True)
        self.executor.set_run_dir(self.current_run_dir)
        print(f"[BRAIN] Run desta execução: {self.current_run_dir}")
        print(f"[BRAIN] Report markdown: runs/{self.current_run_dir.name}/report.md")
        print("="*60)

        try:
            while self.run_once():
                pass
        except KeyboardInterrupt:
            print("\n[BRAIN] Interrompido pelo usuario")
        
        print("\n" + "="*60)
        print(" AGENTE FINALIZADO ")
        print(f" Iteracoes: {self.state.iteration}")
        print(f" Objetivos atingidos: {self.state.goals_achieved}")
        print("="*60)


# CLI
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Agent Brain")
    parser.add_argument(
        "--mode",
        choices=["auto", "run_all", "step", "plan"],
        default="auto",
        help="auto (default): loop OODA - planejamento, analises (vision), retries, edicoes. "
             "run_all: roda 01_load_data ate o fim em sequencia (sem LLM/vision). step: uma iteracao. plan: so planeja.",
    )
    parser.add_argument("--max-iterations", type=int, default=50,
                        help="Maximo de iteracoes (modo auto)")
    parser.add_argument("--resume", action="store_true",
                        help="Continuar de onde parou (nao reseta tarefas; primeiro pendente pode ser 11).")
    parser.add_argument("--run-from", type=str, default=None, metavar="STEP",
                        help="Roda do step indicado ate o fim (ex: 06 ou 06_feature_cleanup). Ignora --mode.")

    args = parser.parse_args()

    brain = AgentBrain()

    if args.run_from:
        brain.run_from_step(args.run_from)
    elif args.mode == "run_all":
        brain.run_from_step("01_load_data")
    elif args.mode == "auto":
        brain.run(max_iterations=args.max_iterations, from_zero=not args.resume)  # OODA: planejamento, vision, retries, report
    elif args.mode == "step":
        brain.run_once()
    elif args.mode == "plan":
        obs = brain.observe()
        analysis = brain.orient(obs)
        action, details = brain.decide(obs, analysis)
        print(f"Proxima acao sugerida: {action.value}")
        details_display = _details_for_display(details)
        details_str = json.dumps(details_display, indent=2)
        if len(details_str) > 800:
            details_str = details_str[:800].rstrip() + "\n... (truncado)"
        print(f"Detalhes: {details_str}")
