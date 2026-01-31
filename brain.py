"""
brain.py - Cerebro Cognitivo do Agente Cientista de Dados

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


class Action(Enum):
    """Acoes que o agente pode tomar."""
    PLAN = "plan"                    # Planejar proximos passos
    WRITE_CODE = "write_code"        # Escrever novo script
    EDIT_CODE = "edit_code"          # Editar script existente
    RUN_STEP = "run_step"            # Executar um script
    ANALYZE = "analyze"              # Analisar output/plot
    ROLLBACK = "rollback"            # Reverter para estado anterior
    UPDATE_CONFIG = "update_config"  # Atualizar config.yaml
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
        self.src_dir = self.base_dir / "src"
        self.notebooks_dir = self.base_dir / "notebooks"
        self.backups_dir = self.base_dir / "backups"
        
        # Estado do agente
        self.state = AgentState()
        
        # Carregar configuracoes
        self.config = self._load_config()
        self.goals = self._load_goals()
        self.tasks = self._load_tasks()
        
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
        if self.task_list_path.exists():
            content = self.task_list_path.read_text(encoding='utf-8')
            # Parse markdown checkboxes
            for line in content.split('\n'):
                match = re.match(r'- \[([ x])\] (\w+): (.+)', line)
                if match:
                    done = match.group(1) == 'x'
                    name = match.group(2)
                    desc = match.group(3)
                    tasks.append({
                        "name": name,
                        "description": desc,
                        "done": done,
                        "current": "**<-- ATUAL**" in line
                    })
        return tasks
    
    def _save_tasks(self):
        """Salva TASK_LIST.md atualizado."""
        lines = ["## Tarefas do Pipeline\n"]
        for task in self.tasks:
            check = "x" if task["done"] else " "
            current = " **<-- ATUAL**" if task.get("current") else ""
            lines.append(f"- [{check}] {task['name']}: {task['description']}{current}")
        
        self.task_list_path.write_text('\n'.join(lines), encoding='utf-8')

    def _load_project_context(self) -> str:
        """
        Carrega contexto do projeto para a LLM: nomes de colunas (metadata),
        resumo do README_ANALISE/report e decisoes ja tomadas.
        Evita sugestoes 'junior' (ex: Feature_1 vs feature_1).
        """
        parts = []
        meta = self.executor.metadata
        # Nomes de colunas no projeto (sempre lowercase: feature_1, feature_2, ...)
        col_names = []
        for source in ("df_train", "df_test"):
            names = meta.get("data", {}).get(source, {}).get("column_names")
            if isinstance(names, list) and names:
                col_names = names
                break
        if col_names:
            sample = col_names[:40]
            suffix = f" ... e mais {len(col_names) - len(sample)} colunas" if len(col_names) > len(sample) else ""
            parts.append(
                "CONTEXTO DO PROJETO - NOMES DE COLUNAS:\n"
                "No dataset, as colunas de features sao sempre em minusculo: feature_1, feature_2, feature_3, etc.\n"
                f"Exemplo (primeiras): {sample}{suffix}.\n"
                "NAO use 'Feature_1' ou 'Feature_2' com F maiusculo; use 'feature_1', 'feature_2'.\n"
            )
        decisions = meta.get("decisions", {})
        if decisions:
            drop = decisions.get("features_to_drop", [])
            if drop:
                parts.append(f"Decisoes ja aplicadas: features_to_drop tem {len(drop)} itens (ex: {drop[:5]}).\n")
        # Resumo do README_ANALISE ou ultimo report
        readme_analise = self.base_dir / "README_ANALISE.md"
        if readme_analise.exists():
            text = readme_analise.read_text(encoding="utf-8")[:1800]
            parts.append("RESUMO DO RELATORIO DE ANALISE (README_ANALISE.md):\n" + text + "\n")
        else:
            reports_dir = self.base_dir / "reports"
            if reports_dir.exists():
                md_files = sorted(reports_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
                if md_files:
                    text = md_files[0].read_text(encoding="utf-8")[:1500]
                    parts.append("RESUMO DO ULTIMO REPORT (reports/):\n" + text + "\n")
        return "\n".join(parts) if parts else ""

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
        print(f"[BRAIN] Detalhes: {json.dumps(details, indent=2, default=str)}")
        
        if action == Action.WRITE_CODE:
            return self._write_notebook(details["step"], details["description"])
        
        elif action == Action.EDIT_CODE:
            return self._edit_notebook(details["step"], details["changes"])
        
        elif action == Action.RUN_STEP:
            return self._run_step(details["step"])
        
        elif action == Action.ANALYZE:
            return self._analyze_outputs(details["plots"])
        
        elif action == Action.ROLLBACK:
            return self._rollback(details.get("target_step"))
        
        elif action == Action.UPDATE_CONFIG:
            return self._update_config(details["changes"])
        
        elif action == Action.STOP:
            result = ExecutionResult()
            result.success = True
            result.stdout = f"Agente parou: {details.get('reason', 'Sem motivo')}"
            self.state.goals_achieved = True
            return result
        
        else:
            result = ExecutionResult()
            result.success = False
            result.exception = f"Acao desconhecida: {action}"
            return result
    
    # =========================================================================
    # ACOES ESPECIFICAS
    # =========================================================================
    
    def _write_notebook(self, step_name: str, description: str) -> ExecutionResult:
        """Escreve um novo notebook usando LLM."""
        
        # Contexto para a LLM
        metadata = json.dumps(self.executor.metadata, indent=2, default=str)
        
        system_prompt = """Voce e um Senior Data Scientist escrevendo codigo Python para um pipeline de Credit Scoring.

REGRAS DE ARQUITETURA:
1. O codigo deve ser STATEFUL: Carregue o pickle do passo anterior se houver.
2. O codigo deve ser CONFIG-DRIVEN: Leia 'state/metadata.json' para saber quais features dropar/transformar.
   Exemplo de inicio de script:
   ```python
   import json
   with open('state/metadata.json', 'r') as f:
       meta = json.load(f)
   drop_cols = meta['decisions'].get('features_to_drop', [])
   # Aplicar decisoes anteriores
   if drop_cols:
       df = df.drop(columns=drop_cols, errors='ignore')
   ```

MODO DEV (2% da base):
- As variaveis MODE, IS_DEV, DEV_SAMPLE_FRACTION ja estao no namespace
- Em modo DEV, apos carregar dados use: df = df.sample(frac=DEV_SAMPLE_FRACTION, random_state=42)
- Isso acelera o desenvolvimento sem mudar a logica

REGRAS CRITICAS:
1. Use APENAS: pandas, numpy, matplotlib, seaborn, sklearn, xgboost, scipy
2. Assuma que 'pd', 'np', 'plt', 'sns', 'MODE', 'IS_DEV' ja estao importados no namespace
3. Para carregar dados: pd.read_parquet('train.parquet')
4. Todo plot DEVE ter comentarios [INTENT], [ESPERADO], [ALERTA] ANTES do plot
5. Salve resultados importantes em variaveis (nao apenas print)
6. Atualize o metadata.json com novas decisoes ao final do script

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
        
        # Se sucesso, marcar tarefa como concluida
        if result.success:
            for task in self.tasks:
                if task["name"] == step_name:
                    task["done"] = True
                    task["current"] = False
            self._save_tasks()
        
        return result
    
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
- **Detalhes:** {json.dumps(details, indent=2, default=str)[:500]}
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
|-- reports/           # Relatorios markdown + imagens
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
    
    def run(self, max_iterations: Optional[int] = None):
        """
        Executa o loop autonomo ate atingir objetivos ou limite.
        """
        if max_iterations:
            self.MAX_ITERATIONS = max_iterations
        
        print("\n" + "="*60)
        print(" AGENTE CIENTISTA DE DADOS - INICIANDO ")
        print("="*60)
        print(f"Objetivos: {self.goals_path}")
        print(f"Config: {self.config_path}")
        print(f"Tarefas: {len(self.tasks)} definidas")
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
    parser.add_argument("--mode", choices=["auto", "step", "plan"], default="auto",
                       help="Modo de execucao")
    parser.add_argument("--max-iterations", type=int, default=50,
                       help="Maximo de iteracoes")
    
    args = parser.parse_args()
    
    brain = AgentBrain()
    
    if args.mode == "auto":
        brain.run(max_iterations=args.max_iterations)
    elif args.mode == "step":
        brain.run_once()
    elif args.mode == "plan":
        obs = brain.observe()
        analysis = brain.orient(obs)
        action, details = brain.decide(obs, analysis)
        print(f"Proxima acao sugerida: {action.value}")
        print(f"Detalhes: {json.dumps(details, indent=2)}")
