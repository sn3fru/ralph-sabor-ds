"""
executor.py - Code Executor with Persistent State

Executa codigo Python mantendo estado entre chamadas:
- Carrega variaveis de pickles anteriores
- Captura stdout, stderr, exceptions
- Salva plots como imagens
- Persiste estado em pickle para proximo passo
- Gera metadata.json com schema dos dados
"""

import sys
import io
import os
import json
import pickle
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from contextlib import redirect_stdout, redirect_stderr
import matplotlib
matplotlib.use('Agg')  # Backend sem GUI
import matplotlib.pyplot as plt
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """
    Encoder JSON customizado que converte tipos NumPy para tipos Python nativos.
    
    Isso evita erros como:
    - TypeError: keys must be str, int, float, bool or None, not int64
    - TypeError: Object of type int64 is not JSON serializable
    """
    def default(self, obj):
        # Inteiros NumPy
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        # Floats NumPy
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        # Booleanos NumPy
        elif isinstance(obj, np.bool_):
            return bool(obj)
        # Arrays NumPy
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        # NaN e Inf
        elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        # Fallback
        return super().default(obj)


def convert_numpy_types(obj):
    """
    Converte recursivamente tipos NumPy em um objeto (dict, list) para tipos Python nativos.
    Util para preparar dados para json.dump quando NumpyEncoder nao e suficiente (ex: chaves de dict).
    """
    if isinstance(obj, dict):
        return {str(k) if isinstance(k, (np.integer, np.floating)) else k: convert_numpy_types(v) 
                for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


class ExecutionResult:
    """Resultado de uma execucao de codigo."""
    
    def __init__(self):
        self.success: bool = False
        self.stdout: str = ""
        self.stderr: str = ""
        self.exception: Optional[str] = None
        self.exception_line: Optional[int] = None
        self.plots: List[str] = []
        self.state_file: Optional[str] = None
        self.metadata: Dict[str, Any] = {}
        self.execution_time: float = 0.0
        self.variables_created: List[str] = []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exception": self.exception,
            "exception_line": self.exception_line,
            "plots": self.plots,
            "state_file": self.state_file,
            "metadata": self.metadata,
            "execution_time": self.execution_time,
            "variables_created": self.variables_created
        }
    
    def __repr__(self):
        status = "SUCCESS" if self.success else "FAILED"
        return f"ExecutionResult({status}, plots={len(self.plots)}, vars={len(self.variables_created)})"


class CodeExecutor:
    """
    Executor de codigo Python com estado persistente.
    
    Funciona como um Jupyter Kernel simplificado:
    - Mantem namespace entre execucoes
    - Captura outputs e plots
    - Persiste estado em pickles
    """
    
    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent
        self.state_dir = self.base_dir / "state"
        self.notebooks_dir = self.base_dir / "notebooks"
        self.reports_dir = self.base_dir / "reports"
        self.backups_dir = self.base_dir / "backups"
        
        # Criar diretorios necessarios
        for d in [self.state_dir, self.notebooks_dir, self.reports_dir, self.backups_dir]:
            d.mkdir(exist_ok=True)
        
        # Namespace persistente (como variaveis do Jupyter)
        self.namespace: Dict[str, Any] = {}
        
        # Imports padrao sempre disponiveis
        self._setup_default_imports()
        
        # Metadata do estado atual
        self.metadata_path = self.state_dir / "metadata.json"
        self.metadata = self._load_metadata()
        
        # Contador de plots
        self._plot_counter = 0
    
    def _setup_default_imports(self):
        """Configura imports padrao no namespace."""
        import pandas as pd
        import numpy as np
        import yaml
        
        self.namespace['pd'] = pd
        self.namespace['np'] = np
        self.namespace['plt'] = plt
        self.namespace['Path'] = Path
        
        # Carregar modo DEV/PROD do config.yaml
        config_path = self.base_dir / "config.yaml"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            mode = config.get('pipeline', {}).get('mode', 'DEV')
        else:
            mode = 'DEV'
        
        # Disponibilizar modo e constantes no namespace
        self.namespace['MODE'] = mode
        self.namespace['DEV_SAMPLE_FRACTION'] = 0.02  # 2% em modo DEV
        self.namespace['IS_DEV'] = (mode == 'DEV')
        
        # Disponibilizar utilidades para JSON com NumPy
        self.namespace['NumpyEncoder'] = NumpyEncoder
        self.namespace['convert_numpy_types'] = convert_numpy_types
        
        print(f"[EXECUTOR] Modo: {mode} {'(2% da base)' if mode == 'DEV' else '(base completa)'}")
        
        # Tentar imports opcionais
        try:
            import xgboost as xgb
            self.namespace['xgb'] = xgb
        except ImportError:
            pass
        
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import roc_auc_score, roc_curve
            self.namespace['train_test_split'] = train_test_split
            self.namespace['roc_auc_score'] = roc_auc_score
            self.namespace['roc_curve'] = roc_curve
        except ImportError:
            pass
        
        try:
            import seaborn as sns
            self.namespace['sns'] = sns
        except ImportError:
            pass
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Carrega metadata do estado atual."""
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            "current_step": None,
            "data": {},
            "decisions": {
                "features_to_drop": [],
                "features_to_transform": {},
                "imputation_strategy": "median"
            },
            "metrics": {},
            "warnings": [],
            "history": []
        }
    
    def _save_metadata(self):
        """Salva metadata atualizado, convertendo tipos NumPy automaticamente."""
        # Converter tipos NumPy para tipos Python nativos (incluindo chaves de dict)
        clean_metadata = convert_numpy_types(self.metadata)
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(clean_metadata, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    
    def load_state(self, step_name: str) -> bool:
        """
        Carrega estado de um passo anterior.
        
        Args:
            step_name: Nome do passo (ex: "step_01_raw")
        
        Returns:
            True se carregou com sucesso
        """
        state_file = self.state_dir / f"{step_name}.pkl"
        if state_file.exists():
            with open(state_file, 'rb') as f:
                state = pickle.load(f)
            self.namespace.update(state)
            print(f"[EXECUTOR] Estado carregado: {step_name} ({len(state)} variaveis)")
            return True
        return False
    
    def save_state(self, step_name: str, variables: Optional[List[str]] = None) -> str:
        """
        Salva estado atual em pickle.
        
        Args:
            step_name: Nome do passo
            variables: Lista de variaveis a salvar (None = todas)
        
        Returns:
            Caminho do arquivo salvo
        """
        state_file = self.state_dir / f"{step_name}.pkl"
        
        # Filtrar variaveis serializaveis
        state = {}
        vars_to_save = variables or list(self.namespace.keys())
        
        for var in vars_to_save:
            if var in self.namespace and not var.startswith('_'):
                try:
                    # Testar se e serializavel
                    pickle.dumps(self.namespace[var])
                    state[var] = self.namespace[var]
                except (pickle.PicklingError, TypeError, AttributeError):
                    pass  # Pular variaveis nao serializaveis
        
        with open(state_file, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"[EXECUTOR] Estado salvo: {step_name} ({len(state)} variaveis)")
        return str(state_file)
    
    def _capture_plots(self, step_name: str) -> List[str]:
        """Captura e salva todos os plots abertos."""
        plots = []
        
        # Pegar todas as figuras abertas
        fig_nums = plt.get_fignums()
        for fig_num in fig_nums:
            fig = plt.figure(fig_num)
            if fig.get_axes():  # So salvar se tiver conteudo
                self._plot_counter += 1
                timestamp = datetime.now().strftime("%H%M%S")
                plot_name = f"{step_name}_{timestamp}_{self._plot_counter}.png"
                plot_path = self.reports_dir / plot_name
                
                fig.savefig(plot_path, dpi=100, bbox_inches='tight', facecolor='white')
                plots.append(str(plot_path))
        
        # Fechar todos os plots
        plt.close('all')
        
        return plots
    
    def _extract_dataframe_metadata(self) -> Dict[str, Any]:
        """Extrai metadata de DataFrames no namespace."""
        import pandas as pd
        
        meta = {}
        for name, obj in self.namespace.items():
            if isinstance(obj, pd.DataFrame) and not name.startswith('_'):
                meta[name] = {
                    "rows": len(obj),
                    "columns": len(obj.columns),
                    "column_names": list(obj.columns[:20]),  # Primeiras 20
                    "dtypes": {str(k): str(v) for k, v in obj.dtypes.items()},
                    "null_counts": obj.isnull().sum().to_dict(),
                    "memory_mb": obj.memory_usage(deep=True).sum() / 1024 / 1024
                }
        return meta
    
    def run_code(self, code: str, step_name: str = "unnamed") -> ExecutionResult:
        """
        Executa codigo Python capturando outputs.
        
        Args:
            code: Codigo Python a executar
            step_name: Nome do passo para logs e estado
        
        Returns:
            ExecutionResult com todos os outputs
        """
        result = ExecutionResult()
        
        # Capturar stdout e stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        # Variaveis antes da execucao
        vars_before = set(self.namespace.keys())
        
        start_time = datetime.now()
        
        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Executar no namespace persistente
                exec(code, self.namespace)
            
            result.success = True
            
        except Exception as e:
            result.success = False
            result.exception = f"{type(e).__name__}: {str(e)}"
            
            # Tentar extrair numero da linha do erro
            tb = traceback.extract_tb(sys.exc_info()[2])
            if tb:
                result.exception_line = tb[-1].lineno
            
            # Adicionar traceback ao stderr
            stderr_capture.write(traceback.format_exc())
        
        # Calcular tempo de execucao
        result.execution_time = (datetime.now() - start_time).total_seconds()
        
        # Capturar outputs
        result.stdout = stdout_capture.getvalue()
        result.stderr = stderr_capture.getvalue()
        
        # Capturar plots
        result.plots = self._capture_plots(step_name)
        
        # Variaveis criadas
        vars_after = set(self.namespace.keys())
        result.variables_created = list(vars_after - vars_before)
        
        # Extrair metadata de DataFrames
        result.metadata = self._extract_dataframe_metadata()
        
        # Atualizar metadata global
        self.metadata["current_step"] = step_name
        self.metadata["data"].update(result.metadata)
        self.metadata["history"].append({
            "step": step_name,
            "timestamp": datetime.now().isoformat(),
            "success": result.success,
            "execution_time": result.execution_time
        })
        self._save_metadata()
        
        # Salvar estado se sucesso
        if result.success:
            result.state_file = self.save_state(step_name)
        
        return result
    
    def run_step(self, step_name: str, load_previous: Optional[str] = None) -> ExecutionResult:
        """
        Executa um script da pasta notebooks/.
        
        Args:
            step_name: Nome do script (sem .py)
            load_previous: Estado anterior a carregar antes
        
        Returns:
            ExecutionResult
        """
        script_path = self.notebooks_dir / f"{step_name}.py"
        
        if not script_path.exists():
            result = ExecutionResult()
            result.success = False
            result.exception = f"Script nao encontrado: {script_path}"
            return result
        
        # Carregar estado anterior se especificado
        if load_previous:
            self.load_state(load_previous)
        
        # Ler e executar codigo
        code = script_path.read_text(encoding='utf-8')
        return self.run_code(code, step_name)
    
    def rollback(self, step_name: str) -> bool:
        """
        Reverte para um estado anterior.
        
        Args:
            step_name: Nome do estado para reverter
        
        Returns:
            True se sucesso
        """
        # Limpar namespace atual (manter imports)
        keys_to_remove = [k for k in self.namespace.keys() 
                         if k not in ['pd', 'np', 'plt', 'Path', 'xgb', 'sns',
                                     'train_test_split', 'roc_auc_score', 'roc_curve']]
        for k in keys_to_remove:
            del self.namespace[k]
        
        return self.load_state(step_name)
    
    def get_variable(self, name: str) -> Any:
        """Retorna uma variavel do namespace."""
        return self.namespace.get(name)
    
    def set_variable(self, name: str, value: Any):
        """Define uma variavel no namespace."""
        self.namespace[name] = value
    
    def list_states(self) -> List[Dict[str, Any]]:
        """Lista todos os estados salvos."""
        states = []
        for f in self.state_dir.glob("*.pkl"):
            stat = f.stat()
            states.append({
                "name": f.stem,
                "size_mb": stat.st_size / 1024 / 1024,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
        return sorted(states, key=lambda x: x["modified"])
    
    def list_notebooks(self) -> List[str]:
        """Lista todos os scripts em notebooks/."""
        return sorted([f.stem for f in self.notebooks_dir.glob("*.py")])


def format_execution_report(result: ExecutionResult, step_name: str) -> str:
    """Formata resultado da execucao como markdown."""
    status_emoji = "SUCCESS" if result.success else "FAILED"
    
    report = f"""## Execution Report: {step_name}

**Status:** {status_emoji}
**Execution Time:** {result.execution_time:.2f}s
**Variables Created:** {', '.join(result.variables_created) or 'None'}
**Plots Generated:** {len(result.plots)}

### Standard Output
```
{result.stdout or '(empty)'}
```

### Standard Error
```
{result.stderr or '(empty)'}
```
"""
    
    if result.exception:
        report += f"""
### Exception
```
{result.exception}
Line: {result.exception_line}
```
"""
    
    if result.plots:
        report += "\n### Plots Generated\n"
        for plot in result.plots:
            report += f"- {plot}\n"
    
    if result.metadata:
        report += "\n### DataFrames in Memory\n"
        for name, meta in result.metadata.items():
            report += f"- **{name}**: {meta['rows']} rows x {meta['columns']} cols\n"
    
    return report


# CLI para testes
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Code Executor")
    parser.add_argument("--step", help="Run a specific step from notebooks/")
    parser.add_argument("--rollback", help="Rollback to a specific state")
    parser.add_argument("--list-states", action="store_true", help="List saved states")
    parser.add_argument("--list-notebooks", action="store_true", help="List available notebooks")
    
    args = parser.parse_args()
    
    executor = CodeExecutor()
    
    if args.list_states:
        states = executor.list_states()
        print("Saved States:")
        for s in states:
            print(f"  - {s['name']} ({s['size_mb']:.2f} MB, {s['modified']})")
    
    elif args.list_notebooks:
        notebooks = executor.list_notebooks()
        print("Available Notebooks:")
        for n in notebooks:
            print(f"  - {n}")
    
    elif args.rollback:
        success = executor.rollback(args.rollback)
        print(f"Rollback to {args.rollback}: {'SUCCESS' if success else 'FAILED'}")
    
    elif args.step:
        result = executor.run_step(args.step)
        print(format_execution_report(result, args.step))
    
    else:
        # Modo interativo simples
        print("Code Executor - Interactive Mode")
        print("Type Python code, 'run <step>' to run a notebook, or 'quit' to exit")
        
        while True:
            try:
                code = input(">>> ")
                if code.strip() == 'quit':
                    break
                elif code.startswith('run '):
                    step = code[4:].strip()
                    result = executor.run_step(step)
                    print(result)
                else:
                    result = executor.run_code(code, "interactive")
                    if result.stdout:
                        print(result.stdout)
                    if result.exception:
                        print(f"Error: {result.exception}")
            except (KeyboardInterrupt, EOFError):
                break
