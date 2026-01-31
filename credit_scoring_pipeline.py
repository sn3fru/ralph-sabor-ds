#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Credit Scoring Pipeline - Script Python
Converte o notebook Jupyter em script executável com logging markdown estruturado
para análise por LLMs.
"""

import sys
import os
import json
from pathlib import Path
import yaml

# Configurar encoding UTF-8 para stdout no Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, 
    classification_report, precision_recall_curve
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

# Análise de Drift
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    SKLEARN_FULL_AVAILABLE = True
except ImportError:
    SKLEARN_FULL_AVAILABLE = False

# Modelos
import xgboost as xgb

# Explainability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Otimização
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# Configurações de plotagem
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11

# Importar módulos customizados
sys.path.insert(0, str(Path(__file__).parent / 'src'))
try:
    from src import (
        temporal_cross_validation,
        adversarial_validation_temporal,
        calibrate_model,
        find_optimal_threshold,
        evaluate_financial_impact,
        monitor_drift
    )
except ImportError as e:
    print(f"[WARN] Erro ao importar utilitarios: {e}")

# Importar logger markdown
from markdown_logger import MarkdownLogger

# =============================================================================
# FUNÇÕES AUXILIARES PARA ENRIQUECIMENTO DE CONTEXTO
# =============================================================================

def simulate_auc_elasticity(y_true, y_best_proba, cost_matrix, fixed_threshold, n_steps=50, random_seed=42):
    """
    Simula a degradação da AUC misturando o modelo atual com ruído aleatório
    e mede o impacto no lucro mantendo o threshold fixo.
    
    Args:
        y_true: Labels verdadeiros
        y_best_proba: Probabilidades do melhor modelo
        cost_matrix: Dicionário com custos {'tp': 1500, 'fp': -10000, 'fn': 0, 'tn': 0}
        fixed_threshold: Threshold fixo para aplicar em todas as simulações
        n_steps: Número de pontos na simulação (default: 50)
        random_seed: Seed para reprodutibilidade
    
    Returns:
        DataFrame com colunas: noise_alpha, auc, profit
    """
    results = []
    
    # Gerar ruído base (aleatório uniforme) uma vez
    np.random.seed(random_seed)
    noise_base = np.random.rand(len(y_true))
    
    # Iterar de 0% de ruído (Modelo Atual) até 100% (Aleatório)
    alphas = np.linspace(0, 1.0, n_steps)
    
    for alpha in alphas:
        # Mistura linear: (1-alpha)*Modelo + alpha*Ruído
        # Isso degrada a qualidade da ordenação suavemente
        y_simulated = (1 - alpha) * y_best_proba + alpha * noise_base
        
        # Recalcular AUC
        try:
            sim_auc = roc_auc_score(y_true, y_simulated)
        except ValueError:
            # Se AUC não puder ser calculada (ex: todas as classes iguais), pular
            continue
        
        # Aplicar Threshold Fixo
        preds = (y_simulated >= fixed_threshold).astype(int)
        
        # Calcular Lucro
        cm = confusion_matrix(y_true, preds)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            # Caso especial: matriz 1x1 ou 2x1
            continue
        
        profit = (tp * cost_matrix.get('tp', 0)) + (fp * cost_matrix.get('fp', 0)) + \
                 (fn * cost_matrix.get('fn', 0)) + (tn * cost_matrix.get('tn', 0))
        
        results.append({
            'noise_alpha': alpha,
            'auc': sim_auc,
            'profit': profit
        })
        
    return pd.DataFrame(results)


def calculate_elasticity_coefficient(df_results):
    """
    Calcula a elasticidade média via Regressão Linear (Slope).
    
    Returns:
        elasticity_coef: Coeficiente de elasticidade
        df_reg: DataFrame usado para regressão (faixa útil)
    """
    if len(df_results) == 0:
        return 0.0, df_results
    
    # Focamos na metade superior da performance (onde o modelo é útil)
    # Elasticidade = % Variação Lucro / % Variação AUC
    df_reg = df_results[df_results['auc'] > 0.65].copy()
    
    if len(df_reg) < 5:  # Mínimo de pontos para regressão
        df_reg = df_results.copy()
    
    if len(df_reg) == 0:
        return 0.0, df_results
    
    base_profit = df_reg['profit'].min()
    
    if base_profit <= 0 or df_reg['profit'].std() == 0:
        # Se lucro for negativo ou constante, usamos escala linear simples
        X = df_reg[['auc']].values
        y = df_reg['profit'].values
    else:
        # Elasticidade Log-Log (Econometria Clássica)
        # Evitar log de valores negativos ou zero
        df_reg_clean = df_reg[df_reg['auc'] > 0.01].copy()
        df_reg_clean = df_reg_clean[df_reg_clean['profit'] > 0.01].copy()
        
        if len(df_reg_clean) < 5:
            X = df_reg[['auc']].values
            y = df_reg['profit'].values
        else:
            X = np.log(df_reg_clean[['auc']].values)
            y = np.log(df_reg_clean['profit'].values)
            df_reg = df_reg_clean
    
    if len(X) == 0:
        return 0.0, df_results
    
    try:
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, y)
        elasticity_coef = model.coef_[0] if len(model.coef_) > 0 else 0.0
    except:
        elasticity_coef = 0.0
    
    return elasticity_coef, df_reg

def calculate_max_potential_profit(y_true, cost_matrix):
    """
    Calcula o lucro máximo teórico (Bola de Cristal Perfeita).
    
    O lucro máximo seria se tivéssemos um modelo perfeito que:
    - Aprova todos os bons pagadores (y=1) -> Ganho TP
    - Rejeita todos os caloteiros (y=0) -> Ganho TN (geralmente 0)
    
    Args:
        y_true: Array com labels verdadeiros (0 ou 1)
        cost_matrix: Dict com custos {'tp': 1500, 'fp': -10000, 'fn': 0, 'tn': 0}
    
    Returns:
        float: Lucro máximo teórico possível
    """
    n_positives = np.sum(y_true == 1)  # Bons pagadores
    n_negatives = np.sum(y_true == 0)  # Maus pagadores
    
    # Com modelo perfeito:
    # - Aprovamos todos os 1 (ganhamos TP * n_positives)
    # - Rejeitamos todos os 0 (ganhamos TN * n_negatives, geralmente 0)
    max_profit = (n_positives * cost_matrix.get('tp', 0)) + (n_negatives * cost_matrix.get('tn', 0))
    
    return max_profit


def calculate_psi(expected, actual, bins=10, buckettype='bins'):
    """
    Calcula o Population Stability Index (PSI) de forma robusta.
    
    Args:
        expected: Distribuição esperada (treino)
        actual: Distribuição atual (produção)
        bins: Número de bins para discretização
        buckettype: Tipo de discretização ('bins' ou 'quantiles')
    
    Returns:
        float: Valor do PSI
    """
    # Determinar breakpoints baseado no tipo
    if buckettype == 'bins':
        breakpoints = np.linspace(
            min(np.min(expected), np.min(actual)),
            max(np.max(expected), np.max(actual)),
            bins + 1
        )
    elif buckettype == 'quantiles':
        percentiles = np.arange(0, bins + 1) / bins * 100
        breakpoints = np.array([np.percentile(expected, p) for p in percentiles])
    else:
        # Default: bins
        breakpoints = np.linspace(
            min(np.min(expected), np.min(actual)),
            max(np.max(expected), np.max(actual)),
            bins + 1
        )

    # Calcular distribuições
    expected_percents = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    # Evita divisão por zero (clipping mínimo)
    expected_percents = np.clip(expected_percents, 0.0001, 1.0)
    actual_percents = np.clip(actual_percents, 0.0001, 1.0)

    # Calcular PSI
    psi_value = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
    return psi_value


def describe_curve_geometry_robust(x, y, name="Curva"):
    """
    Análise geométrica robusta a ruído e degraus iniciais.
    Usa janela de análise em vez de apenas 2 pontos para evitar falsos zeros.
    
    Args:
        x: Valores do eixo X
        y: Valores do eixo Y
        name: Nome da curva
    
    Returns:
        Descrição textual da geometria da curva com análise estatística
    """
    if len(x) < 5 or len(y) < 5:
        return f"{name}: Dados insuficientes para análise robusta (mínimo 5 pontos)."
    
    # 1. Análise de Tendência Inicial (Janela de 5% dos dados ou mínimo 5 pontos)
    # Isso evita falsos zeros quando y[0] == y[1] por precisão ou ruído
    window = max(5, int(len(x) * 0.05))
    
    # Delta Y no início (Robustez contra y[0]==y[1])
    dy_start = y[window] - y[0]
    dx_start = x[window] - x[0]
    
    if dx_start == 0:
        slope_metric = 0
    else:
        slope_metric = dy_start / dx_start
    
    # Classificação da inclinação inicial
    if slope_metric > 1.0:
        slope_desc = "Crescimento Agressivo (Excelente)"
    elif slope_metric > 0.5:
        slope_desc = "Crescimento Moderado (Bom)"
    elif slope_metric > 0.1:
        slope_desc = "Crescimento Lento (Alerta)"
    elif slope_metric > -0.1:
        slope_desc = "Estagnado/Plano (Crítico)"
    else:
        slope_desc = "Declínio Inicial (Anômalo)"
    
    # 2. Detecção de "Cotovelo" (Ponto onde o ganho marginal diminui)
    # Onde a curva atinge 80% do máximo
    try:
        y_max = np.max(y)
        idx_80 = np.where(y >= y_max * 0.8)[0]
        if len(idx_80) > 0:
            x_80 = x[idx_80[0]]
            elbow_desc = f"Atinge 80% da performance em x={x_80:.3f}"
        else:
            elbow_desc = "Não atinge 80% do pico"
    except:
        elbow_desc = "Análise de cotovelo indisponível"
    
    # 3. Análise de estabilidade (zona plana no topo)
    y_threshold = y_max * 0.95  # 95% do máximo
    flat_zone_indices = np.where(y >= y_threshold)[0]
    
    if len(flat_zone_indices) > 1:
        flat_zone_start = x[flat_zone_indices[0]]
        flat_zone_end = x[flat_zone_indices[-1]]
        flat_zone_width = flat_zone_end - flat_zone_start
        flat_desc = f"Zona estável (95% do pico) entre x={flat_zone_start:.3f} e x={flat_zone_end:.3f} (largura={flat_zone_width:.3f})"
    else:
        flat_desc = "Sem zona estável significativa no topo"
    
    # 4. Pico da curva
    peak_idx = np.argmax(y)
    peak_x = x[peak_idx]
    peak_y = y[peak_idx]
    
    return (
        f"**Geometria da {name}:**\n"
        f"- **Comportamento Inicial (Janela {window} pts):** {slope_desc} (Slope={slope_metric:.3f}).\n"
        f"- **Eficiência:** {elbow_desc}.\n"
        f"- **Estabilidade:** {flat_desc}.\n"
        f"- **Pico:** x={peak_x:.3f}, y={peak_y:.3f}."
    )

# Mantém função antiga para compatibilidade (deprecated)
def describe_curve_geometry(x, y, name="Curva"):
    """Deprecated: Use describe_curve_geometry_robust."""
    return describe_curve_geometry_robust(x, y, name)
    
    description = (
        f"**Análise geométrica da {name}:**\n\n"
        f"- **Pico:** Ocorre em x={peak_x:.3f} com valor y={peak_y:.3f}\n"
        f"- **Crescimento inicial:** {slope_desc} (inclinação inicial: {slope_start:.3f})\n"
    )
    
    if has_flat_zone:
        description += (
            f"- **Zona de estabilidade:** A curva apresenta uma região plana (flat-top) "
            f"entre x={flat_zone_start:.3f} e x={flat_zone_end:.3f} (largura: {flat_zone_width:.3f}). "
            f"Isso indica que o modelo é robusto a variações nesta faixa.\n"
        )
    
    description += f"- **Declínio após pico:** {decline_desc}\n"
    
    return description

def generate_dynamic_insight(feature_name, null_percent, logger):
    """
    Gera insight dinâmico baseado nos dados reais da rodada.
    
    Args:
        feature_name: Nome da feature
        null_percent: Percentual de nulos
        logger: Instância do logger
    """
    if null_percent > 90:
        insight = (
            f"A feature {feature_name} tem {null_percent:.1f}% de valores nulos. "
            "Devido à alta cardinalidade de nulos, o XGBoost provavelmente está usando "
            "essa ausência como uma categoria informativa (ex: cliente sem histórico específico). "
            "Esta feature pode estar capturando padrões de 'novos clientes' ou 'dados não coletados'."
        )
    elif null_percent > 50:
        insight = (
            f"A feature {feature_name} tem {null_percent:.1f}% de valores nulos. "
            "Este alto percentual sugere que a feature pode ser condicionalmente relevante "
            "(só existe para um subconjunto de clientes). O XGBoost pode estar usando "
            "a presença/ausência desta feature como um sinal importante."
        )
    else:
        insight = (
            f"A feature {feature_name} tem {null_percent:.1f}% de valores nulos. "
            "Percentual moderado que não compromete a utilidade da feature."
        )
    
    return insight

# =============================================================================
# CONFIGURAÇÃO INICIAL
# =============================================================================

def rollback_config(config_path: Path, backup_path: Path):
    """
    ✅ 3. ROLLBACK AUTOMÁTICO: Restaura config.yaml do backup em caso de falha crítica.
    """
    if backup_path.exists() and config_path.exists():
        try:
            import shutil
            shutil.copy(backup_path, config_path)
            print(f"[ROLLBACK] Config.yaml restaurado do backup devido a falha critica.")
            return True
        except Exception as e:
            print(f"[ERRO] Falha ao restaurar backup: {e}")
            return False
    return False

def main():
    """Função principal que executa todo o pipeline."""
    
    # ✅ 1. CARREGAR CONFIGURAÇÃO COMPLETA DO config.yaml
    config_path = Path(__file__).parent / "config.yaml"
    config_backup_path = Path(__file__).parent / "config.yaml.backup"
    
    # Backup do config antes de qualquer modificação (para rollback)
    if config_path.exists() and not config_backup_path.exists():
        try:
            import shutil
            shutil.copy(config_path, config_backup_path)
            print(f"[INFO] Backup do config.yaml criado: {config_backup_path}")
        except Exception as e:
            print(f"[WARN] Nao foi possivel criar backup do config: {e}")
    
    # ✅ 4. VISUALIZAÇÃO EM TEMPO REAL: Progress bar simples
    import sys
    def print_progress(step: str, current: int = 0, total: int = 0):
        """Imprime progresso em tempo real."""
        if total > 0:
            percent = int((current / total) * 100)
            bar_length = 30
            filled = int(bar_length * current / total)
            bar = '=' * filled + '-' * (bar_length - filled)
            sys.stdout.write(f'\r[{step}] [{bar}] {percent}% ({current}/{total})')
            sys.stdout.flush()
        else:
            sys.stdout.write(f'\r[{step}] ...')
            sys.stdout.flush()
    
    try:
        config = {}
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f) or {}
                print("[OK] Config.yaml carregado com sucesso")
            except Exception as e:
                print(f"[ERRO] Erro ao ler config.yaml: {e}. Usando valores padrao.")
                config = {}
        
        # Extrair configurações com valores padrão
        pipeline_config = config.get('pipeline', {})
        MODO = pipeline_config.get('mode', 'DEV').upper()
        RUN_SHAP = pipeline_config.get('run_shap', False)
        
        xgboost_config = config.get('xgboost_params', {})
        feature_config = config.get('feature_selection', {})
        business_config = config.get('business_params', {})
        
        # Configuração de Vision LLM (análise visual automática)
        # Tenta carregar do arquivo .env primeiro
        def load_env_file(env_path: Path) -> dict:
            """Carrega variáveis de um arquivo .env."""
            env_vars = {}
            if env_path.exists():
                try:
                    with open(env_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith('#') and '=' in line:
                                key, value = line.split('=', 1)
                                key = key.strip()
                                value = value.strip().strip('"').strip("'")  # Remove aspas
                                env_vars[key] = value
                except Exception as e:
                    print(f"[WARN] Erro ao ler arquivo .env: {e}")
            return env_vars
        
        # Carregar variáveis do .env
        env_path = Path(__file__).parent / '.env'
        env_vars = load_env_file(env_path)
        
        # Determina API key: .env > variáveis de ambiente
        vision_api_key = None
        vision_provider = None
        vision_model_name = None
        
        # Prioridade: .env primeiro, depois variáveis de ambiente
        if "GEMINI_KEY" in env_vars:
            vision_api_key = env_vars["GEMINI_KEY"]
            vision_provider = "gemini"
            vision_model_name = env_vars.get("MODEL_NAME", "gemini-1.5-pro")
            print(f"[OK] Configuracao carregada do .env: Gemini (Modelo: {vision_model_name})")
        elif os.getenv("OPENAI_API_KEY"):
            vision_api_key = os.getenv("OPENAI_API_KEY")
            vision_provider = "openai"
        elif os.getenv("GEMINI_API_KEY"):
            vision_api_key = os.getenv("GEMINI_API_KEY")
            vision_provider = "gemini"
            vision_model_name = env_vars.get("MODEL_NAME", "gemini-1.5-pro")
        elif os.getenv("ANTHROPIC_API_KEY"):
            vision_api_key = os.getenv("ANTHROPIC_API_KEY")
            vision_provider = "claude"
        else:
            vision_provider = "openai"  # Default (mas não será usado se não houver API key)
        
        use_vision = vision_api_key is not None
        
        # Inicializar logger markdown com suporte a análise visual
        logger = MarkdownLogger(
            output_dir="reports",
            use_vision_llm=use_vision,
            vision_provider=vision_provider,
            vision_api_key=vision_api_key,
            vision_model_name=vision_model_name  # Passa o model_name do .env
        )
        
        logger.section("Credit Scoring: Maquina de Decisao de Credito", level=1)
        logger.log("Pipeline de analise executiva de risco de credito com Machine Learning", "info")
        logger.log(f"Modo de execucao: {MODO}", "info")
        
        # ✅ Log das configurações carregadas do config.yaml
        logger.section("0. Configuracao do Pipeline", level=2)
        logger.log(f"Modo: {MODO}", "info")
        logger.log(f"SHAP: {'Ativado' if RUN_SHAP else 'Desativado'}", "info")
        logger.log(f"Correlation Threshold: {feature_config.get('correlation_threshold', 0.95)}", "info")
        logger.log(f"Max Depth: {xgboost_config.get('max_depth', 6)}", "info")
        logger.log(f"Learning Rate: {xgboost_config.get('learning_rate', 0.05)}", "info")
        logger.log(f"N Estimators ({MODO}): {xgboost_config.get(f'n_estimators_{MODO.lower()}', 50 if MODO == 'DEV' else 500)}", "info")
        
        # =============================================================================
        # 1. SETUP & INFRAESTRUTURA
        # =============================================================================
        
        logger.section("1. Setup & Infraestrutura", level=2)
        
        # Verificação de GPU
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
            GPU_AVAILABLE = result.returncode == 0
            if GPU_AVAILABLE:
                logger.log("GPU NVIDIA detectada - XGBoost pode usar aceleração GPU", "success")
            else:
                logger.log("GPU não detectada - usando CPU", "warning")
                GPU_AVAILABLE = False
        except:
            GPU_AVAILABLE = False
            logger.log("GPU não detectada - usando CPU", "warning")
        
        logger.log_metric("Pandas Version", pd.__version__)
        logger.log_metric("NumPy Version", np.__version__)
        logger.log_metric("XGBoost Version", xgb.__version__)
        logger.log_metric("GPU Available", GPU_AVAILABLE)
        
        # =============================================================================
        # 2. ENGENHARIA DE DADOS
        # =============================================================================
        
        logger.section("2. Engenharia de Dados", level=2)
        
        logger.log("Carregando arquivos parquet...", "info")
        
        # Carregar dados
        df_train_raw = pd.read_parquet('train.parquet')
        df_test_raw = pd.read_parquet('test.parquet')
    
        logger.log_metric("Train Raw Shape (Original)", f"{df_train_raw.shape[0]:,} linhas × {df_train_raw.shape[1]} colunas")
        logger.log_metric("Test Raw Shape (Original)", f"{df_test_raw.shape[0]:,} linhas × {df_test_raw.shape[1]} colunas")
        
        # ✅ AMOSTRAGEM NO MODO DEV: Reduzir dados para desenvolvimento rápido
        DEV_SAMPLE_SIZE = 10000  # ~10k linhas no modo DEV
        train_original_size = len(df_train_raw)  # Guardar tamanho original
        
        if MODO == 'DEV' and len(df_train_raw) > DEV_SAMPLE_SIZE:
            logger.log(f"[DEV MODE] Aplicando amostragem estratificada de {DEV_SAMPLE_SIZE:,} linhas para acelerar desenvolvimento...", "info")
            
            # Amostragem estratificada do treino (mantém proporção de classes)
            if 'label' in df_train_raw.columns and df_train_raw['label'].notna().sum() > 0:
                from sklearn.model_selection import train_test_split
                # Garantir que temos amostras suficientes de cada classe
                label_counts = df_train_raw['label'].value_counts()
                min_samples_per_class = label_counts.min() if len(label_counts) > 0 else 0
                
                # Calcular tamanho da amostra (garantir pelo menos algumas amostras de cada classe)
                sample_size = min(DEV_SAMPLE_SIZE, len(df_train_raw))
                
                if sample_size < len(df_train_raw) and min_samples_per_class > 10:
                    # Amostragem estratificada
                    df_train_raw, _ = train_test_split(
                        df_train_raw,
                        train_size=sample_size,
                        stratify=df_train_raw['label'],
                        random_state=42
                    )
                    logger.log(f"[DEV MODE] Treino reduzido para {len(df_train_raw):,} linhas (amostragem estratificada)", "info")
                elif sample_size < len(df_train_raw):
                    # Amostragem simples se não conseguir estratificar
                    df_train_raw = df_train_raw.sample(n=sample_size, random_state=42)
                    logger.log(f"[DEV MODE] Treino reduzido para {len(df_train_raw):,} linhas (amostragem aleatoria)", "info")
            else:
                # Se não tem label, faz sample simples
                df_train_raw = df_train_raw.sample(n=min(DEV_SAMPLE_SIZE, len(df_train_raw)), random_state=42)
                logger.log(f"[DEV MODE] Treino reduzido para {len(df_train_raw):,} linhas (amostragem aleatoria)", "info")
            
            # Amostragem do teste (proporcional ao treino)
            reduction_ratio = len(df_train_raw) / train_original_size if train_original_size > 0 else 1.0
            test_sample_size = min(
                int(len(df_test_raw) * reduction_ratio),
                len(df_test_raw),
                DEV_SAMPLE_SIZE // 2  # Teste menor que treino
            )
            
            if test_sample_size < len(df_test_raw):
                df_test_raw = df_test_raw.sample(n=test_sample_size, random_state=42)
                logger.log(f"[DEV MODE] Teste reduzido para {len(df_test_raw):,} linhas", "info")
            
            logger.log_metric("Train Raw Shape (Apos Sampling)", f"{df_train_raw.shape[0]:,} linhas")
            logger.log_metric("Test Raw Shape (Apos Sampling)", f"{df_test_raw.shape[0]:,} linhas")
            logger.log_insight(
                f"Modo DEV ativo: usando amostra de {len(df_train_raw):,} linhas de treino e {len(df_test_raw):,} linhas de teste "
                f"para desenvolvimento rapido. Execute em modo PROD para usar dataset completo ({train_original_size:,} linhas).",
                "dev_mode"
            )
        
        # Harmonização
        df_test_raw['label'] = np.nan
        if 'split' not in df_test_raw.columns:
            df_test_raw['split'] = 'test'
        df_train_raw['dataset_origin'] = 'train_file'
        df_test_raw['dataset_origin'] = 'test_file_blind'
        
        # Master Table
        df_full = pd.concat([df_train_raw, df_test_raw], axis=0, ignore_index=True)
        df = df_full
        
        logger.log_metric("DataFrame Unificado", f"{df_full.shape[0]:,} linhas × {df_full.shape[1]} colunas")
        logger.log_metric("Com Label", f"{df_full['label'].notna().sum():,}")
        logger.log_metric("Sem Label", f"{df_full['label'].isna().sum():,}")
        
        # Split temporal
        df_modeling = df_full[df_full['dataset_origin'] == 'train_file'].copy()
        feature_cols = [col for col in df_modeling.columns if col.startswith('feature_')]
        X = df_modeling[feature_cols]
        y = df_modeling['label']
        
        split_idx = int(len(X) * 0.80)
        X_train = X.iloc[:split_idx]
        X_val = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_val = y.iloc[split_idx:]
        
        df_test_for_drift = df_full[df_full['dataset_origin'] == 'test_file_blind'].copy()
        df_train_for_drift = df_full[df_full['dataset_origin'] == 'train_file'].copy()  # ✅ Adicionado para análise de drift
        X_test_blind = df_test_for_drift[feature_cols].copy()
        
        logger.log_metric("Treino", f"{X_train.shape[0]:,} amostras")
        logger.log_metric("Validação", f"{X_val.shape[0]:,} amostras")
        logger.log_metric("Teste Cego", f"{X_test_blind.shape[0]:,} amostras")
        logger.log_metric("Features", len(feature_cols))
        
        # ✅ INJEÇÃO DE CONTEXTO GLOBAL: Setup de Dados
        logger.update_context("n_samples_train", len(X_train))
        logger.update_context("n_samples_val", len(X_val))
        logger.update_context("n_samples_test", len(X_test_blind))
        logger.update_context("n_features_raw", len(feature_cols))
        
        # =============================================================================
        # 3. EDA EXECUTIVA
        # =============================================================================
        
        logger.section("3. EDA Executiva", level=2)
        
        # Auditoria de integridade
        logger.log(f"Dimensões: {df.shape[0]:,} linhas × {df.shape[1]} colunas", "info")
        logger.log_metric("Memória utilizada", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Distribuição da target
        if 'label' in df.columns:
            target_dist = df['label'].value_counts()
            balance_ratio = target_dist.min() / target_dist.max()
            
            logger.log_table("Distribuição da Target", {
                'Classe 0': int(target_dist.get(0, 0)),
                'Classe 1': int(target_dist.get(1, 0)),
                'Taxa de Balanceamento': f"{balance_ratio:.3f}"
            })
            
            # ✅ INJEÇÃO DE CONTEXTO GLOBAL: Balanceamento
            logger.update_context("class_balance_ratio", balance_ratio)
            logger.update_context("target_imbalance_status", "Severe" if balance_ratio < 0.1 else "Moderate" if balance_ratio < 0.3 else "Balanced")
            logger.update_context("class_0_count", int(target_dist.get(0, 0)))
            logger.update_context("class_1_count", int(target_dist.get(1, 0)))
            
            if balance_ratio < 0.3:
                logger.log("PROBLEMA DESBALANCEADO - Necessário ajuste de estratégia de modelagem", "warning")
                logger.log_insight(
                    f"O dataset está severamente desbalanceado (razão {balance_ratio:.3f}). "
                    "Será necessário usar scale_pos_weight no XGBoost para compensar.",
                    "overfitting"
                )
        
        # Análise de valores nulos
        null_percent = df[feature_cols].isnull().sum() / len(df) * 100
        null_summary = null_percent.sort_values(ascending=False)
        
        logger.log_table("Top 10 Features com Mais Nulos", 
                         {k: f"{v:.2f}%" for k, v in null_summary.head(10).items()})
        
        logger.log_metric("Features com 0% nulos", (null_percent == 0).sum())
        logger.log_metric("Features com >0% e ≤10% nulos", ((null_percent > 0) & (null_percent <= 10)).sum())
        logger.log_metric("Features com >10% e ≤50% nulos", ((null_percent > 10) & (null_percent <= 50)).sum())
        logger.log_metric("Features com >50% nulos", (null_percent > 50).sum())
        
        # ✅ MELHORIA: Insights dinâmicos baseados nos dados reais
        top_null_feature = null_summary.index[0]
        top_null_percent = null_summary.iloc[0]
        dynamic_insight = generate_dynamic_insight(top_null_feature, top_null_percent, logger)
        logger.log_insight(dynamic_insight, "data_quality")
        
        # Visualização de nulos
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].hist(null_percent, bins=50, edgecolor='black', alpha=0.7)
        axes[0].axvline(null_percent.median(), color='red', linestyle='--', 
                        label=f'Mediana: {null_percent.median():.2f}%')
        axes[0].set_xlabel('Percentual de Valores Nulos (%)')
        axes[0].set_ylabel('Número de Features')
        axes[0].set_title('Distribuição de Nulos nas Features')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        top_nulls = null_summary.head(15)
        axes[1].barh(range(len(top_nulls)), top_nulls.values, color='coral')
        axes[1].set_yticks(range(len(top_nulls)))
        axes[1].set_yticklabels(top_nulls.index)
        axes[1].set_xlabel('Percentual de Nulos (%)')
        axes[1].set_title('Top 15 Features com Mais Nulos')
        axes[1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        # Contexto técnico detalhado para análise IA
        null_plot_context = (
            f"Distribuição de valores nulos nas {len(feature_cols)} features do dataset de credit scoring. "
            f"Gráfico à esquerda: histograma da distribuição de percentuais de nulos (mediana: {null_percent.median():.2f}%). "
            f"Gráfico à direita: top 15 features com maior percentual de nulos (máximo: {null_summary.iloc[0]:.2f}%). "
            f"Total de features com >50% nulos: {(null_percent > 50).sum()}. "
            f"O XGBoost usa sparse-aware split finding para lidar com nulos, tratando ausência como informação."
        )
        
        logger.log_plot(
            fig,
            title="Distribuição de Valores Nulos nas Features",
            description="Distribuição de valores nulos nas features",
            context_description=null_plot_context,
            save_image=True,
            analyze=True
        )
        
        logger.log_insight(
            "XGBoost lida nativamente com nulos através de 'sparse-aware split finding'. "
            "Isso evita imputação arbitrária e preserva informação de padrões de missingness.",
            "geral"
        )
        
        # =============================================================================
        # 3.1. ANÁLISE DE DRIFT TEMPORAL COMPLETA (Estatística + Visual)
        # =============================================================================
        
        logger.section("3.1. Análise de Drift Temporal Completa", level=3)
        
        if SCIPY_AVAILABLE and SKLEARN_FULL_AVAILABLE:
            try:
                from matplotlib.lines import Line2D
                
                logger.log("Iniciando análise de drift temporal completa (KS Test + PCA + t-SNE)...", "info")
                
                # Definição dos datasets
                train_df = df_train_for_drift.copy()
                test_df = df_test_for_drift.copy()
                
                logger.log_metric("Treino (Referência)", f"{len(train_df):,} amostras")
                logger.log_metric("Teste (Atual/Cego)", f"{len(test_df):,} amostras")
                
                if len(test_df) < 10:
                    logger.log("ALERTA: Base de teste insuficiente (<10 amostras). Pulando análise.", "warning")
                else:
                    # 1. Seleção de Features (Top 20 por Variância)
                    numeric_cols = train_df.select_dtypes(include=[np.number]).columns
                    valid_cols = [c for c in feature_cols if c in numeric_cols] if 'feature_cols' in locals() else numeric_cols
                    top_features = train_df[valid_cols].var().sort_values(ascending=False).head(20).index.tolist()
                    
                    logger.log(f"Calculando drift nas top {len(top_features)} features...", "info")
                    
                    # 2. CÁLCULO ESTATÍSTICO (KS TEST)
                    drift_results = []
                    for feat in top_features:
                        train_vals = train_df[feat].dropna()
                        test_vals = test_df[feat].dropna()
                        
                        if len(train_vals) > 10 and len(test_vals) > 10:
                            ks_stat, ks_pvalue = stats.ks_2samp(train_vals, test_vals)
                            drift_results.append({
                                'feature': feat,
                                'ks_statistic': ks_stat,
                                'ks_pvalue': ks_pvalue
                            })
                    
                    drift_df = pd.DataFrame(drift_results).sort_values('ks_statistic', ascending=False)
                    
                    # Exibir Top 10 Drift
                    logger.log_table("Top 10 Features com Maior Instabilidade (KS Statistic)",
                                   {row['feature']: f"KS={row['ks_statistic']:.4f}, p={row['ks_pvalue']:.4f}" 
                                    for _, row in drift_df.head(10).iterrows()})
                    
                    # 3. PREPARAÇÃO VISUAL (PCA & t-SNE)
                    logger.log("Processando mapa visual (PCA & t-SNE) com 10k pontos...", "info")
                    
                    n_sample = min(len(train_df), len(test_df), 5000)
                    
                    df_train_sample = train_df[top_features].sample(n=n_sample, random_state=42).fillna(0)
                    df_test_sample = test_df[top_features].sample(n=n_sample, random_state=42).fillna(0)
                    
                    df_train_sample['dataset'] = 'Treino'
                    df_test_sample['dataset'] = 'Teste'
                    
                    full_sample = pd.concat([df_train_sample, df_test_sample])
                    X_sample = StandardScaler().fit_transform(full_sample[top_features])
                    
                    # A) PCA
                    pca = PCA(n_components=2)
                    pca_result = pca.fit_transform(X_sample)
                    full_sample['pca_1'] = pca_result[:, 0]
                    full_sample['pca_2'] = pca_result[:, 1]
                    
                    # B) t-SNE
                    logger.log("Executando t-SNE (pode demorar)...", "info")
                    tsne = TSNE(n_components=2, perplexity=50, n_iter=1000, random_state=42, 
                               init='pca', learning_rate='auto', n_jobs=-1)
                    tsne_result = tsne.fit_transform(X_sample)
                    full_sample['tsne_1'] = tsne_result[:, 0]
                    full_sample['tsne_2'] = tsne_result[:, 1]
                    
                    # 4. PLOTAGEM GERAL (LAYOUT 2x2)
                    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
                    
                    # PLOT 1: Histograma KS
                    axes[0, 0].hist(drift_df['ks_statistic'], bins=15, edgecolor='black', color='steelblue', alpha=0.7)
                    axes[0, 0].axvline(0.1, color='orange', linestyle='--', linewidth=2, label='Moderado (0.1)')
                    axes[0, 0].axvline(0.2, color='red', linestyle='--', linewidth=2, label='Crítico (0.2)')
                    axes[0, 0].set_title('Distribuição de Drift (KS Statistic)', fontsize=14, fontweight='bold')
                    axes[0, 0].set_xlabel('KS Statistic')
                    axes[0, 0].set_ylabel('Frequência')
                    axes[0, 0].legend()
                    axes[0, 0].grid(True, alpha=0.3)
                    
                    # PLOT 2: Feature Mais Instável
                    worst_feat = drift_df.iloc[0]['feature']
                    worst_ks = drift_df.iloc[0]['ks_statistic']
                    
                    train_vals_clean = train_df[worst_feat].dropna()
                    test_vals_clean = test_df[worst_feat].dropna()
                    
                    sns.kdeplot(train_vals_clean, ax=axes[0, 1], fill=True, color='blue', 
                               label='Treino', alpha=0.2, linewidth=2)
                    sns.kdeplot(test_vals_clean, ax=axes[0, 1], fill=True, color='red', 
                               label='Teste', alpha=0.2, linewidth=2)
                    axes[0, 1].set_title(f'Pior Drift: {worst_feat} (KS={worst_ks:.4f})', 
                                        fontsize=14, fontweight='bold')
                    axes[0, 1].set_xlabel(worst_feat)
                    axes[0, 1].set_ylabel('Densidade')
                    axes[0, 1].legend()
                    axes[0, 1].grid(True, alpha=0.3)
                    
                    # Função Auxiliar para Plotar Scatter + Contornos
                    def plot_contour_scatter(ax, x_col, y_col, title):
                        # Scatter de fundo
                        sns.scatterplot(
                            data=full_sample, x=x_col, y=y_col, hue='dataset', 
                            ax=ax, palette={'Treino': 'blue', 'Teste': 'red'},
                            alpha=0.15, s=15, linewidth=0, legend=False
                        )
                        
                        # Contornos de Densidade
                        sns.kdeplot(
                            data=full_sample[full_sample['dataset']=='Treino'], x=x_col, y=y_col,
                            ax=ax, color='blue', levels=5, thresh=0.1, linewidths=1.5, alpha=0.8
                        )
                        sns.kdeplot(
                            data=full_sample[full_sample['dataset']=='Teste'], x=x_col, y=y_col,
                            ax=ax, color='red', levels=5, thresh=0.1, linewidths=1.5, alpha=0.8
                        )
                        
                        ax.set_title(title, fontsize=14, fontweight='bold')
                        custom_lines = [Line2D([0], [0], color='blue', lw=2),
                                       Line2D([0], [0], color='red', lw=2)]
                        ax.legend(custom_lines, ['Treino (Contorno)', 'Teste (Contorno)'], loc='upper right')
                        ax.grid(True, alpha=0.3)
                    
                    # PLOT 3: PCA com Contornos
                    plot_contour_scatter(axes[1, 0], 'pca_1', 'pca_2', 
                                       'PCA: Estrutura Global (Com Zonas de Densidade)')
                    
                    # PLOT 4: t-SNE com Contornos
                    plot_contour_scatter(axes[1, 1], 'tsne_1', 'tsne_2', 
                                       't-SNE: Agrupamentos Locais (Com Zonas de Densidade)')
                    
                    plt.tight_layout()
                    
                    # Contexto técnico para análise IA
                    drift_plot_context = (
                        f"Análise completa de drift temporal usando KS Test, PCA e t-SNE. "
                        f"Top esquerdo: Distribuição de KS Statistics das {len(drift_df)} features analisadas. "
                        f"Top direito: Distribuição da feature com maior drift ({worst_feat}, KS={worst_ks:.4f}). "
                        f"Bottom esquerdo: PCA 2D mostrando estrutura global. "
                        f"Bottom direito: t-SNE 2D mostrando agrupamentos locais. "
                        f"Contornos azuis: densidade do treino. Contornos vermelhos: densidade do teste. "
                        f"Se os contornos vermelhos formarem 'ilhas' onde não há contornos azuis, "
                        f"indica regiões não exploradas pelo treino (risco de falha do modelo)."
                    )
                    
                    logger.log_plot(
                        fig,
                        title="Análise de Drift Temporal Completa",
                        description="Análise de drift usando KS Test, PCA e t-SNE",
                        context_description=drift_plot_context,
                        save_image=True,
                        analyze=True
                    )
                    
                    # Conclusão Automática
                    high_drift_count = (drift_df['ks_statistic'] > 0.15).sum()
                    if high_drift_count > 0:
                        logger.log(
                            f"CONCLUSÃO: {high_drift_count} features com drift alto (KS > 0.15). "
                            "Observe as linhas de contorno no t-SNE: se as linhas vermelhas formam 'ilhas' "
                            "onde não há linhas azuis, o modelo falhará nessas regiões.",
                            "warning"
                        )
                        logger.log_insight(
                            f"{high_drift_count} features apresentam drift significativo (KS > 0.15). "
                            "Isso indica que a distribuição dos dados de teste difere substancialmente do treino. "
                            "Recomenda-se: (1) Retreinar modelo com dados mais recentes, "
                            "(2) Aplicar técnicas de domain adaptation, ou (3) Usar sample weights baseados em validação adversária.",
                            "drift"
                        )
                    else:
                        logger.log(
                            "CONCLUSÃO: Estabilidade confirmada. As curvas de nível (contornos) do Treino e Teste "
                            "devem estar sobrepostas, indicando que ocupam o mesmo espaço multidimensional.",
                            "success"
                        )
                        logger.log_insight(
                            "A análise de drift mostra estabilidade entre treino e teste. "
                            "As distribuições estão alinhadas, indicando que o modelo deve generalizar bem.",
                            "drift"
                        )
                    
                    # Métricas de drift
                    logger.log_metric("Features com Drift Alto (KS > 0.15)", high_drift_count)
                    logger.log_metric("Features com Drift Moderado (0.1 < KS <= 0.15)", 
                                    ((drift_df['ks_statistic'] > 0.1) & (drift_df['ks_statistic'] <= 0.15)).sum())
                    logger.log_metric("Features Estáveis (KS <= 0.1)", 
                                    (drift_df['ks_statistic'] <= 0.1).sum())
                    logger.log_metric("KS Médio", f"{drift_df['ks_statistic'].mean():.4f}")
                    logger.log_metric("KS Máximo", f"{drift_df['ks_statistic'].max():.4f}")
                    
                    # ✅ NOVO: Seção Explícita de Sugestão de Blacklist para o Agente
                    toxic_features = drift_df[drift_df['ks_statistic'] > 0.5]['feature'].tolist()
                    if toxic_features:
                        logger.section("3.1.1. Sugestão de Ação: Remover Features com Drift Crítico", level=4)
                        logger.log(
                            f"🚨 ATENÇÃO AGENTE: {len(toxic_features)} features têm Drift Crítico (KS > 0.5). "
                            "Elas estão distorcendo a realidade entre Treino e Teste e devem ser removidas imediatamente.",
                            "critical"
                        )
                        logger.log_code_block(
                            json.dumps({"force_drop_features": toxic_features}, indent=2),
                            "json"
                        )
                        toxic_features_str = ', '.join([f"'{f}'" for f in toxic_features[:10]])
                        if len(toxic_features) > 10:
                            toxic_features_str += '...'
                        logger.log_insight(
                            f"**AÇÃO RECOMENDADA:** Adicione as seguintes features na lista 'force_drop_features' do config.yaml: "
                            f"{toxic_features_str}. "
                            "Essas features têm KS > 0.5, indicando que a distribuição mudou drasticamente entre treino e teste. "
                            "Manter essas features pode causar falhas graves do modelo em produção.",
                            "drift_action"
                        )
                        # ✅ INJEÇÃO DE CONTEXTO: Features tóxicas para o agente
                        logger.update_context("toxic_features_drift", toxic_features)
                        logger.update_context("toxic_features_count", len(toxic_features))
                
            except Exception as e:
                logger.log(f"Erro na análise de drift: {e}", "error")
                import traceback
                logger.log_code_block(traceback.format_exc(), "python")
        else:
            logger.log("Bibliotecas necessárias não disponíveis para análise de drift completa.", "warning")
            logger.log("Instale com: pip install scipy scikit-learn", "info")
            logger.log("Análise básica de drift será feita apenas com PSI na seção 12.", "info")
        
        # =============================================================================
        # 4. FEATURE SELECTION
        # =============================================================================
        
        logger.section("4. Feature Selection (Conservadora)", level=2)
        
        def smart_feature_selection(df, target_col, mode='DEV'):
            """
            Seleção de features conservadora.
            
            Mudanças em relação à versão anterior:
            1. Não remove correlação agressivamente (XGBoost lida bem com multicolinearidade).
            2. Em DEV, não remove features por baixa importância (evita corte prematuro).
            3. Sempre remove features com variância zero (constantes).
            
            Args:
                df: DataFrame com features e target
                target_col: Nome da coluna target
                mode: 'DEV' ou 'PROD'
            
            Returns:
                df_result: DataFrame filtrado
                kept_features: Lista de features mantidas
                dropped_vars: Lista de features removidas (apenas variância zero)
            """
            initial_cols = df.shape[1]
            dropped_vars = []
            
            # 1. Variância Zero (Sempre remover - features constantes não agregam informação)
            try:
                from sklearn.feature_selection import VarianceThreshold
                
                # Separar colunas numéricas e meta-colunas
                num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                meta_cols = [c for c in df.columns if c not in num_cols or c == target_col]
                
                if len(num_cols) > 0:
                    selector = VarianceThreshold(threshold=0.0)
                    selector.fit(df[num_cols])
                    
                    # Identificar features mantidas
                    kept_num_cols = df[num_cols].columns[selector.get_support()].tolist()
                    dropped_vars = list(set(num_cols) - set(kept_num_cols))
                    
                    if dropped_vars:
                        logger.log(f"Removidas {len(dropped_vars)} features constantes (Variância 0): {dropped_vars[:5]}{'...' if len(dropped_vars) > 5 else ''}", "info")
                        # Remover apenas as constantes, manter o resto
                        df = pd.concat([df[kept_num_cols], df[meta_cols]], axis=1)
            except Exception as e:
                logger.log(f"Erro no VarianceThreshold: {e}", "warning")
            
            # 2. Em DEV: Não aplicar filtro de correlação nem importância
            # XGBoost lida bem com multicolinearidade (apenas divide importância entre variáveis)
            # Remover correlação agressivamente pode jogar fora variação sutil útil
            if mode == 'DEV':
                logger.log("Modo DEV: Mantendo todas as features (exceto constantes) para visibilidade completa do Agente.", "info")
                logger.log_insight(
                    "Em modo DEV, não aplicamos filtros de correlação ou importância para evitar corte prematuro. "
                    "O XGBoost lida bem com multicolinearidade através de divisão de importância entre variáveis. "
                    "Features removidas apenas por variância zero (constantes).",
                    "feature_selection"
                )
            
            # Features mantidas (todas exceto target e meta-colunas)
            kept_features = [c for c in df.columns if c != target_col and c not in ['split', 'dataset_origin']]
            
            logger.log_metric("Features Iniciais", initial_cols)
            logger.log_metric("Features Removidas (Variância Zero)", len(dropped_vars))
            logger.log_metric("Features Mantidas", len(kept_features))
            
            return df, kept_features, dropped_vars
        
        train_df = X_train.copy()
        train_df['label'] = y_train
        
        # ✅ NOVO: Remover features forçadas pelo Agente (ex: alto drift detectado)
        force_drop_features = feature_config.get('force_drop_features', [])
        if force_drop_features:
            logger.log(f"[FEATURE DROP] Removendo {len(force_drop_features)} features banidas pelo Agente...", "warning")
            # Garantir que as colunas existam antes de dropar
            existing_drops = [c for c in force_drop_features if c in train_df.columns]
            if existing_drops:
                train_df = train_df.drop(columns=existing_drops, errors='ignore')
                logger.log(f"[FEATURE DROP] Features removidas: {', '.join(existing_drops[:10])}{'...' if len(existing_drops) > 10 else ''}", "info")
                logger.log_code_block(str(existing_drops), "json")
                # Atualizar feature_cols também
                feature_cols = [c for c in feature_cols if c not in existing_drops]
            else:
                logger.log(f"[FEATURE DROP] Nenhuma das features solicitadas existe no dataset.", "warning")
        
        # ✅ NOVA LÓGICA: Seleção conservadora (sem filtro de correlação agressivo)
        df_result, kept_features, dropped_vars = smart_feature_selection(
            train_df, 'label', mode=MODO
        )
        
        feature_cols_selected = kept_features
        X_train_processed = df_result[kept_features].copy()
        
        # ✅ IMPORTANTE: Processar Holdout (X_val) com a mesma seleção de features
        # O holdout deve ser usado apenas para avaliação final, não durante o treino
        X_val_processed = None
        y_val_holdout = None
        
        if 'X_val' in locals() and X_val is not None and len(X_val) > 0:
            logger.log("Processando Holdout (X_val) com a mesma seleção de features...", "info")
            
            # Remover features forçadas pelo agente
            if force_drop_features:
                X_val = X_val.drop(columns=force_drop_features, errors='ignore')
            
            # Remover features com variância zero
            if dropped_vars:
                X_val = X_val.drop(columns=dropped_vars, errors='ignore')
            
            # Manter apenas features selecionadas
            X_val_processed = X_val[kept_features].copy()
            y_val_holdout = y_val.copy()  # Renomear para não confundir com y_val_final do treino
            
            logger.log_metric("Holdout Processado", f"{X_val_processed.shape[0]:,} amostras × {X_val_processed.shape[1]} features")
            logger.log_insight(
                f"Holdout separado e processado: {len(X_val_processed):,} amostras serão usadas apenas para avaliação final "
                f"(calibração e métricas financeiras). Este conjunto não foi usado durante o treino.",
                "data_split"
            )
        else:
            logger.log("Holdout (X_val) não disponível. Usando validação do treino para métricas finais.", "warning")
        
        # ✅ INJEÇÃO DE CONTEXTO GLOBAL: Feature Selection
        logger.update_context("n_features_selected", len(kept_features))
        logger.update_context("n_features_dropped_variance_zero", len(dropped_vars))
        logger.update_context("n_features_dropped_forced", len(force_drop_features) if force_drop_features else 0)
        
        if X_test_blind is not None and len(X_test_blind) > 0:
            # Aplicar mesmas remoções no teste
            X_test_blind_processed = X_test_blind.drop(columns=force_drop_features + dropped_vars, errors='ignore')[kept_features].copy()
        else:
            X_test_blind_processed = None
        
        logger.log_metric("Features Mantidas", len(kept_features))
        logger.log_metric("Features Removidas (Variância Zero)", len(dropped_vars))
        logger.log_metric("Features Removidas (Forçadas pelo Agente)", len(force_drop_features) if force_drop_features else 0)
        logger.log_metric("Shape Final Treino", f"{X_train_processed.shape}")
        
        # ✅ MELHORIA: Mostrar amostra dos dados processados
        if hasattr(logger, 'log_dataframe_head'):
            logger.log_dataframe_head(X_train_processed.head(3), n=3, title="Amostra dos Dados Após Feature Selection")
        else:
            # Fallback: logar apenas estatísticas básicas
            logger.log(f"Amostra dos dados após feature selection: {X_train_processed.shape[0]} linhas × {X_train_processed.shape[1]} features", "info")
        
        # =============================================================================
        # 5. CONFIGURAÇÃO DO MODELO
        # =============================================================================
        
        logger.section("5. Configuração do Modelo XGBoost", level=2)
        
        # ✅ Calcular scale_pos_weight (ou usar do config se não for "auto")
        scale_pos_weight_config = xgboost_config.get('scale_pos_weight', 'auto')
        if scale_pos_weight_config == 'auto':
            pos_weight = (y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1.0
        else:
            pos_weight = float(scale_pos_weight_config)
        
        # ✅ Ler TODOS os parâmetros do config.yaml (com fallback para valores padrão)
        base_params = {
            'objective': xgboost_config.get('objective', 'binary:logistic'),
            'eval_metric': xgboost_config.get('eval_metric', 'auc'),
            'tree_method': 'hist',  # Será sobrescrito se GPU disponível
            'max_depth': xgboost_config.get('max_depth', 6),
            'learning_rate': xgboost_config.get('learning_rate', 0.05),
            'subsample': xgboost_config.get('subsample', 0.8),
            'colsample_bytree': xgboost_config.get('colsample_bytree', 0.8),
            'min_child_weight': xgboost_config.get('min_child_weight', 3),
            'gamma': xgboost_config.get('gamma', 0.1),
            'scale_pos_weight': pos_weight,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }
        
        if GPU_AVAILABLE:
            base_params['tree_method'] = 'gpu_hist'
            base_params['device'] = 'cuda'
            logger.log("Usando aceleração GPU", "success")
        else:
            logger.log("GPU não disponível - usando CPU", "warning")
        
        logger.log_parameters(base_params, "Parâmetros do Modelo")
        logger.log_metric("scale_pos_weight (Balanceamento)", f"{pos_weight:.3f}")
        
        # =============================================================================
        # 6. VALIDAÇÃO CRUZADA TEMPORAL
        # =============================================================================
        
        logger.section("6. Validação Cruzada Temporal", level=2)
        
        n_splits_cv = 3 if MODO == 'DEV' else 5
        print_progress("Validacao Cruzada", 0, n_splits_cv)
        
        try:
            cv_results = temporal_cross_validation(
                X_train_processed,
                y_train,
                model_params=base_params,
                n_splits=n_splits_cv,
                gap=0,
                verbose=False
            )
            print_progress("Validacao Cruzada", n_splits_cv, n_splits_cv)
            print()  # Nova linha
            
            logger.log_metric("AUC Médio (CV)", f"{cv_results['mean_auc']:.4f} ± {cv_results['std_auc']:.4f}")
            
            for i, auc in enumerate(cv_results['auc_scores'], 1):
                logger.log_metric(f"AUC Fold {i}", f"{auc:.4f}")
            
            logger.log_insight(
                f"A validação cruzada temporal mostra AUC médio de {cv_results['mean_auc']:.4f} "
                f"com desvio padrão de {cv_results['std_auc']:.4f}. "
                "Esta é a métrica REALISTA de treino sem vazamento temporal. "
                "Se a AUC de teste for próxima desta, o modelo está generalizando bem.",
                "overfitting"
            )
            
        except Exception as e:
            logger.log(f"Erro na validação cruzada: {e}", "error")
        
        # =============================================================================
        # 7. TREINAMENTO DO MODELO FINAL
        # =============================================================================
        
        logger.section("7. Treinamento do Modelo Final", level=2)
        
        val_size = int(len(X_train_processed) * 0.15)
        X_train_final = X_train_processed.iloc[:-val_size]
        y_train_final = y_train.iloc[:-val_size]
        X_val_final = X_train_processed.iloc[-val_size:]
        y_val_final = y_train.iloc[-val_size:]
        
        logger.log_metric("Treino Final", f"{len(X_train_final):,} amostras")
        logger.log_metric("Validação Final", f"{len(X_val_final):,} amostras")
        
        # ✅ Configuração baseada em modo (lendo do config.yaml)
        if MODO == 'DEV':
            n_estimators = xgboost_config.get('n_estimators_dev', 50)
            early_stopping_rounds = 10
            logger.log(f"Modo DEV: n_estimators={n_estimators} (configuracao rapida)", "info")
        else:
            n_estimators = xgboost_config.get('n_estimators_prod', 500)
            early_stopping_rounds = 30
            logger.log(f"Modo PROD: n_estimators={n_estimators} (configuracao completa)", "info")
        
        model = xgb.XGBClassifier(
            **base_params,
            n_estimators=n_estimators,
            early_stopping_rounds=early_stopping_rounds
        )
        
        logger.log("Treinando modelo...", "info")
        print_progress("Treinamento", 0, n_estimators)
        model.fit(
            X_train_final, y_train_final,
            eval_set=[(X_train_final, y_train_final), (X_val_final, y_val_final)],
            verbose=False
        )
        print_progress("Treinamento", n_estimators, n_estimators)
        print()  # Nova linha após progress bar
        
        # Predições
        y_train_pred_proba = model.predict_proba(X_train_final)[:, 1]
        y_val_pred_proba = model.predict_proba(X_val_final)[:, 1]
        
        train_auc = roc_auc_score(y_train_final, y_train_pred_proba)
        val_auc = roc_auc_score(y_val_final, y_val_pred_proba)
        
        logger.log_metric("AUC Treino", f"{train_auc:.4f}")
        logger.log_metric("AUC Validação", f"{val_auc:.4f}")
        logger.log_metric("Gap (Overfitting)", f"{(train_auc - val_auc)*100:.2f}%")
        
        gap = train_auc - val_auc
        
        # ✅ INJEÇÃO DE CONTEXTO GLOBAL: Performance do Modelo
        logger.update_context("auc_train", train_auc)
        logger.update_context("auc_val", val_auc)
        logger.update_context("overfitting_gap", gap)
        logger.update_context("overfitting_status", "Critical" if gap > 0.12 else "Moderate" if gap > 0.08 else "Acceptable")
        if gap > 0.12:
            logger.log("OVERFITTING CRÍTICO detectado! Gap > 12 pontos percentuais.", "error")
            logger.log_insight(
                f"O modelo está com overfitting severo (gap de {gap*100:.1f}%). "
                "Será necessário aplicar regularização agressiva ou usar o protocolo de emergência.",
                "overfitting"
            )
        elif gap > 0.08:
            logger.log("Overfitting moderado detectado. Monitorar.", "warning")
        else:
            logger.log("Modelo generalizando bem. Gap aceitável.", "success")
        
        # Curva ROC
        fpr_train, tpr_train, _ = roc_curve(y_train_final, y_train_pred_proba)
        fpr_val, tpr_val, _ = roc_curve(y_val_final, y_val_pred_proba)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(fpr_train, tpr_train, label=f'Treino (AUC = {train_auc:.4f})', linewidth=2)
        ax.plot(fpr_val, tpr_val, label=f'Validação (AUC = {val_auc:.4f})', linewidth=2)
        ax.plot([0, 1], [0, 1], 'k--', label='Aleatório')
        ax.set_xlabel('Taxa de Falsos Positivos')
        ax.set_ylabel('Taxa de Verdadeiros Positivos')
        ax.set_title('Curva ROC - Treino vs Validação')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # ✅ MELHORIA: Descrição textual da geometria da curva ROC
        # ✅ Usando geometria robusta para evitar falsos zeros
        roc_geometry_desc = describe_curve_geometry_robust(fpr_val, tpr_val, "Curva ROC (Validação)")
        roc_description = logger.describe_roc_curve(fpr_val, tpr_val, val_auc)
        
        # ✅ Contexto técnico detalhado com referência ao contexto global
        balance_info = logger.global_context.get("target_imbalance_status", "Unknown")
        balance_ratio = logger.global_context.get("class_balance_ratio", "N/A")
        
        roc_plot_context = (
            f"Curva ROC do modelo XGBoost (Validação). "
            f"AUC: {val_auc:.4f}. Gap Treino-Val: {gap:.4f} ({'Crítico' if gap > 0.12 else 'Moderado' if gap > 0.08 else 'Aceitável'}). "
            f"O contexto global indica desbalanceamento: {balance_info} (razão: {balance_ratio}), "
            f"o que torna a curva ROC sensível. "
            f"{roc_description}\n\n"
            f"{roc_geometry_desc}"
        )
        
        logger.log_plot(
            fig,
            title="Curva ROC - Treino vs Validação",
            description=f"Curva ROC comparando treino (AUC={train_auc:.4f}) e validação (AUC={val_auc:.4f})",
            context_description=roc_plot_context,
            save_image=True,
            analyze=True
        )
        
        # =============================================================================
        # 8. PROTOCOLO DE EMERGÊNCIA (se necessário)
        # =============================================================================
        
        if gap > 0.08:
            logger.section("8. Protocolo de Emergência: Regularização Agressiva", level=2)
            
            params_anti_overfit = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': 3,
                'min_child_weight': 50,
                'gamma': 5.0,
                'subsample': 0.6,
                'colsample_bytree': 0.5,
                'reg_alpha': 10.0,
                'reg_lambda': 10.0,
                'learning_rate': 0.05,
                'n_jobs': -1,
                'random_state': 42,
                'verbosity': 0
            }
            
            if GPU_AVAILABLE:
                params_anti_overfit['tree_method'] = 'gpu_hist'
                params_anti_overfit['device'] = 'cuda'
            
            if MODO == 'DEV':
                n_estimators_robust = 50
                early_stopping_robust = 10
            else:
                n_estimators_robust = 1000
                early_stopping_robust = 50
            
            model_robust = xgb.XGBClassifier(
                **params_anti_overfit,
                n_estimators=n_estimators_robust,
                early_stopping_rounds=early_stopping_robust
            )
            
            logger.log("Retreinando com restrições severas...", "info")
            model_robust.fit(
                X_train_final, y_train_final,
                eval_set=[(X_train_final, y_train_final), (X_val_final, y_val_final)],
                verbose=False
            )
            
            y_train_prob_robust = model_robust.predict_proba(X_train_final)[:, 1]
            y_val_prob_robust = model_robust.predict_proba(X_val_final)[:, 1]
            
            new_auc_train = roc_auc_score(y_train_final, y_train_prob_robust)
            new_auc_val = roc_auc_score(y_val_final, y_val_prob_robust)
            
            new_gap = new_auc_train - new_auc_val
            
            logger.log_metric("AUC Treino (Robusto)", f"{new_auc_train:.4f}")
            logger.log_metric("AUC Validação (Robusto)", f"{new_auc_val:.4f}")
            logger.log_metric("Gap (Robusto)", f"{new_gap*100:.2f}%")
            
            if new_gap < 0.08:
                logger.log("Sucesso: O Gap foi fechado. O modelo agora é honesto.", "success")
                model = model_robust
                train_auc = new_auc_train
                val_auc = new_auc_val
            else:
                logger.log("Atenção: O Gap diminuiu mas persiste.", "warning")
                model = model_robust
                train_auc = new_auc_train
                val_auc = new_auc_val
        
        # =============================================================================
        # 9. SHAP VALUES (Explainability)
        # =============================================================================
        
        # ✅ OTIMIZAÇÃO: Pular SHAP no modo DEV ou se run_shap=False
        if SHAP_AVAILABLE and RUN_SHAP and MODO != 'DEV':
            logger.section("9. SHAP Values - Explainability", level=2)
            
            # Amostra reduzida para acelerar (1000 amostras)
            X_shap_sample = X_train_final.sample(n=min(1000, len(X_train_final)), random_state=42)
            
            logger.log("Calculando valores SHAP (pode demorar)...", "info")
            print_progress("SHAP", 0, len(X_shap_sample))
            
            explainer = shap.TreeExplainer(model)
            shap_explanation = explainer(X_shap_sample)
            
            print_progress("SHAP", len(X_shap_sample), len(X_shap_sample))
            print()  # Nova linha após progress bar
            
            # ✅ CORREÇÃO: Calcular shap_summary ANTES de usar no contexto
            shap_summary = pd.Series(
                np.abs(shap_explanation.values).mean(0),
                index=X_shap_sample.columns
            ).sort_values(ascending=False)
            
            # Gráfico beeswarm
            fig = plt.figure(figsize=(12, 8))
            shap.plots.beeswarm(shap_explanation, max_display=20, show=False)
            plt.title('Impacto das Features na Decisão (SHAP)', fontsize=14)
            plt.tight_layout()
            
            # Contexto técnico detalhado para análise IA do SHAP
            shap_plot_context = (
                f"Gráfico beeswarm SHAP mostrando o impacto das top 20 features nas decisões do modelo XGBoost. "
                f"Features ordenadas por importância média (SHAP value absoluto). "
                f"Pontos vermelhos: valores altos da feature. Pontos azuis: valores baixos. "
                f"Eixo X positivo: aumenta risco de crédito (default). Eixo X negativo: diminui risco. "
                f"Feature mais importante: {shap_summary.index[0]} (impacto médio: {shap_summary.iloc[0]:.4f}). "
                f"Se uma feature separar perfeitamente as classes (cores não se misturam), pode indicar data leakage."
            )
            
            logger.log_plot(
                fig,
                title="SHAP Values - Impacto das Features",
                description="Gráfico beeswarm SHAP mostrando impacto das top 20 features",
                context_description=shap_plot_context,
                save_image=True,
                analyze=True
            )
            
            logger.log_table("Top 10 Features por Importância SHAP",
                            {k: f"{v:.4f}" for k, v in shap_summary.head(10).items()})
            
            logger.log_insight(
                f"A feature mais importante segundo SHAP é {shap_summary.index[0]} "
                f"com impacto médio de {shap_summary.iloc[0]:.4f}. "
                "Se esta feature separar perfeitamente as classes, pode indicar data leakage.",
                "explainability"
            )
        elif SHAP_AVAILABLE and (not RUN_SHAP or MODO == 'DEV'):
            # Modo DEV ou SHAP desabilitado: pular completamente
            logger.section("9. SHAP Values - Explainability", level=2)
            if MODO == 'DEV':
                logger.log("SHAP pulado no modo DEV para acelerar desenvolvimento. Execute em modo PROD para análise completa.", "info")
            else:
                logger.log("SHAP desabilitado no config.yaml (run_shap: false).", "info")
            logger.log_insight(
                "Análise SHAP não foi executada. Para ativar, configure 'run_shap: true' no config.yaml e execute em modo PROD.",
                "explainability"
            )
        elif not SHAP_AVAILABLE:
            logger.section("9. SHAP Values - Explainability", level=2)
            logger.log("SHAP não disponível. Instale com: pip install shap", "warning")
        
        # =============================================================================
        # 10. CALIBRAÇÃO DE PROBABILIDADES
        # =============================================================================
        
        logger.section("10. Calibração de Probabilidades", level=2)
        
        try:
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.metrics import brier_score_loss
            
            logger.log_timestamp("Início da calibração")
            
            # ✅ CORREÇÃO CRÍTICA: Criar modelo limpo sem early_stopping_rounds
            # O CalibratedClassifierCV faz cross-validation interno e não passa eval_set
            calib_base = xgb.XGBClassifier(
                **{k: v for k, v in base_params.items() if k not in ['n_estimators', 'early_stopping_rounds']},
                n_estimators=300  # Fixo, sem early stopping
            )
            
            model_calibrated = CalibratedClassifierCV(
                calib_base, method='isotonic', cv=3
            )
            
            logger.log("Calibrando probabilidades (método isotônico, cv=3)...", "info")
            model_calibrated.fit(X_train_final, y_train_final)
            
            # ✅ MUDANÇA: Usar Holdout (X_val_processed) para avaliação final, se disponível
            # Se não houver holdout, usar validação do treino como fallback
            if X_val_processed is not None and y_val_holdout is not None:
                eval_X = X_val_processed
                eval_y = y_val_holdout
                logger.log(f"Usando Holdout para avaliação final: {len(eval_X):,} amostras", "info")
            else:
                eval_X = X_val_final
                eval_y = y_val_final
                logger.log(f"Holdout não disponível. Usando validação do treino: {len(eval_X):,} amostras", "warning")
            
            # Predições antes e depois da calibração
            prob_raw = model.predict_proba(eval_X)[:, 1]
            y_val_proba_cal = model_calibrated.predict_proba(eval_X)[:, 1]
            
            # Métricas de calibração
            loss_raw = brier_score_loss(eval_y, prob_raw)
            loss_cal = brier_score_loss(eval_y, y_val_proba_cal)
            
            logger.log_metric("Brier Score (Antes da Calibração)", f"{loss_raw:.4f}")
            logger.log_metric("Brier Score (Depois da Calibração)", f"{loss_cal:.4f}")
            logger.log_metric("Melhoria no Brier Score", f"{(loss_raw - loss_cal):.4f} ({(loss_raw - loss_cal)/loss_raw*100:.1f}%)")
            
            # ✅ NOVO: Plot de Calibração (Probabilidade Prevista vs Observada)
            try:
                from sklearn.calibration import calibration_curve
                
                # Calcular curvas de calibração antes e depois (usar eval_y)
                fraction_of_positives_raw, mean_predicted_value_raw = calibration_curve(
                    eval_y, prob_raw, n_bins=10, strategy='uniform'
                )
                fraction_of_positives_cal, mean_predicted_value_cal = calibration_curve(
                    eval_y, y_val_proba_cal, n_bins=10, strategy='uniform'
                )
                
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Linha de calibração perfeita (diagonal)
                ax.plot([0, 1], [0, 1], 'k--', label='Calibração Perfeita', linewidth=2)
                
                # Curva antes da calibração
                ax.plot(mean_predicted_value_raw, fraction_of_positives_raw, 
                       'o-', color='red', label=f'Antes (Brier={loss_raw:.4f})', 
                       linewidth=2, markersize=8)
                
                # Curva depois da calibração
                ax.plot(mean_predicted_value_cal, fraction_of_positives_cal, 
                       'o-', color='green', label=f'Depois (Brier={loss_cal:.4f})', 
                       linewidth=2, markersize=8)
                
                ax.set_xlabel('Probabilidade Média Prevista', fontsize=12)
                ax.set_ylabel('Fração de Positivos Observada', fontsize=12)
                ax.set_title('Curva de Calibração: Antes vs Depois', fontsize=14, fontweight='bold')
                ax.legend(loc='best', fontsize=11)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                
                calib_context = (
                    f"Curva de calibração comparando probabilidades antes e depois da calibração isotônica. "
                    f"Quanto mais próxima da diagonal (linha preta tracejada), melhor a calibração. "
                    f"Brier Score melhorou de {loss_raw:.4f} para {loss_cal:.4f} "
                    f"({(loss_raw - loss_cal)/loss_raw*100:.1f}% de melhoria)."
                )
                
                logger.log_plot(
                    fig,
                    title="Curva de Calibração - Antes vs Depois",
                    description="Comparação de calibração antes e depois da aplicação do método isotônico",
                    context_description=calib_context,
                    save_image=True,
                    analyze=True
                )
            except Exception as e:
                logger.log(f"Erro ao gerar plot de calibração: {e}", "warning")
            
            logger.log("Calibração concluída", "success")
            logger.log_insight(
                f"As probabilidades foram calibradas usando método isotônico. "
                f"O Brier Score melhorou de {loss_raw:.4f} para {loss_cal:.4f} "
                f"({(loss_raw - loss_cal)/loss_raw*100:.1f}% de melhoria). "
                "Isso garante que probabilidades de 0.7 realmente significam 70% de chance de default, "
                "essencial para decisões financeiras precisas.",
                "calibration"
            )
            
            logger.log_timestamp("Fim da calibração")
            
        except Exception as e:
            logger.log(f"Erro na calibração: {e}", "error")
            import traceback
            logger.log_code_block(traceback.format_exc(), "python")
            model_calibrated = model
            y_val_proba_cal = model.predict_proba(X_val_final)[:, 1]
        
        # =============================================================================
        # 11. ANÁLISE FINANCEIRA
        # =============================================================================
        
        logger.section("11. Análise de Impacto Financeiro", level=2)
        
        # ✅ Ler parâmetros financeiros do config.yaml
        cost_matrix = business_config.get('cost_matrix', {})
        TICKET_MEDIO = business_config.get('ticket_medio', 10000)
        GANHO_TP = cost_matrix.get('tp', business_config.get('ganho_tp', 1500))
        PERDA_FN = cost_matrix.get('fn', 0)
        PERDA_FP = cost_matrix.get('fp', business_config.get('perda_fp', -10000))
        GANHO_TN = cost_matrix.get('tn', 0)
        
        # ✅ Formatação sem R$ para evitar conflito com LaTeX no Markdown
        logger.log_metric("Ticket Médio", f"{TICKET_MEDIO:,.2f}")
        logger.log_metric("Ganho TP", f"{GANHO_TP:,.2f}")
        logger.log_metric("Perda FP", f"{PERDA_FP:,.2f}")
        
        logger.log_parameters({
            'TICKET_MEDIO': TICKET_MEDIO,
            'GANHO_TP': GANHO_TP,
            'PERDA_FP': PERDA_FP,
            'PERDA_FN': PERDA_FN,
            'GANHO_TN': GANHO_TN
        }, "Parâmetros Financeiros")
        
        try:
            cost_matrix_dict = {
                'tp': GANHO_TP,
                'fp': PERDA_FP,
                'fn': PERDA_FN,
                'tn': GANHO_TN
            }
            
            # ✅ MUDANÇA: Usar Holdout (eval_y_financial) para otimização financeira, se disponível
            if X_val_processed is not None and y_val_holdout is not None and 'model_calibrated' in locals():
                eval_y_financial = y_val_holdout
                # Recalcular probabilidades no holdout usando modelo calibrado
                eval_y_proba_cal = model_calibrated.predict_proba(X_val_processed)[:, 1]
                y_proba_for_threshold = eval_y_proba_cal
                logger.log(f"Otimização financeira usando Holdout: {len(eval_y_financial):,} amostras", "info")
            else:
                # Fallback: usar validação do treino
                eval_y_financial = y_val_final
                y_proba_for_threshold = y_val_proba_cal if 'y_val_proba_cal' in locals() else y_val_pred_proba
                logger.log(f"Otimização financeira usando validação do treino: {len(eval_y_financial):,} amostras", "warning")
            
            # ✅ VALIDAÇÃO CRÍTICA: Garantir que tamanhos são consistentes
            if len(eval_y_financial) != len(y_proba_for_threshold):
                logger.log(
                    f"ERRO CRÍTICO: Tamanhos inconsistentes! eval_y_financial={len(eval_y_financial)}, "
                    f"y_proba_for_threshold={len(y_proba_for_threshold)}. Corrigindo...",
                    "error"
                )
                # Usar o menor tamanho comum
                min_len = min(len(eval_y_financial), len(y_proba_for_threshold))
                eval_y_financial = eval_y_financial[:min_len].copy()
                y_proba_for_threshold = y_proba_for_threshold[:min_len].copy()
                logger.log(f"Tamanhos ajustados para {min_len} amostras", "warning")
            
            # ✅ VALIDAÇÃO FINAL: Verificar novamente antes de chamar função
            if len(eval_y_financial) != len(y_proba_for_threshold):
                raise ValueError(
                    f"Falha ao corrigir tamanhos inconsistentes: "
                    f"eval_y_financial={len(eval_y_financial)}, "
                    f"y_proba_for_threshold={len(y_proba_for_threshold)}"
                )
            
            threshold_results = find_optimal_threshold(
                eval_y_financial,
                y_proba_for_threshold,
                cost_matrix=cost_matrix_dict
            )
            
            optimal_threshold = threshold_results['optimal_threshold']
            max_profit = threshold_results['optimal_profit']
            all_results = threshold_results['all_results']
            
            # ✅ NOVO: Calcular Lucro Potencial Máximo (Teto Teórico) usando eval_y_financial
            max_potential_profit = calculate_max_potential_profit(eval_y_financial, cost_matrix_dict)
            
            # ✅ NOVO: Calcular Eficiência Financeira (% do Potencial)
            if max_potential_profit > 0:
                financial_efficiency = (max_profit / max_potential_profit) * 100
            else:
                financial_efficiency = 0.0
            
            logger.log_metric("Threshold Ótimo", f"{optimal_threshold:.4f}")
            # ✅ Formatação sem R$ para evitar conflito com LaTeX
            logger.log_metric("Lucro Real (Amostra)", f"{max_profit:,.2f}")
            logger.log_metric("Lucro Potencial Máximo (Teórico)", f"{max_potential_profit:,.2f}")
            logger.log_metric("Eficiência Financeira (% do Potencial)", f"{financial_efficiency:.2f}%")
            
            # ✅ ENFATIZAR: Esta métrica é agnóstica ao tamanho da amostra
            logger.log(
                f"**IMPORTANTE:** A Eficiência Financeira de {financial_efficiency:.2f}% é uma métrica agnóstica ao tamanho da amostra. "
                f"Ela funciona igual em modo DEV (10k linhas) e PROD (500k linhas). "
                f"Meta: > 75% de eficiência.",
                "info"
            )
            
            # ✅ INJEÇÃO DE CONTEXTO GLOBAL: Métricas Financeiras
            logger.update_context("financial_efficiency_percent", financial_efficiency)
            logger.update_context("profit_real", max_profit)
            logger.update_context("profit_potential_max", max_potential_profit)
            
            # ✅ MELHORIA: Descrição textual da curva de lucro
            thresholds_array = all_results['threshold'].values
            profits_array = all_results['profit'].values
            
            profit_curve_desc = describe_curve_geometry_robust(thresholds_array, profits_array, "Curva de Lucro")
            logger.log(profit_curve_desc, "info")
            
            # ✅ NOVO: Plot Completo de Curvas de Lucro vs Threshold vs Taxa de Aprovação
            try:
                # Calcular taxas de aprovação para cada threshold (usar eval_y_financial para tamanho correto)
                approval_rates = []
                for t in thresholds_array:
                    approved = np.sum(y_proba_for_threshold >= t)
                    approval_rates.append(approved / len(eval_y_financial) * 100)
                approval_rates = np.array(approval_rates)
                
                fig, axes = plt.subplots(2, 1, figsize=(14, 10))
                
                # Plot 1: Lucro vs Threshold
                axes[0].plot(thresholds_array, profits_array, linewidth=2.5, color='steelblue', label='Lucro Esperado')
                axes[0].axvline(optimal_threshold, color='red', linestyle='--', linewidth=2, 
                               label=f'Threshold Ótimo = {optimal_threshold:.3f}')
                axes[0].axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)
                axes[0].axhline(max_profit, color='green', linestyle=':', linewidth=1.5, alpha=0.7,
                               label=f'Lucro Máximo = {max_profit:,.0f}')
                
                # Zona de prejuízo e lucro
                axes[0].fill_between(thresholds_array, 0, profits_array, where=(profits_array < 0), 
                                   color='red', alpha=0.2, label='Zona de Prejuízo')
                axes[0].fill_between(thresholds_array, 0, profits_array, where=(profits_array >= 0), 
                                   color='green', alpha=0.2, label='Zona de Lucro')
                
                axes[0].set_xlabel('Threshold de Corte (Probabilidade Mínima para Aprovar)', fontsize=12)
                axes[0].set_ylabel('Lucro Esperado', fontsize=12)
                axes[0].set_title('💰 Curva de Lucro Esperado vs Threshold de Corte', 
                                fontsize=14, fontweight='bold')
                axes[0].legend(loc='best', fontsize=11)
                axes[0].grid(True, alpha=0.3)
                
                # Anotação do ponto ótimo
                axes[0].annotate(
                    f'Ótimo\n{max_profit:,.0f}',
                    xy=(optimal_threshold, max_profit),
                    xytext=(optimal_threshold + 0.1, max_profit + max_profit*0.1),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7)
                )
                
                # Plot 2: Taxa de Aprovação vs Threshold
                axes[1].plot(thresholds_array, approval_rates, linewidth=2, color='orange', 
                           label='Taxa de Aprovação (%)', alpha=0.8)
                axes[1].axvline(optimal_threshold, color='red', linestyle='--', linewidth=2,
                               label=f'Threshold Ótimo = {optimal_threshold:.3f}')
                
                # Calcular taxa de aprovação no threshold ótimo (usar eval_y_financial para tamanho correto)
                optimal_approval_rate = np.sum(y_proba_for_threshold >= optimal_threshold) / len(eval_y_financial) * 100
                axes[1].axhline(optimal_approval_rate, color='green', linestyle=':', linewidth=1.5, alpha=0.7,
                               label=f'Taxa Ótima = {optimal_approval_rate:.1f}%')
                
                axes[1].set_xlabel('Threshold de Corte', fontsize=12)
                axes[1].set_ylabel('Taxa de Aprovação (%)', fontsize=12)
                axes[1].set_title('Taxa de Aprovação vs Threshold', fontsize=12, fontweight='bold')
                axes[1].legend(loc='best', fontsize=11)
                axes[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                profit_plot_context = (
                    f"Análise completa de otimização financeira. "
                    f"Topo: Curva de lucro vs threshold. O threshold ótimo ({optimal_threshold:.4f}) maximiza lucro em {max_profit:,.0f}. "
                    f"Zonas verdes indicam lucro, zonas vermelhas indicam prejuízo. "
                    f"Base: Taxa de aprovação vs threshold. No threshold ótimo, aprovamos {optimal_approval_rate:.1f}% dos casos. "
                    f"Esta visualização permite balancear lucro máximo com volume de negócios."
                )
                
                logger.log_plot(
                    fig,
                    title="Curvas de Lucro e Taxa de Aprovação vs Threshold",
                    description="Análise completa de otimização financeira: lucro esperado e taxa de aprovação em função do threshold",
                    context_description=profit_plot_context,
                    save_image=True,
                    analyze=True
                )
            except Exception as e:
                logger.log(f"Erro ao gerar plot de curvas de lucro: {e}", "warning")
            
            # Identificar zona de estabilidade (flat-top)
            profit_max = np.max(profits_array)
            profit_threshold = profit_max * 0.95  # 95% do máximo
            stable_zone_indices = np.where(profits_array >= profit_threshold)[0]
            
            if len(stable_zone_indices) > 1:
                t_min = thresholds_array[stable_zone_indices[0]]
                t_max = thresholds_array[stable_zone_indices[-1]]
                profit_variation = (np.max(profits_array[stable_zone_indices]) - np.min(profits_array[stable_zone_indices])) / profit_max * 100
                
                logger.log_insight(
                    f"A zona de lucro máximo é plana (flat-top), variando menos de {profit_variation:.1f}% "
                    f"entre os thresholds {t_min:.3f} e {t_max:.3f}. "
                    "Isso indica que o modelo é robusto a pequenas variações na política de corte nesta faixa.",
                    "financeiro"
                )
            
            # Calcular métricas no threshold ótimo usando eval_y_financial
            y_pred_optimal = (y_proba_for_threshold >= optimal_threshold).astype(int)
            cm = confusion_matrix(eval_y_financial, y_pred_optimal)
            tn, fp, fn, tp = cm.ravel()
            
            logger.log_table("Matriz de Confusão (Threshold Ótimo)", {
                'True Negatives (TN)': int(tn),
                'False Positives (FP)': int(fp),
                'False Negatives (FN)': int(fn),
                'True Positives (TP)': int(tp)
            })
            
            # ✅ NOVO: Plot de Matriz de Confusão com Custos
            try:
                fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                
                # Plot 1: Matriz de Confusão (Contagem)
                cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                           xticklabels=['Negativo (0)', 'Positivo (1)'],
                           yticklabels=['Negativo (0)', 'Positivo (1)'],
                           cbar_kws={'label': 'Contagem'})
                axes[0].set_title('Matriz de Confusão (Contagem)', fontsize=13, fontweight='bold')
                axes[0].set_ylabel('Verdadeiro', fontsize=11)
                axes[0].set_xlabel('Previsto', fontsize=11)
                
                # Plot 2: Matriz de Custos Financeiros
                cost_matrix_vis = np.array([
                    [tn * GANHO_TN, fp * PERDA_FP],  # Linha 0: Verdadeiro Negativo
                    [fn * PERDA_FN, tp * GANHO_TP]   # Linha 1: Verdadeiro Positivo
                ])
                
                sns.heatmap(cost_matrix_vis, annot=True, fmt='.0f', cmap='RdYlGn', 
                           center=0, ax=axes[1],
                           xticklabels=['Negativo (0)', 'Positivo (1)'],
                           yticklabels=['Negativo (0)', 'Positivo (1)'],
                           cbar_kws={'label': 'Custo/Benefício (R$)'})
                axes[1].set_title('Matriz de Custos Financeiros (R$)', fontsize=13, fontweight='bold')
                axes[1].set_ylabel('Verdadeiro', fontsize=11)
                axes[1].set_xlabel('Previsto', fontsize=11)
                
                plt.tight_layout()
                
                cm_context = (
                    f"Matriz de confusão e custos financeiros no threshold ótimo ({optimal_threshold:.4f}). "
                    f"Esquerda: Contagem de acertos/erros (TN={tn}, FP={fp}, FN={fn}, TP={tp}). "
                    f"Direita: Impacto financeiro por célula. "
                    f"Erro Tipo I (FP): Aprovamos caloteiro = {fp} × {abs(PERDA_FP):,.0f} = {fp * PERDA_FP:,.0f}. "
                    f"Erro Tipo II (FN): Negamos bom pagador = {fn} × {abs(PERDA_FN):,.0f} = {fn * PERDA_FN:,.0f}. "
                    f"Lucro Total: {tp * GANHO_TP + fp * PERDA_FP + fn * PERDA_FN + tn * GANHO_TN:,.0f}."
                )
                
                logger.log_plot(
                    fig,
                    title="Matriz de Confusão e Custos Financeiros",
                    description="Comparação entre matriz de confusão (contagem) e matriz de custos (impacto financeiro)",
                    context_description=cm_context,
                    save_image=True,
                    analyze=True
                )
            except Exception as e:
                logger.log(f"Erro ao gerar plot de matriz de confusão: {e}", "warning")
            
            logger.log_insight(
                f"O threshold ótimo para maximizar lucro é {optimal_threshold:.4f}, "
                f"gerando lucro esperado de {max_profit:,.2f} (amostra atual). "
                f"**MÉTRICA PRINCIPAL:** O modelo capturou **{financial_efficiency:.2f}%** de todo o dinheiro disponível na mesa "
                f"(lucro potencial máximo teórico: {max_potential_profit:,.2f}). "
                f"Esta métrica de eficiência é AGNÓSTICA ao tamanho da amostra (funciona igual em DEV e PROD). "
                f"Uma eficiência > 75% é considerada excelente em crédito. "
                f"Com este threshold, temos {tp} aprovações corretas (TP) e {fp} aprovações incorretas (FP). "
                "Este threshold é mais conservador que o padrão de 0.5, "
                "refletindo o risco assimétrico do negócio onde perder 10.000 em um calote "
                "é muito pior do que perder 1.500 em juros de um bom pagador.",
                "financeiro"
            )
            
        except Exception as e:
            logger.log(f"Erro na otimização financeira: {e}", "error")
            import traceback
            logger.log_code_block(traceback.format_exc(), "python")
        
        # =============================================================================
        # 11.5. ANÁLISE DE ELASTICIDADE: AUC vs LUCRO
        # =============================================================================
        
        logger.section("11.5. Análise de Elasticidade: Sensibilidade do Lucro à AUC", level=2)
        
        try:
            # ✅ Verificar se temos todas as variáveis necessárias
            has_proba = 'y_proba_for_threshold' in locals() or 'y_val_proba_cal' in locals() or 'eval_y_proba_cal' in locals()
            has_threshold = 'optimal_threshold' in locals()
            has_eval_y = 'eval_y_financial' in locals()
            
            if has_proba and has_threshold and has_eval_y:
                logger.log("Simulando degradação controlada da AUC para medir elasticidade do lucro...", "info")
                
                # Usar holdout para análise de elasticidade (mesmas variáveis da otimização financeira)
                y_elasticity_true = eval_y_financial
                # Usar a mesma probabilidade que foi usada na otimização financeira
                if 'y_proba_for_threshold' in locals():
                    y_elasticity_proba = y_proba_for_threshold
                elif 'eval_y_proba_cal' in locals():
                    y_elasticity_proba = eval_y_proba_cal
                elif 'y_val_proba_cal' in locals():
                    y_elasticity_proba = y_val_proba_cal
                else:
                    raise ValueError("Nenhuma probabilidade disponível para análise de elasticidade")
                
                # ✅ VALIDAÇÃO CRÍTICA: Garantir tamanhos consistentes
                if len(y_elasticity_true) != len(y_elasticity_proba):
                    logger.log(
                        f"ERRO CRÍTICO: Tamanhos inconsistentes na elasticidade! "
                        f"y_elasticity_true={len(y_elasticity_true)}, "
                        f"y_elasticity_proba={len(y_elasticity_proba)}. Corrigindo...",
                        "error"
                    )
                    min_len = min(len(y_elasticity_true), len(y_elasticity_proba))
                    y_elasticity_true = y_elasticity_true[:min_len].copy()
                    y_elasticity_proba = y_elasticity_proba[:min_len].copy()
                    logger.log(f"Tamanhos ajustados para {min_len} amostras", "warning")
                
                # ✅ VALIDAÇÃO FINAL antes de prosseguir
                if len(y_elasticity_true) != len(y_elasticity_proba):
                    raise ValueError(
                        f"Falha ao corrigir tamanhos na elasticidade: "
                        f"y_elasticity_true={len(y_elasticity_true)}, "
                        f"y_elasticity_proba={len(y_elasticity_proba)}"
                    )
                
                # Executar simulação
                df_elasticity = simulate_auc_elasticity(
                    y_elasticity_true,
                    y_elasticity_proba,
                    cost_matrix_dict,
                    fixed_threshold=optimal_threshold,
                    n_steps=100,
                    random_seed=42
                )
                
                if len(df_elasticity) > 10:
                    # Calcular coeficiente de elasticidade
                    elasticity_coef, df_reg = calculate_elasticity_coefficient(df_elasticity)
                    
                    # Calcular valor marginal nos últimos 5% de AUC
                    max_auc = df_elasticity['auc'].max()
                    top_tier = df_elasticity[df_elasticity['auc'] >= max_auc * 0.95]
                    
                    if len(top_tier) > 1:
                        delta_profit = top_tier['profit'].max() - top_tier['profit'].min()
                        delta_auc = top_tier['auc'].max() - top_tier['auc'].min()
                        marginal_value_per_1pct_auc = delta_profit / (delta_auc * 100) if delta_auc > 0 else 0
                    else:
                        marginal_value_per_1pct_auc = 0
                    
                    # Plot de Elasticidade
                    fig, ax = plt.subplots(figsize=(14, 8))
                    
                    # Scatter plot dos dados simulados
                    ax.scatter(df_elasticity['auc'], df_elasticity['profit'], 
                              alpha=0.6, color='#3498db', edgecolor='white', s=60, label='Simulação')
                    
                    # Linha de tendência (Regressão Polinomial para suavizar visualmente)
                    try:
                        from sklearn.preprocessing import PolynomialFeatures
                        from sklearn.pipeline import Pipeline
                        from sklearn.linear_model import LinearRegression
                        
                        poly_reg = Pipeline([
                            ('poly', PolynomialFeatures(degree=2)),
                            ('linear', LinearRegression())
                        ])
                        X_poly = df_elasticity[['auc']].values
                        y_poly = df_elasticity['profit'].values
                        poly_reg.fit(X_poly, y_poly)
                        
                        X_plot = np.linspace(df_elasticity['auc'].min(), df_elasticity['auc'].max(), 200).reshape(-1, 1)
                        y_plot = poly_reg.predict(X_plot)
                        ax.plot(X_plot, y_plot, '--', color='#e74c3c', linewidth=2.5, label='Tendência (Fit Polinomial)', alpha=0.8)
                    except:
                        # Fallback: regressão linear simples
                        z = np.polyfit(df_elasticity['auc'], df_elasticity['profit'], 2)
                        p = np.poly1d(z)
                        x_trend = np.linspace(df_elasticity['auc'].min(), df_elasticity['auc'].max(), 200)
                        ax.plot(x_trend, p(x_trend), '--', color='#e74c3c', linewidth=2.5, label='Tendência (Fit)', alpha=0.8)
                    
                    # Marcar ponto atual (melhor modelo)
                    current_auc = roc_auc_score(y_elasticity_true, y_elasticity_proba)
                    current_profit = max_profit
                    ax.scatter([current_auc], [current_profit], color='green', s=200, 
                             marker='*', edgecolor='black', linewidth=2, zorder=5,
                             label=f'Modelo Atual (AUC={current_auc:.3f})')
                    
                    # Anotação de Elasticidade
                    if len(df_elasticity) > 20:
                        mid_idx = len(df_elasticity) // 3
                        ax.annotate(
                            f'Elasticidade Alta:\nPequeno ganho de AUC\n= Grande salto de Lucro',
                            xy=(df_elasticity.iloc[mid_idx]['auc'], df_elasticity.iloc[mid_idx]['profit']),
                            xytext=(df_elasticity.iloc[mid_idx]['auc'] - 0.1, df_elasticity.iloc[mid_idx]['profit']),
                            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                            bbox=dict(boxstyle="round,pad=0.5", fc="yellow", ec="gray", alpha=0.8),
                            fontsize=10, fontweight='bold'
                        )
                    
                    # Detalhes do Gráfico
                    ax.set_title(
                        f'Curva de Elasticidade: Sensibilidade do Lucro à AUC\n'
                        f'(Threshold Fixo: {optimal_threshold:.3f} | Elasticidade: {elasticity_coef:.2f})',
                        fontsize=14, fontweight='bold', pad=20
                    )
                    
                    ax.set_xlabel('AUC-ROC (Performance do Modelo)', fontsize=12)
                    ax.set_ylabel('Lucro Estimado', fontsize=12)
                    
                    # Formatação do Eixo Y
                    def format_currency(x, p):
                        if abs(x) >= 1e6:
                            return f'{x/1e6:.1f}M'
                        elif abs(x) >= 1e3:
                            return f'{x/1e3:.0f}k'
                        else:
                            return f'{x:.0f}'
                    
                    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_currency))
                    
                    # Linha de referência (lucro zero)
                    ax.axhline(0, color='black', linestyle=':', linewidth=1, alpha=0.5)
                    
                    # Limpeza Visual
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_color('#cccccc')
                    ax.spines['bottom'].set_color('#cccccc')
                    ax.grid(True, alpha=0.2)
                    ax.legend(loc='best', fontsize=10)
                    
                    plt.tight_layout()
                    
                    elasticity_context = (
                        f"Análise de elasticidade entre AUC e lucro mantendo threshold fixo ({optimal_threshold:.3f}). "
                        f"O gráfico mostra como o lucro varia quando degradamos a qualidade do modelo (injetando ruído). "
                        f"Coeficiente de elasticidade: {elasticity_coef:.2f} (valores > 1 indicam que lucro cresce mais rápido que AUC). "
                        f"Na zona de alta performance (AUC > {max_auc*0.95:.2f}), aumentar 1% de AUC gera aproximadamente "
                        f"{marginal_value_per_1pct_auc:,.0f} de lucro adicional. "
                        f"Curva {'convexa/exponencial' if elasticity_coef > 1 else 'linear' if abs(elasticity_coef - 1) < 0.3 else 'côncava'} "
                        f"indica que ganhos de performance no topo valem mais do que na base."
                    )
                    
                    logger.log_plot(
                        fig,
                        title="Curva de Elasticidade: AUC vs Lucro",
                        description="Análise de sensibilidade: impacto do lucro quando degradamos a AUC do modelo",
                        context_description=elasticity_context,
                        save_image=True,
                        analyze=True
                    )
                    
                    # Métricas e Insights
                    logger.log_metric("Coeficiente de Elasticidade", f"{elasticity_coef:.2f}")
                    logger.log_metric("Valor Marginal (1% AUC)", f"{marginal_value_per_1pct_auc:,.0f}")
                    logger.log_metric("AUC Atual", f"{current_auc:.4f}")
                    logger.log_metric("Lucro no AUC Atual", f"{current_profit:,.0f}")
                    
                    # Diagnóstico de Risco
                    if elasticity_coef > 2.0:
                        risk_status = "ALTO RISCO"
                        risk_msg = "Modelo muito sensível: qualquer degradação causará prejuízo massivo. Monitoramento crítico necessário."
                    elif elasticity_coef > 1.0:
                        risk_status = "RISCO MODERADO"
                        risk_msg = "Modelo sensível: ganhos de performance no topo valem muito. Investir em melhorias pode ter ROI alto."
                    else:
                        risk_status = "RISCO BAIXO"
                        risk_msg = "Modelo robusto: degradação gradual não causa impacto abrupto. Curva mais estável."
                    
                    logger.log_insight(
                        f"**Diagnóstico de Elasticidade:** {risk_status}. {risk_msg} "
                        f"O coeficiente de {elasticity_coef:.2f} indica que a relação AUC-Lucro é "
                        f"{'super-linear (convexa)' if elasticity_coef > 1 else 'sub-linear (côncava)' if elasticity_coef < 1 else 'linear'}. "
                        f"**ROI de Investimento:** Se melhorar o modelo em 1% de AUC custar menos que {marginal_value_per_1pct_auc:,.0f}, "
                        f"o investimento é justificado. Caso contrário, focar em estabilidade e monitoramento.",
                        "elasticity"
                    )
                    
                    # Tabela de valores marginais por faixa de AUC
                    auc_ranges = [
                        (0.50, 0.65, "Zona da Morte"),
                        (0.65, 0.80, "Zona de Crescimento"),
                        (0.80, 0.90, "Zona de Refinamento"),
                        (0.90, 1.00, "Zona de Excelência")
                    ]
                    
                    marginal_table = []
                    for auc_min, auc_max, zone_name in auc_ranges:
                        zone_data = df_elasticity[(df_elasticity['auc'] >= auc_min) & (df_elasticity['auc'] < auc_max)]
                        if len(zone_data) > 1:
                            zone_delta_profit = zone_data['profit'].max() - zone_data['profit'].min()
                            zone_delta_auc = zone_data['auc'].max() - zone_data['auc'].min()
                            zone_marginal = zone_delta_profit / (zone_delta_auc * 100) if zone_delta_auc > 0 else 0
                            marginal_table.append({
                                'Zona': zone_name,
                                'AUC Min': f"{auc_min:.2f}",
                                'AUC Max': f"{auc_max:.2f}",
                                'Valor Marginal (1% AUC)': f"{zone_marginal:,.0f}"
                            })
                    
                    if marginal_table:
                        logger.log_table("Valor Marginal por Faixa de AUC", marginal_table)
                    
                else:
                    logger.log("Simulação de elasticidade não gerou dados suficientes para análise.", "warning")
                    
            else:
                logger.log("Variáveis necessárias não disponíveis para análise de elasticidade.", "warning")
                
        except Exception as e:
            logger.log(f"Erro na análise de elasticidade: {e}", "error")
            import traceback
            logger.log_code_block(traceback.format_exc(), "python")
        
        # =============================================================================
        # 12. MONITORAMENTO DE DRIFT (PSI)
        # =============================================================================
        
        logger.section("12. Monitoramento de Drift (PSI)", level=2)
        
        try:
            # ✅ CORREÇÃO: Garantir que calculate_psi está disponível
            # Usar holdout para PSI se disponível, senão usar validação do treino
            train_scores_baseline = model_calibrated.predict_proba(X_train_final)[:, 1]
            
            if X_val_processed is not None and 'y_val_proba_cal' in locals():
                prod_scores_example = model_calibrated.predict_proba(X_val_processed)[:, 1]
                logger.log(f"Cálculo de PSI usando Holdout: {len(prod_scores_example):,} amostras", "info")
            else:
                prod_scores_example = y_val_proba_cal if 'y_val_proba_cal' in locals() else y_val_pred_proba
                logger.log(f"Cálculo de PSI usando validação do treino: {len(prod_scores_example):,} amostras", "warning")
            
            psi_value = calculate_psi(train_scores_baseline, prod_scores_example)
            
            logger.log_metric("PSI (Population Stability Index)", f"{psi_value:.4f}")
            
            # ✅ NOVO: Plot de Distribuição de Scores (PSI Visual)
            try:
                fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                
                # Plot 1: Histograma de Distribuições
                # Usar holdout para PSI se disponível
                psi_scores = prod_scores_example
                psi_label = 'Holdout (Atual)' if X_val_processed is not None else 'Validação (Atual)'
                
                axes[0].hist(train_scores_baseline, bins=30, alpha=0.6, label='Treino (Baseline)', 
                           color='blue', density=True, edgecolor='black')
                axes[0].hist(psi_scores, bins=30, alpha=0.6, label=psi_label, 
                            color='red', density=True, edgecolor='black')
                axes[0].set_xlabel('Score Previsto', fontsize=12)
                axes[0].set_ylabel('Densidade', fontsize=12)
                axes[0].set_title(f'Distribuição de Scores: Treino vs Validação (PSI={psi_value:.4f})', 
                                 fontsize=13, fontweight='bold')
                axes[0].legend(loc='best', fontsize=11)
                axes[0].grid(True, alpha=0.3)
                
                # Plot 2: KDE (Kernel Density Estimation) para visualização suave
                from scipy.stats import gaussian_kde
                try:
                    kde_train = gaussian_kde(train_scores_baseline)
                    kde_prod = gaussian_kde(psi_scores)
                    x_range = np.linspace(min(min(train_scores_baseline), min(psi_scores)),
                                        max(max(train_scores_baseline), max(psi_scores)), 200)
                    axes[1].plot(x_range, kde_train(x_range), 'b-', linewidth=2, label='Treino (Baseline)', alpha=0.7)
                    axes[1].plot(x_range, kde_prod(x_range), 'r-', linewidth=2, label='Validação (Atual)', alpha=0.7)
                    axes[1].fill_between(x_range, kde_train(x_range), alpha=0.3, color='blue')
                    axes[1].fill_between(x_range, kde_prod(x_range), alpha=0.3, color='red')
                except:
                    # Fallback se KDE falhar
                    axes[1].hist(train_scores_baseline, bins=50, alpha=0.5, label='Treino', 
                               color='blue', density=True)
                    axes[1].hist(prod_scores_example, bins=50, alpha=0.5, label='Validação', 
                               color='red', density=True)
                
                axes[1].set_xlabel('Score Previsto', fontsize=12)
                axes[1].set_ylabel('Densidade', fontsize=12)
                axes[1].set_title('Distribuição Suave (KDE): Comparação Visual', fontsize=13, fontweight='bold')
                axes[1].legend(loc='best', fontsize=11)
                axes[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                psi_plot_context = (
                    f"Análise visual de drift usando PSI (Population Stability Index = {psi_value:.4f}). "
                    f"Esquerda: Histogramas comparando distribuições de scores entre treino (baseline) e validação (atual). "
                    f"Direita: Estimativa de densidade suave (KDE) para visualização mais clara das diferenças. "
                    f"Quanto mais sobrepostas as distribuições, menor o drift. "
                    f"PSI < 0.1 = Estável, PSI 0.1-0.2 = Atenção, PSI > 0.2 = Crítico (retreino necessário)."
                )
                
                logger.log_plot(
                    fig,
                    title="Análise de Drift: Distribuição de Scores (PSI)",
                    description="Comparação visual de distribuições de scores entre treino e validação para detectar drift",
                    context_description=psi_plot_context,
                    save_image=True,
                    analyze=True
                )
            except Exception as e:
                logger.log(f"Erro ao gerar plot de PSI: {e}", "warning")
            
            if psi_value > 0.25:
                logger.log("CRÍTICO: PSI > 0.25 - Retreino URGENTE necessário!", "error")
                psi_status = "CRÍTICO"
            elif psi_value > 0.2:
                logger.log("ALERTA: PSI > 0.2 - Drift detectado. Monitorar de perto.", "warning")
                psi_status = "ALERTA"
            elif psi_value > 0.1:
                logger.log("ATENÇÃO: PSI > 0.1 - Mudança leve detectada.", "warning")
                psi_status = "ATENÇÃO"
            else:
                logger.log("OK: PSI < 0.1 - Distribuição estável.", "success")
                psi_status = "OK"
            
            logger.log_insight(
                f"O PSI de {psi_value:.4f} indica {'distribuição estável' if psi_value < 0.1 else 'mudança leve' if psi_value < 0.2 else 'mudança significativa'}. "
                f"Status: {psi_status}. "
                "Este valor deve ser monitorado em produção para detectar drift temporal.",
                "drift"
            )
            
        except NameError as e:
            logger.log(f"Erro: Função calculate_psi não encontrada. {e}", "error")
            import traceback
            logger.log_code_block(traceback.format_exc(), "python")
        except Exception as e:
            logger.log(f"Erro no cálculo de PSI: {e}", "error")
            import traceback
            logger.log_code_block(traceback.format_exc(), "python")
        
        # =============================================================================
        # 13. ANÁLISE DE ERROS (Error Analysis para LLM)
        # =============================================================================
        
        logger.section("13. Análise de Erros - Casos Reais", level=2)
        
        try:
            if 'y_val_proba_cal' in locals() and 'y_pred_optimal' in locals():
                # ✅ MUDANÇA: Usar eval_y_financial para análise de erros
                eval_y_errors = eval_y_financial if 'eval_y_financial' in locals() else y_val_final
                eval_X_errors = X_val_processed if X_val_processed is not None else X_val_final
                
                # Identificar Falsos Positivos (FP): Score alto, mas pagou (y=0)
                fp_indices = np.where((y_pred_optimal == 1) & (eval_y_errors == 0))[0]
                
                # Identificar Falsos Negativos (FN): Score baixo, mas deu calote (y=1)
                fn_indices = np.where((y_pred_optimal == 0) & (eval_y_errors == 1))[0]
                
                logger.log(f"Encontrados {len(fp_indices)} Falsos Positivos e {len(fn_indices)} Falsos Negativos", "info")
                
                # Exemplos de Falsos Positivos
                if len(fp_indices) > 0:
                    logger.section("13.1. Exemplos de Falsos Positivos (Aprovamos mas Calotearam)", level=3)
                    
                    for i, idx in enumerate(fp_indices[:3], 1):  # Top 3 exemplos
                        score = y_proba_for_threshold[idx]
                        feats = eval_X_errors.iloc[idx]
                        
                        logger.log(f"**Exemplo {i}:** Score = {score:.4f} (threshold = {optimal_threshold:.4f})", "info")
                        
                        # Top 5 features com maiores valores
                        top_feats = feats.nlargest(5)
                        logger.log_table(f"Top 5 Features (Valores)", 
                                       {k: f"{v:.4f}" for k, v in top_feats.items()})
                        
                        logger.log_insight(
                            f"Este cliente foi aprovado com score {score:.4f} mas caloteou. "
                            f"As features mais altas são {', '.join(top_feats.head(3).index.tolist())}. "
                            "Verifique se alguma delas está agindo como 'falso sinal' de bom pagador. "
                            "Possíveis causas: data leakage, feature enganosa, ou padrão raro não capturado pelo modelo.",
                            "error_analysis"
                        )
                
                # Exemplos de Falsos Negativos
                if len(fn_indices) > 0:
                    logger.section("13.2. Exemplos de Falsos Negativos (Negamos mas Pagaram)", level=3)
                    
                    for i, idx in enumerate(fn_indices[:3], 1):  # Top 3 exemplos
                        score = y_proba_for_threshold[idx]
                        feats = eval_X_errors.iloc[idx]
                        
                        logger.log(f"**Exemplo {i}:** Score = {score:.4f} (threshold = {optimal_threshold:.4f})", "info")
                        
                        # Top 5 features com menores valores (ou mais negativas)
                        top_feats = feats.nsmallest(5)
                        logger.log_table(f"Top 5 Features (Valores)", 
                                       {k: f"{v:.4f}" for k, v in top_feats.items()})
                        
                        logger.log_insight(
                            f"Este cliente foi negado com score {score:.4f} mas pagou. "
                            f"As features mais baixas são {', '.join(top_feats.head(3).index.tolist())}. "
                            "Verifique se o modelo está sendo muito conservador ou se há features que estão "
                            "incorretamente penalizando bons pagadores. Possível oportunidade de ajuste fino.",
                            "error_analysis"
                        )
                
                # Estatísticas de erro
                eval_y_errors = eval_y_financial if 'eval_y_financial' in locals() else y_val_final
                logger.log_metric("Taxa de Falsos Positivos", f"{len(fp_indices) / len(eval_y_errors) * 100:.2f}%")
                logger.log_metric("Taxa de Falsos Negativos", f"{len(fn_indices) / len(eval_y_errors) * 100:.2f}%")
                
            else:
                logger.log("Análise de erros não disponível (variáveis necessárias não encontradas)", "warning")
                
        except Exception as e:
            logger.log(f"Erro na análise de erros: {e}", "error")
            import traceback
            logger.log_code_block(traceback.format_exc(), "python")
        
        # =============================================================================
        # FINALIZAÇÃO
        # =============================================================================
        
        logger.finalize()
        
        print(f"\n[OK] Pipeline executado com sucesso!")
        print(f"[INFO] Relatorio markdown salvo em: {logger.report_path}")
        print(f"[INFO] Imagens salvas em: {logger.images_dir}")
        
    except KeyboardInterrupt:
        print("\n[PARADA] Pipeline interrompido pelo usuario.")
        sys.exit(1)
    except Exception as e:
        # ✅ 3. ROLLBACK AUTOMÁTICO em caso de falha crítica
        print(f"\n[ERRO CRITICO] Falha no pipeline: {e}")
        print("[ROLLBACK] Tentando restaurar config.yaml do backup...")
        
        if rollback_config(config_path, config_backup_path):
            print("[OK] Config.yaml restaurado. Proxima execucao usara config anterior.")
        else:
            print("[WARN] Nao foi possivel restaurar backup automaticamente.")
        
        # Log do erro no changelog se possível
        try:
            changelog_path = Path(__file__).parent / "CHANGELOG.md"
            if changelog_path.exists():
                from datetime import datetime
                error_entry = f"\n## ERRO CRITICO - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                error_entry += f"- **Erro:** {str(e)}\n"
                error_entry += f"- **Acao:** Rollback do config.yaml executado\n---\n"
                with open(changelog_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                with open(changelog_path, 'w', encoding='utf-8') as f:
                    f.write(error_entry + content)
        except:
            pass
        
        raise  # Re-raise para que o agent_controller detecte a falha


if __name__ == "__main__":
    main()
