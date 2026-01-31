"""
Módulo de Logging Markdown Multimodal para Credit Scoring
Gera relatórios estruturados enriquecidos com análise visual via LLM.
"""

import os
import base64
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
import json
import matplotlib.pyplot as plt


# ==============================================================================
# 🧠 CÉREBRO VISUAL (Integração com LLM)
# ==============================================================================

def load_env_file(env_path: Path) -> dict:
    """
    Carrega variáveis de um arquivo .env.
    
    Args:
        env_path: Caminho para o arquivo .env
    
    Returns:
        Dicionário com as variáveis do .env
    """
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
            print(f"⚠️ Erro ao ler arquivo .env: {e}")
    return env_vars


class VisionAnalyst:
    """Classe responsável por enviar imagens para uma LLM e obter análises."""
    
    def __init__(self, api_key: str = None, provider: str = "openai", model_name: str = None):
        """
        Inicializa o analista visual.
        
        Args:
            api_key: Chave da API (se None, tenta pegar do ambiente ou .env)
            provider: Provedor a usar ('openai', 'gemini', 'claude')
            model_name: Nome do modelo a usar (especialmente para Gemini, lê do .env se None)
        """
        # Tentar carregar do .env primeiro
        env_path = Path(__file__).parent / '.env'
        env_vars = load_env_file(env_path)
        
        # Determinar API key: parâmetro > .env > variável de ambiente
        if api_key:
            self.api_key = api_key
        elif provider.lower() == "gemini" and "GEMINI_KEY" in env_vars:
            self.api_key = env_vars["GEMINI_KEY"]
            print(f"   ✅ API Key carregada do arquivo .env")
        else:
            self.api_key = (
                os.getenv("OPENAI_API_KEY") or 
                os.getenv("GEMINI_API_KEY") or 
                os.getenv("ANTHROPIC_API_KEY")
            )
        
        self.provider = provider.lower()
        
        # Determinar model_name: parâmetro > .env > default
        if model_name:
            self.model_name = model_name
        elif provider.lower() == "gemini" and "MODEL_NAME" in env_vars:
            self.model_name = env_vars["MODEL_NAME"]
            print(f"   ✅ Modelo carregado do arquivo .env: {self.model_name}")
        elif provider.lower() == "gemini":
            self.model_name = "gemini-1.5-pro"  # Default
        elif provider.lower() == "openai":
            self.model_name = "gpt-4o"
        elif provider.lower() == "claude":
            self.model_name = "claude-3-5-sonnet-20241022"
        else:
            self.model_name = None
        
        self._client = None
    
    def _get_client(self):
        """Lazy loading do cliente da API."""
        if self._client is not None:
            return self._client
        
        if not self.api_key:
            return None
        
        try:
            if self.provider == "openai":
                try:
                    from openai import OpenAI
                    self._client = OpenAI(api_key=self.api_key)
                    return self._client
                except ImportError:
                    print("⚠️ Biblioteca 'openai' não instalada. Instale com: pip install openai")
                    return None
            
            elif self.provider == "gemini":
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=self.api_key)
                    # Usa o model_name do .env ou parâmetro, senão usa default
                    model_to_use = self.model_name or "gemini-1.5-pro"
                    self._client = genai.GenerativeModel(model_to_use)
                    print(f"   📌 Usando modelo Gemini: {model_to_use}")
                    return self._client
                except ImportError:
                    print("⚠️ Biblioteca 'google-generativeai' não instalada. Instale com: pip install google-generativeai")
                    return None
            
            elif self.provider == "claude":
                try:
                    from anthropic import Anthropic
                    self._client = Anthropic(api_key=self.api_key)
                    return self._client
                except ImportError:
                    print("⚠️ Biblioteca 'anthropic' não instalada. Instale com: pip install anthropic")
                    return None
            
        except Exception as e:
            print(f"⚠️ Erro ao inicializar cliente {self.provider}: {e}")
            return None
    
    def analyze_image(self, image_path: str, local_context: str, global_context: Dict = None) -> str:
        """
        Envia a imagem para a LLM e retorna a análise textual.
        Agora com suporte a contexto global para evitar alucinações.
        
        Args:
            image_path: Caminho para a imagem
            local_context: Contexto técnico específico deste gráfico
            global_context: Dicionário com contexto global do estudo (dataset, métricas, etc)
        
        Returns:
            Análise textual da imagem
        """
        if not self.api_key:
            return (
                "> *[IA Vision - Modo Simulação]*: API Key não configurada. "
                "Configure GEMINI_KEY no arquivo .env ou variáveis de ambiente para análise visual automática. "
                "O gráfico foi salvo e pode ser analisado manualmente."
            )
        
        client = self._get_client()
        if not client:
            return (
                "> *[IA Vision - Erro]*: Não foi possível inicializar o cliente da API. "
                "Verifique se a biblioteca do provider está instalada e a API key está correta."
            )
        
        try:
            # Codificar imagem em base64
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Prompt Engineering Hierárquico Aprimorado
            system_prompt = (
                "Você é um Cientista de Dados Sênior (Head de Risco) especialista em Credit Scoring. "
                "Sua função é validar visualmente os dados numéricos fornecidos e gerar insights acionáveis.\n\n"
                "REGRAS DE OURO:\n"
                "1. O 'Contexto Global' contém a verdade sobre os dados (tamanhos, balanceamento, métricas anteriores). "
                "Use isso para contextualizar sua análise.\n"
                "2. O 'Contexto Local' contém cálculos geométricos da curva. "
                "Se o cálculo parecer contradizer a imagem (ex: texto diz 'slope 0' mas imagem mostra subida), "
                "APONTE A DISCREPÂNCIA e confie na imagem para a análise qualitativa.\n"
                "3. Seja cético e crítico. Se o AUC é alto mas a curva parece ruim, alerte. "
                "Se há contradição entre texto e imagem, identifique e explique.\n"
                "4. Conecte os pontos: Use o contexto global (ex: desbalanceamento severo) para explicar "
                "comportamentos observados no gráfico.\n"
                "5. Seja técnico mas prático: Foque em geometria da curva, tendências, anomalias, "
                "sinais de overfitting, e conclusões para tomada de decisão."
            )
            
            # Formatar contexto global como JSON legível
            if global_context and len(global_context) > 0:
                global_summary = json.dumps(global_context, indent=2, ensure_ascii=False)
                global_section = f"--- CONTEXTO GLOBAL DO ESTUDO ---\n{global_summary}\n\n"
            else:
                global_section = ""
            
            user_prompt = (
                f"{global_section}"
                f"--- CONTEXTO TÉCNICO DESTE GRÁFICO ---\n{local_context}\n\n"
                f"--- TAREFA ---\n"
                "Analise a imagem anexa. Valide se a geometria visual condiz com os números fornecidos. "
                "Se houver contradição entre texto e imagem, identifique e explique. "
                "Forneça:\n"
                "1. Observações sobre a forma/geometria da curva (validando contra o contexto local)\n"
                "2. Sinais de problemas (overfitting, drift, etc) considerando o contexto global\n"
                "3. Conclusões práticas para o negócio conectando contexto global + visual\n"
                "Seja conciso mas completo (máximo 250 palavras)."
            )
            
            # Chamada para OpenAI GPT-4o
            if self.provider == "openai":
                response = client.chat.completions.create(
                    model=self.model_name or "gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": [
                            {"type": "text", "text": user_prompt},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/png;base64,{encoded_string}",
                                "detail": "high"
                            }}
                        ]}
                    ],
                    max_tokens=400,
                    temperature=0.3
                )
                return response.choices[0].message.content
            
            # Chamada para Google Gemini
            elif self.provider == "gemini":
                import PIL.Image
                image = PIL.Image.open(image_path)
                # O cliente já foi inicializado com o model_name correto em _get_client()
                # Não precisa especificar novamente aqui
                # Gemini recebe system + user prompt combinados
                full_prompt = system_prompt + "\n\n" + user_prompt
                response = client.generate_content([full_prompt, image])
                return response.text
            
            # Chamada para Anthropic Claude
            elif self.provider == "claude":
                with open(image_path, "rb") as image_file:
                    image_data = image_file.read()
                
                message = client.messages.create(
                    model=self.model_name or "claude-3-5-sonnet-20241022",
                    max_tokens=400,
                    system=system_prompt,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": encoded_string
                                }},
                                {"type": "text", "text": user_prompt}
                            ]
                        }
                    ]
                )
                return message.content[0].text
            
            else:
                return f"> *[Erro]*: Provider '{self.provider}' não implementado."
        
        except Exception as e:
            error_msg = str(e)
            return (
                f"> *[Erro na Análise Visual]*: {error_msg}\n"
                "> O gráfico foi salvo mas a análise automática falhou. "
                "Verifique sua API key e conexão com a internet."
            )
    
    def text_inference(self, system_prompt: str, user_prompt: str) -> str:
        """
        Método para inferência de texto puro (sem imagem).
        Usado pelo meta_controller para decisões baseadas em relatórios.
        
        Args:
            system_prompt: Prompt do sistema
            user_prompt: Prompt do usuário
        
        Returns:
            Resposta textual da LLM
        """
        if not self.api_key:
            return json.dumps({
                "decision": "STOP",
                "reasoning": "API Key não configurada. Não é possível fazer inferência.",
                "changes": {}
            })
        
        client = self._get_client()
        if not client:
            return json.dumps({
                "decision": "STOP",
                "reasoning": "Cliente LLM não inicializado.",
                "changes": {}
            })
        
        try:
            # Chamada para OpenAI GPT-4o
            if self.provider == "openai":
                response = client.chat.completions.create(
                    model=self.model_name or "gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=1000,
                    temperature=0.3,
                    response_format={"type": "json_object"}  # Força resposta JSON
                )
                return response.choices[0].message.content
            
            # Chamada para Google Gemini
            elif self.provider == "gemini":
                # Gemini precisa de instrução explícita para JSON
                json_prompt = f"{system_prompt}\n\nIMPORTANTE: Responda APENAS com JSON válido, sem markdown, sem texto adicional.\n\n{user_prompt}"
                response = client.generate_content(json_prompt)
                text = response.text.strip()
                # Remove markdown code blocks se houver
                if text.startswith("```json"):
                    text = text.replace("```json", "").replace("```", "").strip()
                elif text.startswith("```"):
                    text = text.replace("```", "").strip()
                return text
            
            # Chamada para Anthropic Claude
            elif self.provider == "claude":
                message = client.messages.create(
                    model=self.model_name or "claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    system=system_prompt,
                    messages=[
                        {
                            "role": "user",
                            "content": f"{user_prompt}\n\nIMPORTANTE: Responda APENAS com JSON válido, sem markdown."
                        }
                    ]
                )
                text = message.content[0].text.strip()
                # Remove markdown code blocks se houver
                if text.startswith("```json"):
                    text = text.replace("```json", "").replace("```", "").strip()
                elif text.startswith("```"):
                    text = text.replace("```", "").strip()
                return text
            
            else:
                return json.dumps({
                    "decision": "STOP",
                    "reasoning": f"Provider '{self.provider}' não implementado para text_inference.",
                    "changes": {}
                })
        
        except Exception as e:
            error_msg = str(e)
            return json.dumps({
                "decision": "STOP",
                "reasoning": f"Erro na inferência LLM: {error_msg}",
                "changes": {}
            })


# ==============================================================================
# 📝 LOGGER PRINCIPAL
# ==============================================================================

class MarkdownLogger:
    """
    Logger que gera relatórios markdown estruturados para análise por LLMs.
    Substitui prints e displays do notebook por logging estruturado.
    """
    
    def __init__(self, output_dir: str = "reports", run_name: Optional[str] = None, 
                 use_vision_llm: bool = False, vision_provider: str = "openai", 
                 vision_api_key: str = None, vision_model_name: str = None):
        """
        Inicializa o logger markdown.
        
        Args:
            output_dir: Diretório onde salvar os relatórios
            run_name: Nome da execução (se None, usa timestamp)
            use_vision_llm: Se True, ativa análise visual automática de gráficos
            vision_provider: Provedor de LLM ('openai', 'gemini', 'claude')
            vision_api_key: Chave da API (se None, tenta pegar do ambiente ou .env)
            vision_model_name: Nome do modelo (especialmente para Gemini, lê do .env se None)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"credit_scoring_{timestamp}"
        
        self.run_name = run_name
        self.report_path = self.output_dir / f"{run_name}.md"
        self.images_dir = self.output_dir / f"{run_name}_images"
        self.images_dir.mkdir(exist_ok=True)
        
        self.sections = []
        self.current_section = None
        self.image_counter = 0
        
        # ✅ MEMÓRIA DE CONTEXTO GLOBAL (Global Context Memory)
        self.global_context = {}  # Armazena fatos importantes do pipeline
        
        # Inicializa o analista visual se solicitado
        self.vision_analyst = None
        if use_vision_llm:
            self.vision_analyst = VisionAnalyst(
                api_key=vision_api_key, 
                provider=vision_provider,
                model_name=vision_model_name
            )
            model_info = f" ({self.vision_analyst.model_name})" if self.vision_analyst.model_name else ""
            print(f"✅ Análise Visual ativada (Provider: {vision_provider}{model_info})")
        
        # Inicializar relatório
        self._init_report()
    
    def update_context(self, key: str, value: Any):
        """
        Atualiza a memória global do logger com fatos importantes do pipeline.
        Este contexto será injetado automaticamente em todas as análises visuais.
        
        Args:
            key: Chave do contexto (ex: 'class_balance_ratio', 'n_features')
            value: Valor do contexto (será formatado automaticamente se for float)
        """
        # Se for float, formata para não poluir o JSON e evitar precisão excessiva
        if isinstance(value, float):
            value = round(value, 4)
        elif isinstance(value, (int, str, bool)):
            value = value
        else:
            # Para outros tipos, converte para string
            value = str(value)
        
        self.global_context[key] = value
    
    def _init_report(self):
        """Inicializa o arquivo markdown com cabeçalho."""
        vision_status = "Ativado" if self.vision_analyst else "Desativado"
        vision_provider = self.vision_analyst.provider if self.vision_analyst else "N/A"
        
        header = f"""# 📊 Relatório de Execução: Credit Scoring Model

**Data/Hora:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Execução:** `{self.run_name}`
**Modo Vision:** `{vision_status}` ({vision_provider})

---

"""
        with open(self.report_path, 'w', encoding='utf-8') as f:
            f.write(header)
    
    def section(self, title: str, level: int = 2):
        """
        Inicia uma nova seção no relatório.
        
        Args:
            title: Título da seção
            level: Nível do cabeçalho (2 = ##, 3 = ###, etc)
        """
        self.current_section = {
            'title': title,
            'level': level,
            'content': [],
            'images': [],
            'metrics': {},
            'insights': []
        }
        self.sections.append(self.current_section)
        
        prefix = '#' * level
        with open(self.report_path, 'a', encoding='utf-8') as f:
            f.write(f"\n{prefix} {title}\n\n")
    
    def log(self, message: str, level: str = "info"):
        """
        Adiciona uma mensagem ao relatório.
        
        Args:
            message: Mensagem a ser logada
            level: Nível da mensagem (info, warning, error, success)
        """
        if self.current_section is None:
            self.section("Log Geral", level=2)
        
        # Emojis por nível
        emoji_map = {
            'info': 'ℹ️',
            'warning': '⚠️',
            'error': '❌',
            'success': '✅',
            'critical': '🚨'
        }
        
        emoji = emoji_map.get(level, 'ℹ️')
        formatted_message = f"{emoji} {message}"
        
        self.current_section['content'].append(formatted_message)
        
        with open(self.report_path, 'a', encoding='utf-8') as f:
            f.write(f"{formatted_message}\n\n")
    
    def _format_value(self, value: Any) -> str:
        """
        Formata valores de forma segura para Markdown.
        Remove cifrão R$ para evitar conflito com LaTeX.
        
        Args:
            value: Valor a ser formatado
        
        Returns:
            String formatada
        """
        if isinstance(value, float):
            formatted = f"{value:.4f}"
        elif isinstance(value, (int, str)):
            formatted = str(value)
        else:
            formatted = str(value)
        
        # ✅ Remover R$ para evitar conflito com LaTeX no Markdown
        # Substituir por BRL ou apenas remover o prefixo
        if "R$" in formatted:
            formatted = formatted.replace("R$", "").strip()
            # Também remover versão escapada se existir (usando raw string para evitar SyntaxWarning)
            formatted = formatted.replace(r"R\$", "").strip()
            # Se houver valor numérico, manter apenas o número
            # Ex: "R$ 1.000,00" -> "1.000,00" ou "1000.00"
        
        return formatted
    
    def log_metric(self, name: str, value: Any, description: Optional[str] = None):
        """
        Registra uma métrica no relatório.
        
        Args:
            name: Nome da métrica
            value: Valor da métrica
            description: Descrição opcional
        """
        if self.current_section is None:
            self.section("Métricas", level=2)
        
        # ✅ Usar formatação segura
        formatted_value = self._format_value(value)
        
        self.current_section['metrics'][name] = {
            'value': formatted_value,
            'raw': value,
            'description': description
        }
        
        # Escrever no markdown
        with open(self.report_path, 'a', encoding='utf-8') as f:
            f.write(f"**{name}:** {formatted_value}")
            if description:
                f.write(f" - {description}")
            f.write("\n\n")
    
    def log_table(self, title: str, data: Dict[str, Any] or List[Dict], headers: Optional[List[str]] = None):
        """
        Registra uma tabela no relatório.
        
        Args:
            title: Título da tabela
            data: Dados da tabela (dict ou lista de dicts)
            headers: Cabeçalhos opcionais
        """
        if self.current_section is None:
            self.section("Tabelas", level=2)
        
        with open(self.report_path, 'a', encoding='utf-8') as f:
            f.write(f"### {title}\n\n")
            
            if isinstance(data, dict):
                # Tabela de chave-valor
                f.write("| Métrica | Valor |\n")
                f.write("|---------|-------|\n")
                for key, value in data.items():
                    # ✅ Usar formatação segura (remove R$)
                    formatted_value = self._format_value(value)
                    f.write(f"| {key} | {formatted_value} |\n")
            elif isinstance(data, list) and len(data) > 0:
                # Tabela de múltiplas linhas
                if headers is None:
                    headers = list(data[0].keys())
                
                f.write("| " + " | ".join(headers) + " |\n")
                f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
                
                for row in data[:20]:  # Limitar a 20 linhas
                    values = [str(row.get(h, "")) for h in headers]
                    f.write("| " + " | ".join(values) + " |\n")
                
                if len(data) > 20:
                    f.write(f"\n*Mostrando 20 de {len(data)} linhas*\n")
            
            f.write("\n")
    
    def log_insight(self, insight: str, category: str = "geral"):
        """
        Registra um insight ou conclusão importante.
        
        Args:
            insight: Texto do insight
            category: Categoria do insight (geral, overfitting, drift, financeiro, etc)
        """
        if self.current_section is None:
            self.section("Insights", level=2)
        
        self.current_section['insights'].append({
            'text': insight,
            'category': category
        })
        
        with open(self.report_path, 'a', encoding='utf-8') as f:
            f.write(f"**💡 Insight ({category}):** {insight}\n\n")
    
    def log_plot(self, fig, description: str, save_image: bool = True, 
                 title: Optional[str] = None, context_description: Optional[str] = None,
                 analyze: bool = True):
        """
        Salva um gráfico e adiciona descrição textual ao relatório.
        Opcionalmente analisa o gráfico com IA Vision se ativado.
        
        Args:
            fig: Figura matplotlib
            description: Descrição textual do gráfico (legado, mantido para compatibilidade)
            save_image: Se True, salva a imagem
            title: Título do gráfico (se None, usa descrição)
            context_description: Contexto técnico detalhado para análise IA (recomendado)
            analyze: Se True e vision_analyst ativo, chama análise visual automática
        """
        if self.current_section is None:
            self.section("Visualizações", level=2)
        
        self.image_counter += 1
        image_filename = f"img_{self.image_counter:03d}_{self.run_name}.png"
        image_path = self.images_dir / image_filename
        
        # Usar title se fornecido, senão usar primeira parte da description
        plot_title = title or description.split('.')[0] if description else f"Gráfico {self.image_counter}"
        
        # Contexto técnico: usar context_description se fornecido, senão description
        technical_context = context_description or description or "Gráfico gerado durante análise de credit scoring."
        
        if save_image:
            fig.savefig(image_path, dpi=100, bbox_inches='tight')
            plt.close(fig)  # Fecha para liberar memória
        
        self.current_section['images'].append({
            'path': str(image_path),
            'description': description or technical_context,
            'filename': image_filename,
            'analyzed': analyze and self.vision_analyst is not None
        })
        
        # Preparar conteúdo para escrita (evita erro de I/O em arquivo fechado)
        content_to_write = f"### {plot_title}\n\n"
        content_to_write += f"![{plot_title}]({self.images_dir.name}/{image_filename})\n\n"
        content_to_write += f"**Contexto Técnico:** {technical_context}\n\n"
        
        # 🧠 ANÁLISE VISUAL AUTOMÁTICA (Processamento antes de escrever)
        # ✅ Agora passa o contexto global para evitar alucinações
        ai_analysis_text = ""
        if analyze and self.vision_analyst:
            print(f"   👁️ Analisando gráfico '{plot_title}' com IA Vision ({self.vision_analyst.provider})...")
            try:
                ai_response = self.vision_analyst.analyze_image(
                    str(image_path), 
                    technical_context,
                    self.global_context  # ✅ INJEÇÃO DO CONTEXTO GLOBAL
                )
                # Formata a análise com blockquote para ficar bonito no markdown
                formatted_analysis = ai_response.replace('\n', '\n> ')
                ai_analysis_text = f"> 🤖 **Análise Visual Automática:**\n>\n> {formatted_analysis}\n\n"
            except Exception as e:
                ai_analysis_text = f"> ⚠️ **Erro na análise visual:** {str(e)}\n\n"
        elif analyze and not self.vision_analyst:
            ai_analysis_text = f"> ℹ️ *Análise visual automática disponível. Configure `use_vision_llm=True` e uma API key para ativar.*\n\n"
        
        # Escrita única no arquivo (evita erro de I/O em arquivo fechado)
        with open(self.report_path, 'a', encoding='utf-8') as f:
            f.write(content_to_write)
            if ai_analysis_text:
                f.write(ai_analysis_text)
    
    def log_plot_description(self, description: str, analysis: str):
        """
        Adiciona descrição textual detalhada de um gráfico.
        
        Args:
            description: Descrição do que o gráfico mostra
            analysis: Análise e conclusões do gráfico
        """
        with open(self.report_path, 'a', encoding='utf-8') as f:
            f.write(f"**Análise do Gráfico:**\n\n")
            f.write(f"{description}\n\n")
            f.write(f"**Conclusões:**\n\n")
            f.write(f"{analysis}\n\n")
    
    def log_code_block(self, code: str, language: str = "python"):
        """
        Adiciona um bloco de código ao relatório.
        
        Args:
            code: Código a ser exibido
            language: Linguagem do código
        """
        with open(self.report_path, 'a', encoding='utf-8') as f:
            f.write(f"```{language}\n{code}\n```\n\n")
    
    def log_summary(self, title: str, items: List[str]):
        """
        Adiciona um resumo em lista ao relatório.
        
        Args:
            title: Título do resumo
            items: Lista de itens
        """
        with open(self.report_path, 'a', encoding='utf-8') as f:
            f.write(f"### {title}\n\n")
            for item in items:
                f.write(f"- {item}\n")
            f.write("\n")
    
    def log_parameters(self, params: Dict[str, Any], section_name: str = "Parâmetros"):
        """
        Registra parâmetros de configuração.
        
        Args:
            params: Dicionário de parâmetros
            section_name: Nome da seção
        """
        with open(self.report_path, 'a', encoding='utf-8') as f:
            f.write(f"### {section_name}\n\n")
            f.write("| Parâmetro | Valor |\n")
            f.write("|-----------|-------|\n")
            for key, value in params.items():
                formatted_value = f"{value:.4f}" if isinstance(value, float) else str(value)
                f.write(f"| `{key}` | {formatted_value} |\n")
            f.write("\n")
    
    def describe_roc_curve(self, fpr: List[float], tpr: List[float], auc: float, 
                          threshold: Optional[float] = None) -> str:
        """
        Gera descrição textual de uma curva ROC.
        
        Args:
            fpr: Taxa de falsos positivos
            tpr: Taxa de verdadeiros positivos
            auc: Valor AUC
            threshold: Threshold usado (opcional)
        
        Returns:
            Descrição textual da curva
        """
        description = f"""
**Análise da Curva ROC:**

- **AUC-ROC:** {auc:.4f}
  - {'Excelente' if auc >= 0.9 else 'Bom' if auc >= 0.8 else 'Moderado' if auc >= 0.7 else 'Fraco'} poder discriminativo
  - O modelo consegue distinguir {'muito bem' if auc >= 0.9 else 'bem' if auc >= 0.8 else 'moderadamente'} entre bons e maus pagadores

- **Forma da Curva:**
  - A curva {'sobe rapidamente' if tpr[10] > 0.5 else 'sobe gradualmente'} no início, indicando que o modelo identifica {'facilmente' if tpr[10] > 0.5 else 'gradualmente'} os casos de maior risco
  - {'Curva próxima do canto superior esquerdo' if auc >= 0.9 else 'Curva acima da diagonal'} indica boa separação de classes

"""
        if threshold:
            description += f"- **Threshold Ótimo:** {threshold:.4f}\n"
            idx = min(range(len(fpr)), key=lambda i: abs(fpr[i] - (1 - threshold)))
            description += f"  - Neste ponto: TPR = {tpr[idx]:.4f}, FPR = {fpr[idx]:.4f}\n"
        
        return description
    
    def describe_distribution(self, data: List[float] or Any, name: str) -> str:
        """
        Gera descrição textual de uma distribuição.
        
        Args:
            data: Dados para análise
            name: Nome da variável
        
        Returns:
            Descrição textual
        """
        import numpy as np
        
        if isinstance(data, (list, np.ndarray)):
            arr = np.array(data)
            mean_val = np.mean(arr)
            std_val = np.std(arr)
            min_val = np.min(arr)
            max_val = np.max(arr)
            median_val = np.median(arr)
            
            description = f"""
**Distribuição de {name}:**

- **Estatísticas Descritivas:**
  - Média: {mean_val:.4f}
  - Mediana: {median_val:.4f}
  - Desvio Padrão: {std_val:.4f}
  - Mínimo: {min_val:.4f}
  - Máximo: {max_val:.4f}

- **Interpretação:**
  - {'Distribuição simétrica' if abs(mean_val - median_val) < 0.1 * std_val else 'Distribuição assimétrica'}
  - {'Baixa variabilidade' if std_val < 0.1 * abs(mean_val) else 'Alta variabilidade'} (CV = {std_val/abs(mean_val) if mean_val != 0 else 'N/A':.2f})
"""
            return description
        
        return f"**{name}:** {str(data)}\n"
    
    def log_dataframe_head(self, df, n=5, title="Amostra de Dados"):
        """
        Adiciona uma tabela markdown do head de um DataFrame.
        
        Args:
            df: DataFrame do pandas
            n: Número de linhas a mostrar
            title: Título da tabela
        """
        if self.current_section is None:
            self.section("Dados", level=2)
        
        with open(self.report_path, 'a', encoding='utf-8') as f:
            f.write(f"### {title}\n\n")
            
            # Cabeçalho
            headers = list(df.columns)
            f.write("| " + " | ".join(headers) + " |\n")
            f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
            
            # Linhas
            for idx, row in df.head(n).iterrows():
                values = [str(val)[:50] if len(str(val)) > 50 else str(val) for val in row]
                f.write("| " + " | ".join(values) + " |\n")
            
            if len(df) > n:
                f.write(f"\n*Mostrando {n} de {len(df)} linhas*\n")
            
            f.write("\n")
    
    def log_timestamp(self, step_name: str):
        """
        Adiciona timestamp para medir performance de cada etapa.
        
        Args:
            step_name: Nome da etapa
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log(f"[{timestamp}] {step_name}", "info")
    
    def finalize(self):
        """Finaliza o relatório adicionando resumo executivo."""
        summary = f"""
---

## 📋 Resumo Executivo

**Data de Execução:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

### Seções do Relatório:

"""
        for i, section in enumerate(self.sections, 1):
            summary += f"{i}. {section['title']}\n"
            if section['metrics']:
                summary += f"   - {len(section['metrics'])} métricas registradas\n"
            if section['insights']:
                summary += f"   - {len(section['insights'])} insights identificados\n"
            if section['images']:
                summary += f"   - {len(section['images'])} visualizações geradas\n"
        
        summary += f"""
### Estatísticas da Execução:

- **Total de Seções:** {len(self.sections)}
- **Total de Métricas:** {sum(len(s['metrics']) for s in self.sections)}
- **Total de Insights:** {sum(len(s['insights']) for s in self.sections)}
- **Total de Visualizações:** {sum(len(s['images']) for s in self.sections)}

---

**Relatório gerado automaticamente pelo sistema de Credit Scoring**
"""
        
        with open(self.report_path, 'a', encoding='utf-8') as f:
            f.write(summary)
        
        print(f"✅ Relatório markdown salvo em: {self.report_path}")
        print(f"📁 Imagens salvas em: {self.images_dir}")
        if self.vision_analyst:
            print(f"🤖 Análise visual: {self.image_counter} gráficos processados")
