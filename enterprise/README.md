# Enterprise AI Multi-Agent Platform

## Arquitetura

Plataforma enterprise de IA multi-agente inspirada em:
- LangGraph
- CrewAI
- AutoGen Studio
- CopilotKit AG-UI
- FlowiseAI
- OpenAI Agents SDK

## Componentes Principais

### 1. Sistema Multi-Agente

- **SupervisorAgent**: Orquestra todos os agentes
- **StrategyAgent**: Planejamento estratégico de longo prazo
- **CombatAgent**: Decisões de combate em tempo real
- **VisionAgent**: Processamento de visão computacional
- **NavigationAgent**: Pathfinding e movimento
- **TacticalPlannerAgent**: Planejamento tático
- **ReplayAnalystAgent**: Análise de replays
- **LearningAgent**: Aprendizagem por reforço
- **MemoryAgent**: Sistema de memória híbrida
- **ReflectionAgent**: Auto-avaliação e crítica
- **CoordinationAgent**: Coordenação entre agentes

### 2. Motor de Orquestração

- Execução paralela de tarefas
- Decomposição recursiva de tarefas
- Sistema de consenso entre agentes
- Dynamic routing e fallbacks
- Workflow automation

### 3. Sistema de Memória

- Short-term memory
- Long-term memory
- Vector memory (embeddings)
- Episodic memory
- Semantic memory

### 4. Observabilidade

- Distributed tracing
- Métricas em tempo real
- Logs estruturados
- Performance profiling

### 5. Computer Vision Pipeline

- YOLOv8/YOLOv11 detector
- ByteTrack tracker
- Minimap understanding
- Heatmap generation

### 6. Aprendizagem Autónoma

- RL Framework (PPO, SAC)
- Imitation Learning
- Curriculum Learning
- Self-play training

## Quick Start

```bash
# Instalar dependências
pip install -r enterprise/requirements.txt

# Executar demo
python enterprise/quickstart.py

# Iniciar API server
cd enterprise/api
python -m uvicorn server:app --reload

# Iniciar dashboard
cd enterprise/dashboard
npm install
npm run dev
```

## API Endpoints

- `GET /` - Informações da plataforma
- `GET /status` - Status do sistema
- `GET /agents` - Lista de agentes
- `POST /agents` - Criar agente
- `GET /agents/{id}` - Detalhes do agente
- `POST /agents/{id}/message` - Enviar mensagem
- `GET /tasks` - Lista de tarefas
- `POST /tasks` - Criar tarefa
- `POST /tasks/{id}/execute` - Executar tarefa
- `WS /ws` - WebSocket para eventos em tempo real

## Dashboard

Interface web moderna com:
- Painel de agentes activos
- Stream de eventos
- Gráfico de workflow
- Métricas do sistema
- Chat com agentes

## Stack Tecnológica

- **Backend**: FastAPI, Python, LangGraph
- **Frontend**: Next.js 15, React 19, TypeScript, TailwindCSS
- **AI/ML**: PyTorch, YOLOv8, RLlib, Stable-Baselines3
- **Infraestrutura**: Redis, PostgreSQL, Qdrant, Docker
