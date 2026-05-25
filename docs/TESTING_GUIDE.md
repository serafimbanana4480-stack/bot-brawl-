# Guia de Teste das Novas Funcionalidades de Treinamento de IA

## Status Atual
- ✅ **Verificação de Emulador**: BlueStacks instalado em `C:\Program Files\BlueStacks_nxt`
- ❌ **Emulador Rodando**: BlueStacks não está rodando atualmente
- 🔄 **Dados Sintéticos**: Gerando 500 amostras melhoradas em background

## Pré-requisitos para Testar com Dados Reais

### 1. Iniciar o Emulador BlueStacks
```bash
# Iniciar BlueStacks manualmente ou via comando:
"C:\Program Files\BlueStacks_nxt\HD-Player.exe"
```

### 2. Configurar ADB no BlueStacks
1. Abra as configurações do BlueStacks
2. Vá em "Settings" > "Advanced"
3. Ative "Android Debug Bridge (ADB)"
4. Anote a porta ADB (geralmente 5555)

### 3. Habilitar Depuração USB no Android
1. Abra as configurações do Android no BlueStacks
2. Vá em "About Phone" > "Build Number"
3. Toque 7 vezes para ativar "Developer Options"
4. Vá em "Developer Options" e ative "USB Debugging"

### 4. Verificar Conexão ADB
```bash
cd "c:\Users\rodri\Desktop\bot brawl"
"C:\Program Files\BlueStacks_nxt\HD-Adb.exe" devices
```

Deve mostrar algo como:
```
List of devices attached
emulator-5554   device
```

## Processo de Teste Completo

### Passo 1: Gerar Dados Sintéticos Melhorados (Já em andamento)
```bash
cd "c:\Users\rodri\Desktop\bot brawl"
"C:\Users\rodri\AppData\Local\Programs\Python\Python312\python.exe" -m brawl_bot.training.synthetic_data_generator --num-samples 500 --sequence-length 5 --output ./dataset/synthetic_v2
```

**Resultado esperado:**
- 500 imagens sintéticas realistas
- Labels no formato YOLO
- Metadados enriquecidos
- Física e trajetórias realistas

### Passo 2: Coletar Dados Reais de Gameplay
Quando o emulador estiver rodando e o Brawl Stars aberto:

```bash
cd "c:\Users\rodri\Desktop\bot brawl"
"C:\Users\rodri\AppData\Local\Programs\Python\Python312\python.exe" -m brawl_bot.main --record
```

**Ou usar o dataset collector v2 diretamente:**
```bash
cd "c:\Users\rodri\Desktop\bot brawl"
"C:\Users\rodri\AppData\Local\Programs\Python\Python312\python.exe" -m brawl_bot.automation.dataset_collector_v2 --adb-id 127.0.0.1:5555 --duration 600 --output ./dataset/real_v2
```

**Dicas para coleta de dados:**
- Jogue 10-20 minutos de gameplay variado
- Inclua diferentes fases: lobby, matchmaking, combate, resultado
- Capture diferentes situações: combate 1v1, 3v3, coleta de power cubes
- Quanto mais variedade, melhor o modelo

### Passo 3: Processar Dados Coletados
Os dados serão automaticamente organizados pelo dataset_collector_v2:
- `./dataset/real_v2/images/` - Imagens capturadas
- `./dataset/real_v2/labels/` - Labels YOLO gerados automaticamente
- `./dataset/real_v2/by_state/` - Organizado por fase do jogo
- `./dataset/real_v2/by_priority/` - Organizado por prioridade (alto/médio/baixo)
- `./dataset/real_v2/metadata/` - Metadados enriquecidos

### Passo 4: Combinar Dados Reais e Sintéticos
```bash
# Criar dataset combinado
mkdir dataset\combined
copy dataset\real_v2\images\* dataset\combined\images\
copy dataset\real_v2\labels\* dataset\combined\labels\
copy dataset\synthetic_v2\images\* dataset\combined\images\
copy dataset\synthetic_v2\labels\* dataset\combined\labels\
```

### Passo 5: Treinar Modelo YOLO com Dados Combinados
```bash
cd "c:\Users\rodri\Desktop\bot brawl"
"C:\Users\rodri\AppData\Local\Programs\Python\Python312\python.exe" -m brawl_bot.training.train_yolo
```

**Nota:** O script `train_yolo.py` precisa ser adaptado para usar o dataset combinado. Modifique o caminho do dataset no script.

### Passo 6: Validar Modelo Treinado
```bash
cd "c:\Users\rodri\Desktop\bot brawl"
"C:\Users\rodri\AppData\Local\Programs\Python\Python312\python.exe" -m brawl_bot.training.training_validator --model-path ./models/best.pt --test-dataset ./dataset/test --output ./validation_report.json
```

### Passo 7: Comparar com Modelo Anterior
```bash
cd "c:\Users\rodri\Desktop\bot brawl"
"C:\Users\rodri\AppData\Local\Programs\Python\Python312\python.exe" -m brawl_bot.training.training_validator --model-path ./models/best_new.pt --compare-with ./models/best_old.pt --output ./comparison_report.json
```

## Configuração do config.json

Adicione estas configurações ao `config.json` para habilitar as novas funcionalidades:

```json
{
  "enable_recording": true,
  "auto_retrain_enabled": true,
  "retrain_triggers": {
    "min_matches": 10,
    "win_rate_threshold": 0.4,
    "min_detection_accuracy": 0.7,
    "max_false_positive_rate": 0.2,
    "decision_accuracy_threshold": 0.6,
    "max_days": 7,
    "min_new_samples": 500
  }
}
```

## Solução de Problemas

### ADB não conecta
```bash
# Reiniciar servidor ADB
"C:\Program Files\BlueStacks_nxt\HD-Adb.exe" kill-server
"C:\Program Files\BlueStacks_nxt\HD-Adb.exe" start-server
"C:\Program Files\BlueStacks_nxt\HD-Adb.exe" connect 127.0.0.1:5555
```

### BlueStacks não detecta janela
- Verifique se o BlueStacks está em tela cheia ou janela normal
- Tente reiniciar o BlueStacks
- Verifique o título da janela no config.json

### Erro ao importar módulos
```bash
# Instalar dependências faltantes
"C:\Users\rodri\AppData\Local\Programs\Python\Python312\python.exe" -m pip install ultralytics opencv-python numpy
```

## Métricas Esperadas

### Antes (dados sintéticos antigos):
- mAP: 19.7%
- Precision: 3.5%
- Recall: 77.0%
- F1 Score: ~6.7%

### Depois (dados reais + sintéticos melhorados):
- mAP: 50-70% (esperado)
- Precision: 60-80% (esperado)
- Recall: 70-85% (esperado)
- F1 Score: 65-80% (esperado)

## Próximos Passos

1. **Iniciar BlueStacks** e abrir Brawl Stars
2. **Aguardar geração de dados sintéticos** completar
3. **Coletar dados reais** por 10-20 minutos
4. **Treinar modelo** com dataset combinado
5. **Validar resultados** e comparar com baseline

## Monitoramento

Durante a coleta de dados, monitore:
- Número de frames capturados
- Distribuição por estado (lobby, game, etc.)
- Frames de alta prioridade (situações interessantes)
- Espaço em disco usado

Logs serão salvos em:
- `./logs/performance/` - Métricas de performance
- `./recordings/` - Gravações de gameplay
- `./dataset/` - Datasets organizados
