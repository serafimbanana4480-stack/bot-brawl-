# Comandos Windows/PowerShell - Soberana Omega Bot

Este guia fornece comandos compatíveis com Windows e PowerShell para interagir com a API do bot.

## Informações Importantes

- **Porta do Servidor:** 8003 (não 8000)
- **URL Base:** `http://localhost:8003`
- **PowerShell:** Usa `Invoke-WebRequest` ou `Invoke-RestMethod` em vez de `curl`
- **CMD:** Pode usar `curl` se instalado, mas sintaxe é diferente do Linux

## Opção 1: Usando PowerShell (Recomendado)

### Status do Bot
```powershell
Invoke-RestMethod -Uri "http://localhost:8003/api/brawl-stars/status" -Method Get
```

### Telemetry Completa
```powershell
Invoke-RestMethod -Uri "http://localhost:8003/api/brawl-stars/telemetry" -Method Get
```

### Logs Recentes
```powershell
Invoke-RestMethod -Uri "http://localhost:8003/api/brawl-stars/logs?n=50&category=COMBAT" -Method Get
```

### Diagnostics
```powershell
Invoke-RestMethod -Uri "http://localhost:8003/api/brawl-stars/diagnostics" -Method Get
```

### Performance
```powershell
Invoke-RestMethod -Uri "http://localhost:8003/api/brawl-stars/performance" -Method Get
```

### Emuladores
```powershell
Invoke-RestMethod -Uri "http://localhost:8003/api/brawl-stars/emulators" -Method Get
```

### Auto-Tuning - Executar
```powershell
Invoke-RestMethod -Uri "http://localhost:8003/api/brawl-stars/auto-tuning/tune" -Method Post
```

### Auto-Tuning - Status
```powershell
Invoke-RestMethod -Uri "http://localhost:8003/api/brawl-stars/auto-tuning/status" -Method Get
```

### Auto-Tuning - Reset
```powershell
Invoke-RestMethod -Uri "http://localhost:8003/api/brawl-stars/auto-tuning/reset" -Method Post
```

### Setup do Bot
```powershell
Invoke-RestMethod -Uri "http://localhost:8003/api/brawl-stars/setup" -Method Post
```

### Iniciar Bot
```powershell
Invoke-RestMethod -Uri "http://localhost:8003/api/brawl-stars/start" -Method Post
```

### Parar Bot
```powershell
Invoke-RestMethod -Uri "http://localhost:8003/api/brawl-stars/stop" -Method Post
```

### Configuração
```powershell
# Obter configuração
Invoke-RestMethod -Uri "http://localhost:8003/api/brawl-stars/config" -Method Get

# Atualizar configuração
$body = @{
    trophy_limit = 400
    warning_trophies = 380
    max_session_hours = 3
    min_apm = 20
    max_apm = 60
    auto_stop_on_detection = $true
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8003/api/brawl-stars/config" -Method Post -Body $body -ContentType "application/json"
```

## Opção 2: Usando CMD com curl (Se curl estiver instalado)

### Instalar curl no Windows (se não estiver instalado)
1. Baixar curl: https://curl.se/windows/
2. Extrair para C:\curl
3. Adicionar C:\curl ao PATH

### Comandos curl no Windows

**Importante:** No Windows, não use `-X POST` antes da URL. O método é inferido automaticamente.

### Status do Bot
```cmd
curl http://localhost:8003/api/brawl-stars/status
```

### Telemetry Completa
```cmd
curl http://localhost:8003/api/brawl-stars/telemetry
```

### Logs Recentes
```cmd
curl "http://localhost:8003/api/brawl-stars/logs?n=50&category=COMBAT"
```

### Auto-Tuning - Executar
```cmd
curl -X POST http://localhost:8003/api/brawl-stars/auto-tuning/tune
```

### Auto-Tuning - Status
```cmd
curl http://localhost:8003/api/brawl-stars/auto-tuning/status
```

### Auto-Tuning - Reset
```cmd
curl -X POST http://localhost:8003/api/brawl-stars/auto-tuning/reset
```

## Opção 3: Usando Git Bash ou WSL

Se você tem Git Bash ou WSL instalado, pode usar os comandos curl normais do Linux:

```bash
curl http://localhost:8003/api/brawl-stars/status
curl -X POST http://localhost:8003/api/brawl-stars/auto-tuning/tune
```

## Scripts PowerShell Úteis

### Script para Verificar Status
```powershell
# check_bot_status.ps1
$status = Invoke-RestMethod -Uri "http://localhost:8003/api/brawl-stars/status" -Method Get
Write-Host "Bot Running: $($status.status.running)"
Write-Host "Current Brawler: $($status.status.current_brawler)"
Write-Host "Session Duration: $($status.status.session_duration)"
```

### Script para Executar Auto-Tuning
```powershell
# run_auto_tuning.ps1
$result = Invoke-RestMethod -Uri "http://localhost:8003/api/brawl-stars/auto-tuning/tune" -Method Post
if ($result.success) {
    Write-Host "Auto-tuning executado com sucesso!"
    Write-Host "Win Rate: $($result.analysis.win_rate)"
    Write-Host "Performance Rating: $($result.analysis.performance_rating)"
    Write-Host "Adjustments: $($result.adjustments | ConvertTo-Json)"
} else {
    Write-Host "Erro: $($result.reason)"
}
```

### Script para Monitorar Bot
```powershell
# monitor_bot.ps1
while ($true) {
    $status = Invoke-RestMethod -Uri "http://localhost:8003/api/brawl-stars/status" -Method Get
    Clear-Host
    Write-Host "=== Bot Status ==="
    Write-Host "Running: $($status.status.running)"
    Write-Host "Brawler: $($status.status.current_brawler)"
    Write-Host "Session: $($status.status.session_duration)s"
    Start-Sleep -Seconds 5
}
```

## Inicialização do Servidor

### Usando PowerShell
```powershell
# Navegar para o diretório do bot
cd "C:\Users\rodri\Desktop\soberana\ia ultima\soberana-omega\backend\brawl_bot"

# Iniciar servidor
python api.py
```

### Usando CMD
```cmd
cd C:\Users\rodri\Desktop\soberana\ia ultima\soberana-omega\backend\brawl_bot
python api.py
```

### Usando Script .bat
Execute `start_server.bat` (criado separadamente)

## Troubleshooting

### Erro "Não é possível estabelecer ligação com o servidor remoto"
- **Causa:** Servidor não está rodando
- **Solução:** Iniciar o servidor com `python api.py`

### Erro "A parameter cannot be found that matches parameter name 'X'"
- **Causa:** PowerShell não suporta `-X POST`
- **Solução:** Usar `Invoke-RestMethod` ou remover `-X POST` no CMD

### Erro "Invoke-WebRequest: The remote server returned an error: (404) Not Found"
- **Causa:** Endpoint não existe ou URL incorreta
- **Solução:** Verificar se a porta é 8003 e o endpoint está correto

### Erro "curl: command not found"
- **Causa:** curl não está instalado ou no PATH
- **Solução:** Instalar curl ou usar PowerShell com `Invoke-RestMethod`

## Testes Rápidos

### Testar se servidor está rodando
```powershell
try {
    $response = Invoke-RestMethod -Uri "http://localhost:8003/api/brawl-stars/health" -Method Get
    Write-Host "Servidor está rodando!"
} catch {
    Write-Host "Servidor não está rodando. Inicie com: python api.py"
}
```

### Testar auto-tuning
```powershell
# Ver status
$status = Invoke-RestMethod -Uri "http://localhost:8003/api/brawl-stars/auto-tuning/status" -Method Get
Write-Host "Auto-tuning enabled: $($status.enabled)"

# Executar tuning
$result = Invoke-RestMethod -Uri "http://localhost:8003/api/brawl-stars/auto-tuning/tune" -Method Post
Write-Host "Success: $($result.success)"
```

## Resumo de Endpoints Importantes

| Endpoint | Método | PowerShell | CMD |
|----------|--------|------------|-----|
| Status | GET | `Invoke-RestMethod -Uri "..." -Method Get` | `curl http://...` |
| Telemetry | GET | `Invoke-RestMethod -Uri "..." -Method Get` | `curl http://...` |
| Auto-Tuning Tune | POST | `Invoke-RestMethod -Uri "..." -Method Post` | `curl -X POST http://...` |
| Auto-Tuning Status | GET | `Invoke-RestMethod -Uri "..." -Method Get` | `curl http://...` |
| Auto-Tuning Reset | POST | `Invoke-RestMethod -Uri "..." -Method Post` | `curl -X POST http://...` |
| Setup | POST | `Invoke-RestMethod -Uri "..." -Method Post` | `curl -X POST http://...` |
| Start | POST | `Invoke-RestMethod -Uri "..." -Method Post` | `curl -X POST http://...` |
| Stop | POST | `Invoke-RestMethod -Uri "..." -Method Post` | `curl -X POST http://...` |

## Notas Finais

- **Sempre use a porta 8003** (não 8000)
- **PowerShell é recomendado** no Windows
- **CMD com curl** funciona se curl estiver instalado
- **Git Bash/WSL** permite comandos Linux normais
- **Verifique se o servidor está rodando** antes de testar endpoints
