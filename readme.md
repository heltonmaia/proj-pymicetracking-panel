# Projeto Modularizado com Panel

## Estrutura de Pastas

```
panel/
├── camera_tab/
│   └── camera_tab.py
├── playback_tab/
│   └── playback_tab.py
├── tracking_tab/
│   └── tracking_tab.py
├── ethological_tab/
│   └── ethological_tab.py
├── irl_tab/
│   └── irl_tab.py
├── synthetic_tab/
│   └── synthetic_tab.py
├── experiments/
├── main.py
└── readme.md
```

- Cada pasta de tab contém a lógica, funções e layout de sua respectiva aba.
- O diretório `experiments/` é utilizado por todas as abas para leitura e gravação de arquivos.
- O `main.py` gerencia a interface geral e importa cada aba como um mini-projeto.

## Como rodar

```bash
panel serve main.py --show --autoreload
```
