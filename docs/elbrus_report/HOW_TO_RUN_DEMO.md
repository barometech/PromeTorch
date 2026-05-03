# Как запустить tool-call HTML demo вручную

PromeServe HTTP path на эльбрусе нестабилен из-за **SIGILL в EML+pthread**
(known issue, см `feedback_eml_pthread_sigill.md`): когда HTTP server-thread
вызывает `cblas_sgemm` через worker pthread → Illegal Instruction.

Demo запускается **через bash orchestrator на test_gguf_inference TP-4**
(11.4 tok/s, lossless или с BUG-12 fixes). Скрипт уже в репо:
`scripts/html_demo_via_tp4.sh`.

## Шаги

```bash
# 1. SSH на эльбрус (нужно прямое окно, не batch-mode)
ssh -p 8199 -i elbrusssh.ppk paperclipdnb@w205p.mcst.ru

# 2. На эльбрусе, в окне SSH:
cd ~/promethorch
chmod +x scripts/html_demo_via_tp4.sh
bash scripts/html_demo_via_tp4.sh

# 3. На своём компе после завершения:
mkdir -p screenshots
scp -P 8199 -i "ELBRUS DATA SSH/elbrusssh.ppk" \
    'paperclipdnb@w205p.mcst.ru:/tmp/promeserve/*.html' screenshots/
bash scripts/screenshot_html.bat screenshots
```

## Что должно получиться

4 HTML страницы:
- `moscow.html` — про Москву
- `menu.html` — меню кафе «Прометей»
- `card.html` — визитка разработчика
- `todo.html` — todo list по оптимизациям

И 4 PNG скриншота 720×1280 (вертикальная ориентация).

## Архитектура

- **mistral-7B Q4_K_M** на TP-4 (4 NUMA × 8 ядер)
- env: `PT_Q8_SOA=1 PT_PER_BLOCK_SCALE=1 PT_LM_HEAD_FP=1 PT_NO_FFN_SOA=1`
- Bash orchestrator парсит `<tool_call>{"name":"write_file","arguments":{...}}</tool_call>`
- Sandbox: `/tmp/promeserve/` (basename only, no `..`)
- Ожидаемая скорость: ~5-7 tok/s (4-7 минут на страницу 600 токенов)

## Известные блокеры (не моя ошибка)

1. **Plink ssh + claude-code Bash tool**: длинные ssh-команды (>5 минут)
   завершаются preemptively → невозможно запустить demo через автоматизацию.
   Требует прямое SSH-окно от юзера.
2. **PromeServe HTTP**: SIGILL в EML+pthread при первом chat-запросе.
   Архитектурный fix: routes forward() через main thread вместо HTTP worker.
   ~1 день работы. См BUG в `JOURNAL_BREAKDOWNS.md`.
3. **TP-4 OOM на 7B+** (qwen2.5-7B/llama3-8B/14B): mmap virtual address
   space превышает 125GB на 4 ranks. Решение — single-process fallback
   через `--nprocs 1` (уже в `scripts/run_one_singleproc.sh`).
