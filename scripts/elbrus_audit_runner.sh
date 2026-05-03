#!/bin/bash
# Wrapper to run full_russian_audit.sh in background reliably.
cd /home/paperclipdnb/promethorch
pkill -9 -f test_gguf_inference 2>/dev/null
sleep 3
rm -f /tmp/audit2.done
bash scripts/full_russian_audit.sh > /tmp/audit2.log 2>&1
echo DONE > /tmp/audit2.done
