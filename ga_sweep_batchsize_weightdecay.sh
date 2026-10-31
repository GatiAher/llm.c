#!/usr/bin/env bash
# Sweep over batch size and weight decay for llm.c train_gpt2cu.
# Runs are executed sequentially, one after another.

set -u

BATCH_SIZES=(32768 16384 8192 4096 2048)

# Weight decay values and the matching label used in the run name
# (decimal point replaced by "p", leading zero dropped, e.g. 0.025 -> p025)
WD_VALUES=(0.025 0.05 0.1 0.2 0.4)
WD_LABELS=(025   05   1   2   4)

TOTAL_TOKENS=1638400

mkdir -p my_logs my_logs_output

for batch_size in "${BATCH_SIZES[@]}"; do
    step_size=$(( TOTAL_TOKENS / batch_size ))
    warm_up=$(( step_size / 10 ))   # 0.10 * step_size
    micro_b=$(( batch_size / 1024 ))   # so B*T == total batch size (single GPU)

    for i in "${!WD_VALUES[@]}"; do
        weight_decay=${WD_VALUES[$i]}
        wd_label=${WD_LABELS[$i]}
        run_name="B${batch_size}_LRp0006_WD_p${wd_label}"

        echo "=== ${run_name} | steps=${step_size} warmup=${warm_up} wd=${weight_decay} ==="

        mkdir -p "my_logs/${run_name}"

        ./train_gpt2cu \
            -i "/home/ubuntu/mydata/fineweb10B/fineweb_train_*.bin" \
            -j "/home/ubuntu/mydata/fineweb10B/fineweb_val_*.bin" \
            -o "my_logs/${run_name}" \
            -e "my_models/mup_base_1l_64w.bin" \
            -b "${micro_b}" -t 1024 \
            -d "${batch_size}" \
            -r 1 -z 1 \
            -v 5 \
            -s 0 \
            -x "${step_size}" \
            -u "${warm_up}" \
            -l 0.0006 \
            -c "${weight_decay}" \
            -me 1 \
            -mm 1 \
            -mw 64 &> "my_logs_output/${run_name}.txt"

        status=$?
        if [ $status -ne 0 ]; then
            echo "    WARNING: ${run_name} exited with status ${status} (see my_logs_output/${run_name}.txt)"
        fi
    done
done

echo "Sweep complete."
