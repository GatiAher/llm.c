#!/usr/bin/env bash
# Sweep over batch size and weight decay for llm.c train_gpt2cu.
# Runs are executed sequentially, one after another.

set -u

BATCH_SIZES=(32768 16384 8192 4096 2048)

# Weight decay values
WD_VALUES=(0.00078125 0.0015625 0.003125 0.00625 0.0125 0.8 1.6 3.2 6.4 12.8)


TOTAL_TOKENS=1638400

mkdir -p my_logs my_logs_output

for batch_size in "${BATCH_SIZES[@]}"; do
    step_size=$(( TOTAL_TOKENS / batch_size ))
    warm_up=$(( step_size / 10 ))   # 0.10 * step_size
    micro_b=$(( batch_size / 1024 ))   # so B*T == total batch size (single GPU)

    for i in "${!WD_VALUES[@]}"; do
        weight_decay=${WD_VALUES[$i]}

	# Format label: '0.05' -> 'p05', '1.6' -> '1p6'
        if [[ "$weight_decay" == 0.* ]]; then
            wd_label="p${weight_decay#0.}"
        else
            wd_label="${weight_decay/./p}"
        fi

        run_name="B${batch_size}_LRp0006_WD_${wd_label}"

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
