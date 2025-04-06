python -m veeksha.prefill_profiler \
--client_config_model "meta-llama/Meta-Llama-3-8B-Instruct" \
--timeout 600 \
--no-prefill_profiler_config_should_train_predictor \
--metrics_config_output_dir "engine_microbenchmark_logs/prefill_sglang_llama-3-8b-tp1" \
--prefill_profiler_config_prefill_lengths 512 1024 2048 4086 8192 \
--metrics_config_should_use_given_dir

python -m veeksha.decode_profiler \
--client_config_model "meta-llama/Meta-Llama-3-8B-Instruct" \
--timeout 600 \
--metrics_config_output_dir "engine_microbenchmark_logs/decode_sglang_llama-3-8b-tp1" \
--decode_profiler_config_context_lengths 512 \
--decode_profiler_config_batch_sizes 1 8 16 32 64 128 \
--metrics_config_should_use_given_dir

python -m veeksha.prefill_profiler \
--client_config_model "meta-llama/Meta-Llama-3-8B-Instruct" \
--timeout 600 \
--no-prefill_profiler_config_should_train_predictor \
--metrics_config_output_dir "engine_microbenchmark_logs/prefill_sglang_llama-3-8b-tp2" \
--prefill_profiler_config_prefill_lengths 512 1024 2048 4086 8192 \
--metrics_config_should_use_given_dir

python -m veeksha.decode_profiler \
--client_config_model "meta-llama/Meta-Llama-3-8B-Instruct" \
--timeout 600 \
--metrics_config_output_dir "engine_microbenchmark_logs/decode_sglang_llama-3-8b-tp2" \
--decode_profiler_config_context_lengths 512 \
--decode_profiler_config_batch_sizes 1 8 16 32 64 128 \
--metrics_config_should_use_given_dir


python -m veeksha.prefill_profiler \
--client_config_model "meta-llama/Meta-Llama-3-8B-Instruct" \
--timeout 600 \
--no-prefill_profiler_config_should_train_predictor \
--metrics_config_output_dir "engine_microbenchmark_logs/prefill_sglang_llama-3-8b-tp4" \
--prefill_profiler_config_prefill_lengths 512 1024 2048 4086 8192 \
--metrics_config_should_use_given_dir

python -m veeksha.decode_profiler \
--client_config_model "meta-llama/Meta-Llama-3-8B-Instruct" \
--timeout 600 \
--metrics_config_output_dir "engine_microbenchmark_logs/decode_sglang_llama-3-8b-tp4" \
--decode_profiler_config_context_lengths 512 \
--decode_profiler_config_batch_sizes 1 8 16 32 64 128 \
--metrics_config_should_use_given_dir


python -m veeksha.prefill_profiler \
--client_config_model "meta-llama/Meta-Llama-3-8B-Instruct" \
--timeout 600 \
--no-prefill_profiler_config_should_train_predictor \
--metrics_config_output_dir "engine_microbenchmark_logs/prefill_sglang_llama-3-8b-tp8" \
--prefill_profiler_config_prefill_lengths 512 1024 2048 4086 8192 \
--metrics_config_should_use_given_dir

python -m veeksha.decode_profiler \
--client_config_model "meta-llama/Meta-Llama-3-8B-Instruct" \
--timeout 600 \
--metrics_config_output_dir "engine_microbenchmark_logs/decode_sglang_llama-3-8b-tp8" \
--decode_profiler_config_context_lengths 512 \
--decode_profiler_config_batch_sizes 1 8 16 32 64 128 \
--metrics_config_should_use_given_dir
