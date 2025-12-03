#!/bin/bash
# Run Shinka EPLB experiments with FIXED speed evaluator
# 3 runs GPT-5 + 3 runs Gemini-3 = 6 runs total
# All 6 runs in parallel (3 GPT-5 + 3 Gemini-3 simultaneously)

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

SHINKA_DIR="/home/ubuntu/ShinkaEvolve"
RESULTS_BASE_DIR="/home/ubuntu/new_results/shinka_eplb"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Shinka EPLB - FIXED Speed Evaluator${NC}"
echo -e "${BLUE}6 runs: 3x GPT-5 + 3x Gemini-3 (parallel)${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# ⚠️ CRITICAL: API Key Validation ⚠️
echo -e "${YELLOW}Validating API Keys...${NC}"

API_KEY_VALID=true

if [ -z "${OPENAI_API_KEY}" ]; then
    echo -e "${RED}❌ ERROR: OPENAI_API_KEY is not set${NC}"
    echo "   Please export OPENAI_API_KEY before running:"
    echo -e "   ${YELLOW}export OPENAI_API_KEY='sk-your-openai-key'${NC}"
    API_KEY_VALID=false
else
    echo -e "${GREEN}✓ OPENAI_API_KEY is set${NC}"
    echo "  Key preview: ${OPENAI_API_KEY:0:10}..."
fi

if [ -z "${GEMINI_API_KEY}" ]; then
    echo -e "${RED}❌ ERROR: GEMINI_API_KEY is not set${NC}"
    echo "   Please export GEMINI_API_KEY before running:"
    echo -e "   ${YELLOW}export GEMINI_API_KEY='your-gemini-api-key'${NC}"
    API_KEY_VALID=false
else
    echo -e "${GREEN}✓ GEMINI_API_KEY is set${NC}"
    echo "  Key preview: ${GEMINI_API_KEY:0:10}..."
fi

if [ "$API_KEY_VALID" = false ]; then
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}ABORTING: Required API keys not set${NC}"
    echo -e "${RED}========================================${NC}"
    echo ""
    echo "Quick fix:"
    echo -e "  ${YELLOW}export OPENAI_API_KEY='your-openai-key'${NC}"
    echo -e "  ${YELLOW}export GEMINI_API_KEY='your-gemini-key'${NC}"
    echo -e "  ${YELLOW}./run_shinka_eplb_fixed.sh${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ All required API keys validated${NC}"
echo ""

# Create results directory
mkdir -p "${RESULTS_BASE_DIR}"

cd "${SHINKA_DIR}"

run_experiment() {
    local run_number=$1
    local model_name=$2
    local config_name=$3
    local log_file="${RESULTS_BASE_DIR}/shinka_eplb_${config_name}_run${run_number}_${TIMESTAMP}.log"
    
    echo -e "${GREEN}[Run ${run_number}] Starting ${model_name}...${NC}"
    
    python examples/eplb/run_evo.py \
        evo_config.num_generations=100 \
        evo_config.llm_models="[${model_name}]" \
        evo_config.meta_llm_models="[${model_name}]" \
        evo_config.llm_dynamic_selection='null' \
        results_dir="${RESULTS_BASE_DIR}" \
        exp_name="shinka_eplb_fixed_${config_name}_run${run_number}_${TIMESTAMP}" \
        variant_suffix="" \
        verbose=true \
        > "${log_file}" 2>&1
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✓ Run ${run_number} (${config_name}) completed successfully${NC}"
    else
        echo -e "${RED}✗ Run ${run_number} (${config_name}) failed (exit code: ${exit_code})${NC}"
        echo -e "${YELLOW}  Check log: ${log_file}${NC}"
    fi
    
    return $exit_code
}

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Launching all 6 experiments in parallel${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Array to store PIDs
declare -a PIDS
declare -a RUN_NAMES

# Launch GPT-5 runs (1-3)
echo -e "${YELLOW}Launching GPT-5 runs...${NC}"
run_experiment 1 "gpt-5" "gpt5" &
PIDS+=($!)
RUN_NAMES+=("GPT-5 Run 1")

run_experiment 2 "gpt-5" "gpt5" &
PIDS+=($!)
RUN_NAMES+=("GPT-5 Run 2")

run_experiment 3 "gpt-5" "gpt5" &
PIDS+=($!)
RUN_NAMES+=("GPT-5 Run 3")

# Launch Gemini-3 runs (4-6)
echo -e "${YELLOW}Launching Gemini-3 runs...${NC}"
run_experiment 4 "gemini-3-pro-preview" "gemini3" &
PIDS+=($!)
RUN_NAMES+=("Gemini-3 Run 4")

run_experiment 5 "gemini-3-pro-preview" "gemini3" &
PIDS+=($!)
RUN_NAMES+=("Gemini-3 Run 5")

run_experiment 6 "gemini-3-pro-preview" "gemini3" &
PIDS+=($!)
RUN_NAMES+=("Gemini-3 Run 6")

echo ""
echo -e "${BLUE}All 6 experiments launched. Waiting for completion...${NC}"
echo -e "${BLUE}PIDs: ${PIDS[*]}${NC}"
echo ""

# Wait for all and track results
FAILED=0
for i in "${!PIDS[@]}"; do
    wait ${PIDS[$i]}
    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo -e "${RED}✗ ${RUN_NAMES[$i]} failed${NC}"
        FAILED=$((FAILED + 1))
    else
        echo -e "${GREEN}✓ ${RUN_NAMES[$i]} completed${NC}"
    fi
done

echo ""
echo -e "${GREEN}========================================${NC}"
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All 6 Shinka EPLB experiments completed!${NC}"
else
    echo -e "${YELLOW}Completed with ${FAILED} failures${NC}"
fi
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Results saved in: ${RESULTS_BASE_DIR}"
echo ""
echo "GPT-5 runs:"
echo "  - shinka_eplb_fixed_gpt5_run{1,2,3}_${TIMESTAMP}"
echo ""
echo "Gemini-3 runs:"
echo "  - shinka_eplb_fixed_gemini3_run{4,5,6}_${TIMESTAMP}"
echo ""
echo -e "${YELLOW}NOTE: This uses the FIXED speed evaluator that measures actual execution time${NC}"
echo -e "${YELLOW}      instead of hardcoding speed_score=1.0${NC}"

