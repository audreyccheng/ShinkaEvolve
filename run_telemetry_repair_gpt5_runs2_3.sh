#!/bin/bash
# Run Telemetry Repair GPT-5 experiments - Runs 2 & 3
# Both runs in parallel

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

SHINKA_DIR="/home/ubuntu/ShinkaEvolve"
RESULTS_BASE_DIR="/home/ubuntu/zresults"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Telemetry Repair - GPT-5 Runs 2 & 3${NC}"
echo -e "${BLUE}Running in parallel${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# API Key Validation
echo -e "${YELLOW}Validating API Keys...${NC}"

if [ -z "${OPENAI_API_KEY}" ]; then
    echo -e "${RED}❌ ERROR: OPENAI_API_KEY is not set${NC}"
    echo "   Please export OPENAI_API_KEY before running:"
    echo -e "   ${YELLOW}export OPENAI_API_KEY='sk-your-openai-key'${NC}"
    exit 1
else
    echo -e "${GREEN}✓ OPENAI_API_KEY is set${NC}"
    echo "  Key preview: ${OPENAI_API_KEY:0:10}..."
fi

echo ""
echo -e "${GREEN}✓ All required API keys validated${NC}"
echo ""

cd "${SHINKA_DIR}"

run_experiment() {
    local run_number=$1
    
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Telemetry Repair GPT-5 Run ${run_number}${NC}"
    echo -e "${GREEN}========================================${NC}"
    
    python examples/telemetry_repair/run_evo.py \
        evo_config.num_generations=100 \
        evo_config.llm_models='["gpt-5"]' \
        evo_config.meta_llm_models='["gpt-5"]' \
        evo_config.llm_dynamic_selection=null \
        +job_config.time="00:10:00" \
        results_dir="${RESULTS_BASE_DIR}" \
        exp_name="shinka_telemetry_repair_gpt5_run${run_number}_${TIMESTAMP}" \
        variant_suffix="" \
        verbose=true
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✓ Run ${run_number} completed successfully${NC}"
    else
        echo -e "${RED}✗ Run ${run_number} failed (exit code: ${exit_code})${NC}"
        return $exit_code
    fi
    echo ""
}

# Run both experiments in parallel
echo -e "${BLUE}Running GPT-5 Runs 2 & 3 in parallel...${NC}"
echo ""

run_experiment 2 &
PID1=$!
run_experiment 3 &
PID2=$!

echo -e "${YELLOW}Waiting for both experiments to complete...${NC}"
wait $PID1
EXIT1=$?
wait $PID2
EXIT2=$?

if [ $EXIT1 -ne 0 ] || [ $EXIT2 -ne 0 ]; then
    echo -e "${RED}One or more parallel runs failed${NC}"
    echo -e "${RED}Run 2: exit code $EXIT1${NC}"
    echo -e "${RED}Run 3: exit code $EXIT2${NC}"
    exit 1
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}All Telemetry Repair GPT-5 runs completed!${NC}"
echo -e "${GREEN}Runs 2 & 3: GPT-5${NC}"
echo -e "${GREEN}========================================${NC}"

