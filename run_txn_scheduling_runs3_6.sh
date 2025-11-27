#!/bin/bash
# Run TXN Scheduling experiments - Runs 3-6 (4 new runs)
# Run 3: GPT-5
# Runs 4-6: Gemini-3.0
# Parallel execution (all 4 at once), no UCB dynamic selection

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
echo -e "${BLUE}TXN Scheduling - Runs 3-6${NC}"
echo -e "${BLUE}Run 3: GPT-5 | Runs 4-6: Gemini-3.0${NC}"
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
    echo ""
    echo -e "${YELLOW}⚠️  Gemini experiments (Runs 4-6) require GEMINI_API_KEY${NC}"
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
    echo -e "  ${YELLOW}./run_txn_scheduling_runs3_6.sh${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ All required API keys validated${NC}"
echo ""

cd "${SHINKA_DIR}"

run_experiment() {
    local run_number=$1
    local model_name=$2
    local config_name=$3
    
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}TXN Scheduling Run ${run_number}${NC}"
    echo -e "${GREEN}Model: ${model_name}${NC}"
    echo -e "${GREEN}========================================${NC}"
    
    python examples/txn_scheduling/run_evo.py \
        evo_config.num_generations=100 \
        evo_config.llm_models="[${model_name}]" \
        evo_config.meta_llm_models="[${model_name}]" \
        evo_config.llm_dynamic_selection='null' \
        +job_config.time=600 \
        results_dir="${RESULTS_BASE_DIR}" \
        exp_name="shinka_txn_scheduling_${config_name}_run${run_number}_${TIMESTAMP}" \
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

# Run all 4 experiments in parallel
echo -e "${BLUE}Running all 4 experiments in parallel...${NC}"
echo -e "${BLUE}Run 3: GPT-5 | Runs 4-6: Gemini-3.0${NC}"
echo ""

# Start GPT-5 run
run_experiment 3 "gpt-5" "gpt5" &
PID1=$!

# Start Gemini-3.0 runs
echo -e "${YELLOW}Note: Using GEMINI_API_KEY for Gemini experiments${NC}"
echo "Current GEMINI_API_KEY: ${GEMINI_API_KEY:0:10}..."
echo ""
run_experiment 4 "gemini-3-pro-preview" "gemini3pro" &
PID2=$!
run_experiment 5 "gemini-3-pro-preview" "gemini3pro" &
PID3=$!
run_experiment 6 "gemini-3-pro-preview" "gemini3pro" &
PID4=$!

echo -e "${YELLOW}Waiting for all 4 experiments to complete...${NC}"
wait $PID1
EXIT1=$?
wait $PID2
EXIT2=$?
wait $PID3
EXIT3=$?
wait $PID4
EXIT4=$?

if [ $EXIT1 -ne 0 ] || [ $EXIT2 -ne 0 ] || [ $EXIT3 -ne 0 ] || [ $EXIT4 -ne 0 ]; then
    echo -e "${RED}One or more parallel runs failed${NC}"
    echo -e "${RED}Run 3 (GPT-5): exit code $EXIT1${NC}"
    echo -e "${RED}Run 4 (Gemini): exit code $EXIT2${NC}"
    echo -e "${RED}Run 5 (Gemini): exit code $EXIT3${NC}"
    echo -e "${RED}Run 6 (Gemini): exit code $EXIT4${NC}"
    exit 1
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}All TXN Scheduling runs completed!${NC}"
echo -e "${GREEN}Run 3: GPT-5${NC}"
echo -e "${GREEN}Runs 4-6: Gemini-3.0${NC}"
echo -e "${GREEN}========================================${NC}"

