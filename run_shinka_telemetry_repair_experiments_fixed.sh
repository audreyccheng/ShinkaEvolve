#!/bin/bash

# Fixed version of Shinka Telemetry Repair experiments
# This version includes proper API key validation

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Directory setup
SHINKA_DIR="/home/ubuntu/ShinkaEvolve"
TELEMETRY_DIR="${SHINKA_DIR}/examples/telemetry_repair"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_BASE_DIR="/home/ubuntu/zresults"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}ShinkaEvolve Telemetry Repair Experiments (FIXED)${NC}"
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
    # Show first 10 characters for verification
    echo "  Key preview: ${OPENAI_API_KEY:0:10}..."
fi

if [ -z "${GEMINI_API_KEY}" ]; then
    echo -e "${RED}❌ ERROR: GEMINI_API_KEY is not set${NC}"
    echo "   Please export GEMINI_API_KEY before running:"
    echo -e "   ${YELLOW}export GEMINI_API_KEY='your-gemini-api-key'${NC}"
    echo ""
    echo -e "${YELLOW}⚠️  This is the issue that caused previous Gemini experiments to fail!${NC}"
    echo "   Shinka code explicitly requires GEMINI_API_KEY environment variable."
    echo "   See: SHINKA_GEMINI_FAILURE_ANALYSIS_AND_FIX.md for details"
    API_KEY_VALID=false
else
    echo -e "${GREEN}✓ GEMINI_API_KEY is set${NC}"
    # Show first 10 characters for verification
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
    echo -e "  ${YELLOW}./run_shinka_telemetry_repair_experiments_fixed.sh${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ All required API keys validated${NC}"
echo ""

echo "Results will be saved to: ${RESULTS_BASE_DIR}/shinka_telemetry_repair_*_run*_${TIMESTAMP}"
echo ""

# Function to run a single experiment
run_experiment() {
    local config_name=$1
    local run_number=$2
    local model_name=$3
    
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Starting Run ${run_number} with ${config_name}${NC}"
    echo -e "${GREEN}Model: ${model_name}${NC}"
    echo -e "${GREEN}========================================${NC}"
    
    cd "${SHINKA_DIR}"
    
    # Run ShinkaEvolve with Hydra overrides
    python examples/telemetry_repair/run_evo.py \
        evo_config.num_generations=100 \
        evo_config.llm_models="[${model_name}]" \
        evo_config.llm_dynamic_selection=null \
        results_dir="${RESULTS_BASE_DIR}" \
        exp_name="shinka_telemetry_repair_${config_name}_run${run_number}_${TIMESTAMP}" \
        variant_suffix="" \
        verbose=true
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✓ Run ${run_number} with ${config_name} completed successfully${NC}"
    else
        echo -e "${RED}✗ Run ${run_number} with ${config_name} failed (exit code: ${exit_code})${NC}"
    fi
    
    echo ""
    return $exit_code
}

# Run 3 experiments with Gemini-3-pro (sequential) - RUNNING FIRST
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Phase 1: Gemini-3-pro Configuration (3 runs, sequential)${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# ⚠️ IMPORTANT: Do NOT reassign OPENAI_API_KEY for Gemini
# The old script did: export OPENAI_API_KEY="${GEMINI_API_KEY}"
# This doesn't work because Shinka code explicitly looks for GEMINI_API_KEY

echo -e "${YELLOW}Note: GEMINI_API_KEY must be set for Gemini experiments${NC}"
echo "Current GEMINI_API_KEY: ${GEMINI_API_KEY:0:10}..."
echo ""

# Run all 3 experiments sequentially
run_experiment "gemini3pro" "1" "gemini-3-pro-preview"
if [ $? -ne 0 ]; then
    echo -e "${RED}Gemini run 1 failed${NC}"
    exit 1
fi

run_experiment "gemini3pro" "2" "gemini-3-pro-preview"
if [ $? -ne 0 ]; then
    echo -e "${RED}Gemini run 2 failed${NC}"
    exit 1
fi

run_experiment "gemini3pro" "3" "gemini-3-pro-preview"
if [ $? -ne 0 ]; then
    echo -e "${RED}Gemini run 3 failed${NC}"
    exit 1
fi

# Run 3 experiments with GPT-5 (sequential)
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Phase 2: GPT-5 Configuration (3 runs, sequential)${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Run all 3 experiments sequentially
run_experiment "gpt5" "1" "gpt-5"
if [ $? -ne 0 ]; then
    exit 1
fi

run_experiment "gpt5" "2" "gpt-5"
if [ $? -ne 0 ]; then
    exit 1
fi

run_experiment "gpt5" "3" "gpt-5"
if [ $? -ne 0 ]; then
    exit 1
fi

# Summary
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}All 6 ShinkaEvolve Telemetry Repair experiments completed!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Results saved in: ${RESULTS_BASE_DIR}/"
echo ""
echo "Summary:"
echo "  - Gemini-3-pro runs: ${RESULTS_BASE_DIR}/shinka_telemetry_repair_gemini3pro_run{1,2,3}_${TIMESTAMP}"
echo "  - GPT-5 runs: ${RESULTS_BASE_DIR}/shinka_telemetry_repair_gpt5_run{1,2,3}_${TIMESTAMP}"
echo ""
echo -e "${GREEN}✓ Experiments completed successfully without GEMINI_API_KEY errors${NC}"
echo ""

# Validation check
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Post-Experiment Validation${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if program files were generated for Gemini runs
GEMINI_PROGRAMS=$(find "${RESULTS_BASE_DIR}" -name "shinka_telemetry_repair_gemini3pro_run*_${TIMESTAMP}" -type d -exec find {} -name "main.py" -path "*/gen_*/main.py" \; | wc -l)

echo "Gemini program files generated: ${GEMINI_PROGRAMS}"

if [ $GEMINI_PROGRAMS -ge 300 ]; then
    echo -e "${GREEN}✓ Gemini experiments appear successful (expected ~300 files, got ${GEMINI_PROGRAMS})${NC}"
else
    echo -e "${YELLOW}⚠️  Warning: Expected ~300 program files, but found only ${GEMINI_PROGRAMS}${NC}"
    echo "   This might indicate some generations failed to produce programs."
fi

echo ""
echo "Next steps:"
echo "  1. Analyze results: python analyze_telemetry_results.py"
echo "  2. Compare to OpenEvolve: see telemetry_repair_comparison.md"
echo "  3. Check convergence patterns in best/ directories"
echo ""

