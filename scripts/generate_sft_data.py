#!/usr/bin/env python3
"""
CLI entrypoint for SFT data generation via vLLM.

Usage:
    python scripts/generate_sft_data.py \
        --input prompts.jsonl \
        --output sft_data/ \
        --endpoint http://localhost:8000/v1 \
        --model Qwen/Qwen3-32B \
        --n-completions 8 \
        --domains solidity sql math \
        --resume

Three-tier config: defaults → JSON config file → CLI overrides.
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rlvr.sft_generator import SFTGenerationConfig, SFTGenerator

VALID_DOMAINS = ["solidity", "sql", "math", "sec_finance", "english"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("generate_sft_data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate verified SFT training data via vLLM teacher models."
    )

    # I/O
    parser.add_argument("--input", required=True, help="Input JSONL file with prompts")
    parser.add_argument("--output", default="sft_data", help="Output directory (default: sft_data)")

    # Config file
    parser.add_argument("--config", help="JSON config file (overrides defaults, overridden by CLI)")

    # vLLM endpoint
    parser.add_argument("--endpoint", default=None, help="Default vLLM endpoint URL")
    parser.add_argument("--model", default=None, help="Model name/path")
    parser.add_argument("--api-key", default=None, help="API key (default: EMPTY)")

    # Per-domain endpoints
    for domain in VALID_DOMAINS:
        parser.add_argument(
            f"--endpoint-{domain.replace('_', '-')}",
            default=None,
            help=f"vLLM endpoint for {domain} domain",
        )

    # Generation params
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--n-completions", type=int, default=None, help="Best-of-N budget per prompt")
    parser.add_argument("--no-early-stop", action="store_true", help="Disable early stopping on first verified")
    parser.add_argument("--reward-threshold", type=float, default=None, help="Default reward threshold")
    parser.add_argument("--batch-size", type=int, default=None, help="Concurrent workers")
    parser.add_argument("--max-retries", type=int, default=None)

    # Filtering
    parser.add_argument("--domains", nargs="+", choices=VALID_DOMAINS, help="Only process these domains")

    # Resume / dry-run
    parser.add_argument("--resume", action="store_true", help="Resume from previous progress")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh (ignore progress)")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and connectivity only")

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> SFTGenerationConfig:
    """Build config from three tiers: defaults → JSON file → CLI overrides."""

    # Tier 1: defaults
    config = SFTGenerationConfig()

    # Tier 2: JSON config file
    if args.config:
        with open(args.config, "r") as f:
            config = SFTGenerationConfig.from_json(f.read())

    # Tier 3: CLI overrides
    if args.endpoint is not None:
        config.default_endpoint = args.endpoint
    if args.model is not None:
        config.model_name = args.model
    if args.api_key is not None:
        config.api_key = args.api_key
    if args.max_tokens is not None:
        config.max_tokens = args.max_tokens
    if args.temperature is not None:
        config.temperature = args.temperature
    if args.top_p is not None:
        config.top_p = args.top_p
    if args.n_completions is not None:
        config.n_completions = args.n_completions
    if args.no_early_stop:
        config.early_stop_on_verified = False
    if args.reward_threshold is not None:
        config.default_reward_threshold = args.reward_threshold
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.max_retries is not None:
        config.max_retries = args.max_retries
    if args.no_resume:
        config.enable_resume = False
    elif args.resume:
        config.enable_resume = True

    config.output_dir = args.output

    # Per-domain endpoints
    domain_endpoints = config.domain_endpoints or {}
    for domain in VALID_DOMAINS:
        cli_key = f"endpoint_{domain}"
        val = getattr(args, cli_key, None)
        if val is not None:
            domain_endpoints[domain] = val
    if domain_endpoints:
        config.domain_endpoints = domain_endpoints

    return config


def dry_run(config: SFTGenerationConfig, input_path: str) -> None:
    """Validate config and optionally test connectivity."""
    logger.info("=== DRY RUN ===")
    logger.info(f"Config:\n{config.to_json()}")

    # Validate input file
    if input_path == "/dev/null" or not os.path.exists(input_path):
        logger.info(f"Input file: {input_path} (empty or missing — OK for dry run)")
        prompt_count = 0
    else:
        prompt_count = 0
        domains_seen = set()
        with open(input_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        ex = json.loads(line)
                        prompt_count += 1
                        domains_seen.add(ex.get("domain", "unknown"))
                    except json.JSONDecodeError:
                        pass
        logger.info(f"Input: {prompt_count} prompts, domains: {domains_seen}")

    # Test verifier creation
    for domain in VALID_DOMAINS:
        try:
            from src.rlvr.domain_verifiers import create_verifier
            v = create_verifier(domain)
            logger.info(f"  Verifier [{domain}]: OK ({v.name})")
        except Exception as e:
            logger.warning(f"  Verifier [{domain}]: FAILED ({e})")

    logger.info("=== DRY RUN COMPLETE ===")


def main():
    args = parse_args()
    config = build_config(args)

    if args.dry_run:
        dry_run(config, args.input)
        return

    # Validate input exists
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    logger.info(f"Starting SFT generation")
    logger.info(f"  Input: {args.input}")
    logger.info(f"  Output: {config.output_dir}")
    logger.info(f"  Model: {config.model_name or '(default)'}")
    logger.info(f"  N-completions: {config.n_completions}")
    logger.info(f"  Early stop: {config.early_stop_on_verified}")
    logger.info(f"  Resume: {config.enable_resume}")

    generator = SFTGenerator(config)

    try:
        output_path = generator.generate_from_jsonl(args.input)
        stats = generator.get_stats()

        logger.info("=== GENERATION COMPLETE ===")
        logger.info(f"  Output: {output_path}")
        logger.info(f"  Total: {stats['total_prompts']}")
        logger.info(f"  Verified: {stats['verified']} ({stats['rate']:.1%})")
        if stats["per_domain"]:
            for domain, ds in stats["per_domain"].items():
                logger.info(f"  [{domain}] {ds['verified']}/{ds['total']} ({ds['rate']:.1%})")
    except KeyboardInterrupt:
        stats = generator.get_stats()
        logger.info("Interrupted. Partial stats:")
        logger.info(f"  Verified: {stats['verified']}/{stats['total_prompts']}")


if __name__ == "__main__":
    main()
